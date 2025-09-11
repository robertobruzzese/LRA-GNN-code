# training/train_rl_flat.py
#!/usr/bin/env python3
import os
import math
import time
import torch
from collections import Counter
from datetime import datetime
from tqdm.auto import tqdm

try:
    from torch_geometric.data import Data
except Exception:
    Data = None

from training.rl_environment import RLEnvironment


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _as_1d_tensor(x):
    """Assicura tensore 1D (512,). Se 2D (N,512) fa mean-pool, se già 1D lo ritorna."""
    if not torch.is_tensor(x):
        raise TypeError(f"atteso torch.Tensor, trovato {type(x)}")
    if x.ndim == 2:
        return x.mean(dim=0)
    if x.ndim == 1:
        return x
    raise ValueError(f"Embedding con ndim non supportato: {x.ndim}")


def _extract_emb_age(batch, device):
    """
    Restituisce (emb[B,512], age[B]) a partire da:
    - (embedding, age)    # nuovo formato
    - Data con .x e .y    # formato legacy (grafo)
    Supporta anche batch con B>1.
    """
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        emb, age = batch
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)  # [1,512]
        age = age.view(-1)          # [B]
        return emb.to(device).float(), age.to(device).float()

    if (Data is not None) and isinstance(batch, Data):
        # Graph → pooling su nodi se serve
        emb1d = _as_1d_tensor(batch.x)
        age1d = batch.y.view(-1)
        return emb1d.unsqueeze(0).to(device).float(), age1d.to(device).float()

    # Alcuni DataLoader potrebbero batching di Data in liste
    if isinstance(batch, list) and batch and ((Data is not None) and isinstance(batch[0], Data)):
        embs, ages = [], []
        for g in batch:
            emb1d = _as_1d_tensor(g.x)
            age1d = g.y.view(-1)
            embs.append(emb1d)
            ages.append(age1d)
        return torch.stack(embs, 0).to(device).float(), torch.cat(ages, 0).to(device).float()

    raise ValueError("Batch non riconosciuto: attesi (embedding, age) oppure Data con .x/.y")


def _buffer_dataset(dataloader, device):
    """
    Carica tutto il dataloader in memoria come:
      X: [N,512]  y: [N]
    Così evitiamo I/O ripetuto ad ogni episodio.
    """
    Xs, Ys = [], []
    for batch in tqdm(dataloader, desc="Buffering dataset", leave=False):
        embB, ageB = _extract_emb_age(batch, device="cpu")  # buffer su CPU
        if embB.ndim == 1:
            embB = embB.unsqueeze(0)
        Xs.append(embB)          # [B,512]
        Ys.append(ageB.view(-1)) # [B]
    X = torch.cat(Xs, dim=0) if Xs else torch.empty(0, 512)
    y = torch.cat(Ys, dim=0) if Ys else torch.empty(0)
    # Sposta su device solo quando necessario nel loop
    return X, y


def _age_to_targets(age_scalar):
    """Da età → (gi, ei): decade e unità (clamp 0..9 per sicurezza)."""
    a = int(float(age_scalar))
    gi = max(0, min(9, a // 10))
    ei = max(0, min(9, a % 10))
    return gi, ei


# ------------------------------------------------------------
# Train PRLAE (flat/graph tolerant)
# ------------------------------------------------------------
def train_prlae(
    agent,
    dataloader,
    device,
    dataset_name: str,
    num_episodes: int = 50,
    start_episode: int = 0,
    save_every: int = 10,
    best_accuracy: float = 0.0,
    max_steps_per_episode: int = 100,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
):
    """
    addestra l'agente su flat embeddings (512D) o grafi (compatibile).
    Richiede che agent.classifier sia stato impostato (MLP a decadi).
    """
    # 1) Safety: serve il classificatore
    classifier = getattr(agent, "classifier", None)
    if classifier is None:
        raise ValueError("❌ Classificatore non fornito (agent.classifier è None)")

    classifier.eval()

    # 2) Buffer dataset (X:[N,512], y:[N])
    X_cpu, y_cpu = _buffer_dataset(dataloader, device)
    N = X_cpu.size(0)
    if N == 0:
        raise RuntimeError("Dataset vuoto: nessun campione caricato dal dataloader.")

    # 3) Conteggi gruppi d'età (decadi)
    age_groups = [int(float(a)) // 10 for a in y_cpu.tolist()]
    counts = Counter(age_groups)
    print("\n📊 Conteggio per gruppi di età:")
    for g in range(10):
        if counts.get(g, 0) > 0:
            print(f"  Gruppo {g}: {counts[g]} campioni")
    majority_class_count = max(counts.values()) if counts else 1
    print(f"📌 Majority class count = {majority_class_count}")

    # 4) Directory dei checkpoint parziali
    ckpt_dir = os.path.join("checkpoints", dataset_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 5) Epsilon schedule (lineare)
    # 5) Epsilon schedule (globale, esponenziale sul numero di episodio assoluto)
    def _epsilon_abs(abs_ep_idx, epsilon_end=0.05, decay=0.98):
        return max(epsilon_end, decay ** abs_ep_idx)

    # 6) Training loop
    env = RLEnvironment()
    total_seen_global = 0

    for ep in range(start_episode, start_episode + num_episodes):
        # ep = indice assoluto (continua a crescere anche dopo una ripartenza)
        eps = _epsilon_abs(ep, epsilon_end=epsilon_end, decay=0.98)
        ep_local = ep - start_episode  # 0..num_episodes-1, solo per visual
        print(f"[EP ABS {ep + 1} | LOCAL {ep_local + 1}/{num_episodes}] Epsilon: {eps:.4f}")

        # shuffle indices ad ogni episodio
        perm = torch.randperm(N)

        ep_total_reward = 0.0
        ep_total_correct = 0
        ep_only_row_correct = 0
        ep_total_correct_decade = 0
        ep_total_samples = 0

        # ciclo campioni
        pbar = tqdm(perm.tolist(), desc=f"Episode {ep - start_episode + 1}/{num_episodes}", leave=False)
        for idx in pbar:
            # sample corrente (su device)
            xi = X_cpu[idx].unsqueeze(0).to(device).float()  # [1,512]
            yi = y_cpu[idx].to(device).float()               # []

            gi, ei = _age_to_targets(yi.item())
            Ngi = counts.get(gi, 1)
            NgM = majority_class_count
            imbalance_ratio = min(NgM / max(Ngi, 1), 1e5)

            # riga iniziale dal classificatore (decade stimata)
            with torch.no_grad():
                logits = classifier(xi)              # [1,10]
                start_row = int(torch.argmax(logits, dim=1).item())

            # reset env
            env.reset(
                x=xi,
                target_row=gi,
                target_col=ei,
                imbalance_ratio=imbalance_ratio,
                actual_age=float(yi.item()),
                start_row=start_row,
            )

            reached_target = False
            step_reward_accum = 0.0

            for step in range(max_steps_per_episode):
                # --- STATO PULITO: [embedding(512), pos_r(1), pos_c(1)] -> 512 + 2 = 514 ---
                pos_tensor   = torch.tensor([[env.r / 10.0, env.c / 10.0]], device=device)
                state_tensor = torch.cat([xi, pos_tensor], dim=1)

                # (facoltativo) verifica una sola volta per episodio
                if step == 0:
                    assert state_tensor.shape[1] == agent.q_network.fc1.in_features, \
                        f"State dim mismatch: got {state_tensor.shape[1]}, expected {agent.q_network.fc1.in_features}"

                # azione
                action = agent.select_action(state_tensor, epsilon=eps)

                # step ambiente
                prev_r, prev_c = env.r, env.c
                _, reward, done = env.step(action)

                # reward shaping lecito (la policy NON vede il target, lo usa solo la reward)
                dist_prev = abs(prev_r - gi) + abs(prev_c - ei)
                dist_now  = abs(env.r - gi) + abs(env.c - ei)
                reward += (dist_prev - dist_now) * 0.5

                # bonus goal
                if env.r == gi and env.c == ei:
                    reward += 20.0
                    reached_target = True
                    done = True

                # --- NEXT STATE con lo stesso schema pulito ---
                pos_tensor2 = torch.tensor([[env.r / 10.0, env.c / 10.0]], device=device)
                next_state  = torch.cat([xi, pos_tensor2], dim=1)

                
                # update Q
                loss = agent.update(state_tensor, action, reward, next_state, done)
                step_reward_accum += float(reward)  

                if done:
                    break

            # metriche per sample
            ep_total_samples += 1
            ep_total_reward += step_reward_accum
            if reached_target:
                ep_total_correct += 1
            if env.r == gi:
                ep_total_correct_decade += 1
                if env.c != ei:
                    ep_only_row_correct += 1

            total_seen_global += 1

        # fine episodio: stampa metriche
        acc = 100.0 * ep_total_correct / max(1, ep_total_samples)
        acc_dec = 100.0 * ep_total_correct_decade / max(1, ep_total_samples)
        only_row_acc = 100.0 * ep_only_row_correct / max(1, ep_total_samples)
        avg_reward = ep_total_reward / max(1, ep_total_samples)

        print(
            f"📊 EP {ep + 1}: "
            f"acc={acc:.2f}% | decade_acc={acc_dec:.2f}% | "
            f"only_row_acc={only_row_acc:.2f}% | avg_reward={avg_reward:.3f}"
        )

        # salva parziale a intervallo oppure se migliora best_accuracy
       # --- salvataggio parziale ogni save_every episodi ---
        if (ep + 1) % save_every == 0:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out = os.path.join(ckpt_dir, f"rl_agent_partial_{ep + 1}_{ts}.pth")
            agent.save(out)
            print(f"💾 Salvato checkpoint parziale: {out}")

        # --- salvataggio del best se migliora l'accuracy ---
        if acc > best_accuracy:
            best_accuracy = acc
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            best = os.path.join(ckpt_dir, f"best_agent_{ts}.pth")
            agent.save(best)
            print(f"🏆 Nuovo best ({best_accuracy:.2f}%) salvato: {best}")

    return best_accuracy