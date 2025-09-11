# training/train_rl_flat_x_mae.py
#!/usr/bin/env python3
import os
import math
import time
import random
import numpy as np
import torch
from collections import Counter
from datetime import datetime
from tqdm.auto import tqdm
import importlib, json
# ---- MONKEY PATCH: abilita extra=… su tutte le ProgressiveRLAgent note ----
import importlib, json
import torch

def _patched_save(self, path, episode: int = None, **kwargs):
    state = {
        "q_network": self.q_network.state_dict(),
        "target_network": self.target_network.state_dict() if hasattr(self, "target_network") else None,
        "optimizer": self.optimizer.state_dict() if hasattr(self, "optimizer") else None,
        "epsilon": getattr(self, "epsilon", None),
        "gamma": getattr(self, "gamma", None),
        "episode": episode,
        "config": getattr(self, "config", None),
    }
    torch.save(state, path)
    extra = kwargs.get("extra")
    if extra is not None:
        sidecar = {"episode": episode}
        sidecar.update(extra if isinstance(extra, dict) else {"info": str(extra)})
        try:
            with open(path + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(sidecar, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

# Prova a patchare tutte le possibili definizioni
_candidate_modules = [
    "models.progressive_rl_no_ablation_x_mae",
    "models.progressive_rl_no_ablation",
    "models.progressive_rl_ablation",
    "models.progressive_rl_ablation_utkface",
    "models.progressive_rl",
]

for _mod in _candidate_modules:
    try:
        m = importlib.import_module(_mod)
        if hasattr(m, "ProgressiveRLAgent"):
            m.ProgressiveRLAgent.save = _patched_save
    except Exception:
        pass
# ---- FINE MONKEY PATCH ----

try:
    from torch_geometric.data import Data
except Exception:
    Data = None

from training.rl_environment import RLEnvironment


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _as_1d_tensor(x):
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
    - (embedding, age)    # formato flat
    - Data con .x e .y    # formato grafi (compatibile)
    """
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        emb, age = batch
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)  # [1,512]
        age = age.view(-1)
        return emb.to(device).float(), age.to(device).float()

    if (Data is not None) and isinstance(batch, Data):
        emb1d = _as_1d_tensor(batch.x)
        age1d = batch.y.view(-1)
        return emb1d.unsqueeze(0).to(device).float(), age1d.to(device).float()

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
    """Carica tutto su CPU: X:[N,512], y:[N]."""
    Xs, Ys = [], []
    for batch in tqdm(dataloader, desc="Buffering dataset", leave=False):
        embB, ageB = _extract_emb_age(batch, device="cpu")  # buffer su CPU
        if embB.ndim == 1:
            embB = embB.unsqueeze(0)
        Xs.append(embB)
        Ys.append(ageB.view(-1))
    X = torch.cat(Xs, dim=0) if Xs else torch.empty(0, 512)
    y = torch.cat(Ys, dim=0) if Ys else torch.empty(0)
    return X, y


def _age_to_targets(age_scalar):
    """Età → (gi, ei): decade e unità (clamp 0..9)."""
    a = int(float(age_scalar))
    gi = max(0, min(9, a // 10))
    ei = max(0, min(9, a % 10))
    return gi, ei


# ------------------------------------------------------------
# Train PRLAE (flat/graph tolerant) con reward allineato al MAE
# ------------------------------------------------------------
def train_prlae(
    agent,
    dataloader,
    device,
    dataset_name: str,
    num_episodes: int = 50,
    start_episode: int = 0,
    save_every: int = 1,          # salva ad ogni episodio
    best_accuracy: float = 0.0,
    best_mae: float = float("inf"),
    max_steps_per_episode: int = 100,
    epsilon_end: float = 0.05,
    eps_decay: float = 0.98,
    lambda_mae: float = 0.5,      # peso della penalità MAE nel reward
    mae_scale: float = 10.0,      # normalizzazione per la penalità in anni
):
    """
    Addestra l'agente. Il reward è:
      reward_base (env + shaping distanza)
      - lambda_mae * |errore_età| / mae_scale
    e si logga/salva anche il best per MAE di episodio.
    """
    # 1) Serve il classificatore
    classifier = getattr(agent, "classifier", None)
    if classifier is None:
        raise ValueError("❌ Classificatore non fornito (agent.classifier è None)")
    classifier.eval()

    # 2) Buffer dataset
    X_cpu, y_cpu = _buffer_dataset(dataloader, device)
    N = X_cpu.size(0)
    if N == 0:
        raise RuntimeError("Dataset vuoto: nessun campione caricato dal dataloader.")

    # 3) Conteggi gruppi d'età
    age_groups = [int(float(a)) // 10 for a in y_cpu.tolist()]
    counts = Counter(age_groups)
    print("\n📊 Conteggio per gruppi di età:")
    for g in range(10):
        if counts.get(g, 0) > 0:
            print(f"  Gruppo {g}: {counts[g]} campioni")
    majority_class_count = max(counts.values()) if counts else 1
    print(f"📌 Majority class count = {majority_class_count}")

    # 4) Directory checkpoint
    ckpt_dir = os.path.join("checkpoints", dataset_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 5) Epsilon schedule (esponenziale sull'episodio assoluto)
    def _epsilon_abs(abs_ep_idx, epsilon_end=0.05, decay=0.98):
        return max(epsilon_end, decay ** abs_ep_idx)

    env = RLEnvironment()
    ep_abs_last = start_episode

    try:
        for ep in range(start_episode, start_episode + num_episodes):
            ep_abs_last = ep
            eps = _epsilon_abs(ep, epsilon_end=epsilon_end, decay=eps_decay)
            ep_local = ep - start_episode  # 0..num_episodes-1
            print(f"[EP ABS {ep + 1} | LOCAL {ep_local + 1}/{num_episodes}] Epsilon: {eps:.4f}")

            # shuffle ad episodio
            perm = torch.randperm(N)

            # metriche episodio
            ep_total_reward = 0.0
            ep_total_correct = 0
            ep_only_row_correct = 0
            ep_total_correct_decade = 0
            ep_total_samples = 0
            ep_mae_sum = 0.0

            pbar = tqdm(perm.tolist(), desc=f"Episode {ep_local + 1}/{num_episodes}", leave=False)
            for idx in pbar:
                xi = X_cpu[idx].unsqueeze(0).to(device).float()  # [1,512]
                yi = y_cpu[idx].to(device).float()               # []

                gi, ei = _age_to_targets(yi.item())
                Ngi = counts.get(gi, 1)
                NgM = majority_class_count
                imbalance_ratio = min(NgM / max(Ngi, 1), 1e5)

                # riga iniziale: classificatore
                with torch.no_grad():
                    logits = classifier(xi)              # [1, C]
                    start_row = int(torch.argmax(logits, dim=1).item())

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
                    # stato pulito: [512] + [pos_r,pos_c]
                    pos_tensor   = torch.tensor([[env.r / 10.0, env.c / 10.0]], device=device)
                    state_tensor = torch.cat([xi, pos_tensor], dim=1)

                    if step == 0:
                        assert state_tensor.shape[1] == agent.q_network.fc1.in_features, \
                            f"State dim mismatch: got {state_tensor.shape[1]}, expected {agent.q_network.fc1.in_features}"

                    # azione
                    action = agent.select_action(state_tensor, epsilon=eps)

                    # step
                    prev_r, prev_c = env.r, env.c
                    _, reward, done = env.step(action)

                    # shaping distanza
                    dist_prev = abs(prev_r - gi) + abs(prev_c - ei)
                    dist_now  = abs(env.r - gi) + abs(env.c - ei)
                    reward += (dist_prev - dist_now) * 0.5

                    # penalità MAE (post-azione)
                    pred_age = env.r * 10 + env.c
                    mae_err  = abs(pred_age - float(yi.item()))
                    reward  -= lambda_mae * (mae_err / mae_scale)

                    # bonus goal
                    if env.r == gi and env.c == ei:
                        reward += 20.0
                        reached_target = True
                        done = True

                    # next state
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
                ep_mae_sum      += mae_err  # ultimo mae_err del sample (alla fine dell'episodio di navigazione)

                if reached_target:
                    ep_total_correct += 1
                if env.r == gi:
                    ep_total_correct_decade += 1
                    if env.c != ei:
                        ep_only_row_correct += 1

            # fine episodio
            acc       = 100.0 * ep_total_correct / max(1, ep_total_samples)
            acc_dec   = 100.0 * ep_total_correct_decade / max(1, ep_total_samples)
            only_row  = 100.0 * ep_only_row_correct / max(1, ep_total_samples)
            avg_rew   = ep_total_reward / max(1, ep_total_samples)
            ep_mae    = ep_mae_sum / max(1, ep_total_samples)

            print(
                f"📊 EP {ep + 1}: "
                f"acc={acc:.2f}% | decade_acc={acc_dec:.2f}% | only_row_acc={only_row:.2f}% "
                f"| avg_reward={avg_rew:.3f} | MAE={ep_mae:.2f}"
            )

            # checkpoint per-episodio
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if (ep + 1) % save_every == 0:
                out = os.path.join(ckpt_dir, f"rl_agent_partial_{ep + 1}_{ts}.pth")
                agent.save(out, episode=ep + 1)
                print(f"💾 Salvato checkpoint parziale: {out}")

            # best per accuracy (se ti serve ancora)
            if acc > best_accuracy:
                best_accuracy = acc
                best_acc_path = os.path.join(ckpt_dir, f"best_agent_acc_{ts}.pth")
                agent.save(best_acc_path, episode=ep + 1, extra={"metric": "accuracy", "value": acc})
                print(f"🏆 Nuovo best accuracy ({best_accuracy:.2f}%) → {best_acc_path}")

            # best per MAE (principale per x_mae)
            if ep_mae < best_mae:
                best_mae = ep_mae
                best_mae_path = os.path.join(ckpt_dir, f"best_agent_mae_{ts}.pth")
                agent.save(best_mae_path, episode=ep + 1, extra={"metric": "mae", "value": ep_mae})
                print(f"🏅 Nuovo best MAE ({best_mae:.2f}) → {best_mae_path}")

        return best_accuracy  # compat con chiamanti esistenti

    except KeyboardInterrupt:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        emer = os.path.join(ckpt_dir, f"rl_agent_EMERGENCY_{ep_abs_last + 1}_{ts}.pth")
        agent.save(emer, episode=ep_abs_last + 1, extra={"reason": "KeyboardInterrupt"})
        print(f"\n🛑 Interrotto. Emergency checkpoint salvato: {emer}")
        raise
    except Exception as e:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        emer = os.path.join(ckpt_dir, f"rl_agent_EMERGENCY_{ep_abs_last + 1}_{ts}.pth")
        agent.save(emer, episode=ep_abs_last + 1, extra={"reason": f"Exception: {e}"})
        print(f"\n💥 Eccezione. Emergency checkpoint salvato: {emer}")
        raise