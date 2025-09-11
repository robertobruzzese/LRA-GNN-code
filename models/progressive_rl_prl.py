# progressive_rl_prl.py
# PRL (Progressive Reinforcement Learning) - setup pulito e comparabile al paper
# - Stato: [embedding, r/10, c/10]
# - Azioni: {up, down, left, right, stay}
# - Update: Double DQN + Replay Buffer + soft target update
# - Loss: PRLAE (eta ~ 0.4-0.5, tau ~ 1.3)
# - Nessun leakage: il GT è usato SOLO dall'ambiente per reward/terminazione

import copy
import math
import numpy as np
from collections import Counter, deque
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ambiente: deve implementare calculate_reward come da paper (senza shaping extra)
#from training.rl_environment import RLEnvironment
from training.rl_environment_prl import RLEnvironment

# (Opzionale) Classificatore per stimare la riga iniziale; deve essere addestrato SOLO su TRAIN
try:
    from models.classifier import AgeGroupClassifier  # noqa: F401
except Exception:
    AgeGroupClassifier = None  # facoltativo

# === build_state helper ===
# Se hai creato models/prl_helpers.py usa quello; altrimenti fallback locale.
try:
    from models.prl_helpers import build_state as _external_build_state
except Exception:
    _external_build_state = None

def build_state(xi: torch.Tensor, env: RLEnvironment, device: torch.device) -> torch.Tensor:
    """
    Costruisce lo stato lecito: [embedding, r/10, c/10]
    xi: [1, D] embedding
    """
    pos = torch.tensor([[env.r / 10.0, env.c / 10.0]], device=device)  # [1, 2]
    s = torch.cat([xi, pos], dim=1)  # [1, D+2]
    return s

if _external_build_state is not None:
    build_state = _external_build_state  # preferisci helper condiviso se presente


# === Q-Network ===
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.do1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.do2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.do3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.out = nn.Linear(hidden_dim, action_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.norm1(self.fc1(s)))
        x = self.do1(x)
        x = F.leaky_relu(self.norm2(self.fc2(x)))
        x = self.do2(x)
        x = F.leaky_relu(self.norm3(self.fc3(x)))
        x = self.do3(x)
        return self.out(x)


# === PRLAE loss (paper) ===
class PRLAELoss(nn.Module):
    def __init__(self, eta: float = 0.4, tau: float = 1.3):
        super().__init__()
        self.eta = float(eta)
        self.tau = float(tau)
        self.mae = nn.L1Loss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred e target hanno shape [B]; pred sono Q(s,a) selezionati
        # componi loss focal-like su pred "normalizzati" con sigmoid
        prob = torch.sigmoid(pred)
        focal = -((1.0 - prob) ** self.tau) * torch.log(prob + 1e-8)
        mae = self.mae(pred, target)
        return self.eta * focal.mean() + (1.0 - self.eta) * mae


# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s: torch.Tensor, a: int, r: float, ns: torch.Tensor, d: bool):
        # Memorizza su CPU per risparmiare VRAM
        self.buf.append((s.detach().cpu(), int(a), float(r), ns.detach().cpu(), bool(d)))

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(len(self.buf), size=batch_size)
        s, a, r, ns, d = zip(*[self.buf[i] for i in idx])
        S = torch.cat(s, dim=0).to(device)                 # [B, state_dim]
        A = torch.tensor(a, dtype=torch.long, device=device)         # [B]
        R = torch.tensor(r, dtype=torch.float32, device=device)      # [B]
        NS = torch.cat(ns, dim=0).to(device)               # [B, state_dim]
        D = torch.tensor(d, dtype=torch.bool, device=device)         # [B]
        return S, A, R, NS, D

    def __len__(self):
        return len(self.buf)


# === Progressive RL Agent (PRL) ===
class ProgressiveRLAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        tau_soft: float = 0.005,
        loss_eta: float = 0.4,
        loss_tau: float = 1.3,
        device: torch.device = torch.device("cpu"),
        classifier: Optional[nn.Module] = None,
    ):
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim, dropout).to(device)
        self.target_network = copy.deepcopy(self.q_network).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = float(gamma)
        self.tau_soft = float(tau_soft)
        self.loss_fn = PRLAELoss(eta=loss_eta, tau=loss_tau)
        self.action_dim = int(action_dim)
        self.device = device

        self.classifier = classifier.to(device) if classifier is not None else None

        # Conteggi per l'imbalance ratio (da impostare con i conteggi del TRAIN)
        self.group_counts: Dict[int, int] = {}
        self.majority_class_count: int = 1

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=100_000)

    # --------- Utility conteggi ----------
    def set_class_counts(self, counts: Dict[int, int]):
        self.group_counts = dict(counts)
        self.majority_class_count = max(1, max(self.group_counts.values()) if self.group_counts else 1)

    @staticmethod
    def compute_counts_from_loader(dataloader) -> Dict[int, int]:
        ages = []
        for batch in dataloader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                age = batch[1]
            else:
                continue
            if torch.is_tensor(age):
                ages.append(float(age.view(-1)[0].item()))
            else:
                try:
                    ages.append(float(age))
                except Exception:
                    pass
        decs = [int(a // 10) for a in ages]
        return dict(Counter(decs))

    def _imbalance_ratio(self, g: int, cap: float = 1e5) -> float:
        Ngi = max(1, self.group_counts.get(int(g), 1))
        NgM = max(1, self.majority_class_count)
        return float(min(NgM / Ngi, cap))

    # --------- Azione (ε-greedy) ----------
    def select_action(self, state: torch.Tensor, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.choice(self.action_dim))
        with torch.no_grad():
            q = self.q_network(state)  # [1, A]
            return int(torch.argmax(q, dim=1).item())

    # --------- Soft update target ----------
    def soft_update_target_network(self):
        with torch.no_grad():
            for tp, p in zip(self.target_network.parameters(), self.q_network.parameters()):
                tp.data.mul_(1.0 - self.tau_soft).add_(self.tau_soft * p.data)

    # --------- Save / load ----------
    def save(self, path: str):
        torch.save({"q_network_state_dict": self.q_network.state_dict()}, path)

    def load(self, path: str):
        state_raw = torch.load(path, map_location=self.device)
        if isinstance(state_raw, dict) and ("q_network_state_dict" in state_raw or "state_dict" in state_raw):
            state = state_raw.get("q_network_state_dict", state_raw.get("state_dict"))
        else:
            state = state_raw
        # rimuovi prefissi eventuali
        clean = { (k[len("q_network."):] if k.startswith("q_network.") else k): v for k, v in state.items() }
        self.q_network.load_state_dict(clean, strict=False)
        self.q_network.to(self.device).eval()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.to(self.device).eval()

    # --------- Training (un'epoca) ----------
    def train_one_epoch(
        self,
        dataloader,
        max_steps: int = 100,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        updates_per_step: int = 1,
        use_classifier_start: bool = True,
        clip_reward: Optional[float] = None,
    ) -> Tuple[float, int]:
        """
        Addestra per un'epoca scorrendo il dataloader.
        Ritorna (loss_media, num_updates).
        """
        device = self.device
        self.q_network.train()
        self.target_network.eval()  # target non si addestra

        if not self.group_counts:
            # Avviso: è meglio passarli dal TRAIN set intero a monte
            counts = self.compute_counts_from_loader(dataloader)
            self.set_class_counts(counts)

        global_step = 0
        total_loss = 0.0
        n_updates = 0

        env = RLEnvironment(max_steps=max_steps)

        for batch in dataloader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                emb, age = batch[0].to(device), batch[1].float().to(device)
            else:
                continue

            B = emb.size(0)
            for i in range(B):
                xi = emb[i].unsqueeze(0)       # [1, D]
                yi = float(age[i].item())
                gi, ei = int(yi) // 10, int(yi) % 10
                rho = self._imbalance_ratio(gi)

                # start_row opzionale dal classifier (allenato SOLO su TRAIN)
                start_row = None
                if use_classifier_start and (self.classifier is not None):
                    with torch.no_grad():
                        start_row = int(torch.argmax(self.classifier(xi), dim=1).item())

                env.reset(x=xi, target_row=gi, target_col=ei,
                          imbalance_ratio=rho, actual_age=yi, start_row=start_row)

                s = build_state(xi, env, device)

                for t in range(max_steps):
                    # schedule epsilon: esponenziale
                    eps = float(epsilon_end + (epsilon_start - epsilon_end) *
                                math.exp(-1.0 * global_step / max(1, epsilon_decay_steps)))

                    a = self.select_action(s, eps)
                    _, r, done = env.step(a)

                    if clip_reward is not None:
                        r = float(np.clip(r, -clip_reward, +clip_reward))

                    ns = build_state(xi, env, device)
                    self.buffer.push(s, a, r, ns, done)
                    s = ns
                    global_step += 1

                    # updates
                    if len(self.buffer) >= batch_size:
                        for _ in range(updates_per_step):
                            S, A, R, NS, D = self.buffer.sample(batch_size, device)
                            # Q(s,a)
                            q_sa = self.q_network(S).gather(1, A.view(-1, 1)).squeeze(1)
                            with torch.no_grad():
                                a_star = self.q_network(NS).argmax(dim=1, keepdim=True)         # online
                                q_tgt = self.target_network(NS).gather(1, a_star).squeeze(1)    # target
                                Y = R + (~D).float() * self.gamma * q_tgt

                            loss = self.loss_fn(q_sa, Y)
                            self.optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
                            self.optimizer.step()
                            self.soft_update_target_network()

                            total_loss += float(loss.item())
                            n_updates += 1

                    if done:
                        break

        # torna in eval dopo il training
        self.q_network.eval()
        self.target_network.eval()
        mean_loss = (total_loss / max(1, n_updates))
        return mean_loss, n_updates

    # --------- Evaluation pulita (no shaping, no leakage) ----------
    @torch.no_grad()
    def evaluate(
        self,
        model,                  # non usato (compatibilità)
        dataloader,
        device: torch.device,
        max_steps: int = 100,
        start_mode: str = "classifier",   # "classifier" oppure "random"
    ):
        self.q_network.eval()
        if self.classifier is not None:
            self.classifier.eval()

        # Prepara conteggi SOLO per logging/reward; NON entrano nello stato
        counts = self.compute_counts_from_loader(dataloader)
        # Se vuoi evitare di "ri-calcolare" su val/test, puoi omettere questa riga
        self.set_class_counts(counts)

        env = RLEnvironment(max_steps=max_steps)

        total_correct = 0
        total_correct_decade = 0
        only_row_correct = 0
        total_samples = 0
        total_return = 0.0

        true_labels, pred_labels = [], []
        true_ages, pred_ages = [], []

        for batch in dataloader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                emb, age = batch[0].to(device), batch[1].float().to(device)
            else:
                continue

            B = emb.size(0)
            for i in range(B):
                xi = emb[i].unsqueeze(0)  # [1, D]
                yi = float(age[i].item())
                gi, ei = int(yi) // 10, int(yi) % 10
                rho = self._imbalance_ratio(gi)

                # start
                start_row = None
                if start_mode == "classifier" and (self.classifier is not None):
                    logits = self.classifier(xi)
                    start_row = int(torch.argmax(logits, dim=1).item())

                env.reset(x=xi, target_row=gi, target_col=ei,
                          imbalance_ratio=rho, actual_age=yi, start_row=start_row)

                s = build_state(xi, env, device)
                episode_return = 0.0
                reached = False

                for _ in range(max_steps):
                    a = self.select_action(s, epsilon=0.0)  # greedy in eval
                    _, r, done = env.step(a)
                    episode_return += float(r)
                    s = build_state(xi, env, device)
                    if done:
                        reached = (env.r == gi and env.c == ei)
                        break

                # metriche discrete
                if env.r == gi:
                    total_correct_decade += 1
                    if env.c != ei:
                        only_row_correct += 1
                if reached:
                    total_correct += 1

                # log
                total_return += episode_return
                total_samples += 1
                true_labels.append(gi)
                pred_labels.append(env.r)
                true_ages.append(yi)
                pred_ages.append(env.r * 10 + env.c)

        return {
            "accuracy": 100.0 * total_correct / max(1, total_samples),
            "decade_accuracy": 100.0 * total_correct_decade / max(1, total_samples),
            "only_row_accuracy": 100.0 * only_row_correct / max(1, total_samples),
            "avg_reward": total_return / max(1, total_samples),
            "true_labels": true_labels,
            "predicted_labels": pred_labels,
            "true_ages": true_ages,
            "predicted_ages": pred_ages,
        }