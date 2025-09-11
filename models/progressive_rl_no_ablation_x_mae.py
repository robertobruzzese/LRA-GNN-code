# models/progressive_rl_no_ablation_x_mae.py

import copy
from collections import Counter
from typing import Optional
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt  # opzionale per eventuali plot

from training.rl_environment import RLEnvironment
from models.classifier import AgeGroupClassifier


# -------------------------------
# Q-Network (MLP 3 layer + LN)
# -------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)
        self.apply(self._init_weights)

    def forward(self, state):
        x = F.leaky_relu(self.norm1(self.fc1(state)))
        x = F.leaky_relu(self.norm2(self.fc2(x)))
        x = F.leaky_relu(self.norm3(self.fc3(x)))
        return self.output(x)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
            nn.init.constant_(m.bias, 0.0)


# -------------------------------------------------
# Progressive RL Agent (no ablation, X+MAE variant)
# -------------------------------------------------
class ProgressiveRLAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        target_tau: float = 0.005,  # Polyak
        device: str = "cpu",
        classifier: Optional[AgeGroupClassifier] = None,
    ):
        self.device = torch.device(device)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.gamma = float(gamma)
        self.target_tau = float(target_tau)
        self.tau = self.target_tau
        self.loss_fn = nn.SmoothL1Loss()  # Huber per TD-error
        self.classifier = classifier.to(self.device) if classifier is not None else None

        self.verbose = False  # metti True se vuoi stampe di debug

    # ---------------- Policy (greedy/ε-greedy) ----------------
    def select_action(self, state: torch.Tensor, epsilon: float):
        if np.random.rand() < float(epsilon):
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            q = self.q_network(state)
            return int(torch.argmax(q, dim=1).item())

    # ---------------- Aggiornamento Double-DQN ----------------
    def update(self, state, action, reward, next_state, done):
        """
        state/next_state: [1, state_dim]
        action: int
        reward: float
        done: bool
        """
        state = state.to(self.device).float()
        next_state = next_state.to(self.device).float()
        action = int(action)
        done = bool(done)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)

        q_values = self.q_network(state)              # [1, action_dim]
        q_value = q_values[0, action].unsqueeze(0)    # [1]

        with torch.no_grad():
            # Online network per best action
            next_q_online = self.q_network(next_state)         # [1, action_dim]
            best_action = torch.argmax(next_q_online, dim=1).item()

            # Target network per valore del best action
            next_q_target = self.target_network(next_state)    # [1, action_dim]
            next_q_value = next_q_target[0, best_action]       # scalar tensor

            target_scalar = reward if done else reward + self.gamma * next_q_value.item()
            target = torch.tensor([target_scalar], dtype=torch.float32, device=state.device)  # [1]

        loss = self.loss_fn(q_value, target)  # [1] vs [1]

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update_target_network()
        return float(loss.item())

    # ---------------- Salvataggio/Caricamento ----------------
    def save(self, path, episode: int = None, full_state: bool = True, **kwargs):
        """
        Salva lo stato dell'agente.
        - Accetta **kwargs (es. extra=...) per retro-compatibilità con i caller.
        - Se presente 'extra', salva anche un sidecar JSON con i metadati.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if full_state:
            torch.save({
                "episode": int(episode) if episode is not None else None,
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "gamma": self.gamma,
                "tau": self.tau,
                "state_dim": self.q_network.fc1.in_features,
                "action_dim": self.action_dim,
            }, path)
        else:
            torch.save(self.q_network.state_dict(), path)

        # Sidecar opzionale con metadati
        extra = kwargs.get("extra")
        if extra is not None:
            sidecar = {"episode": episode}
            sidecar.update(extra if isinstance(extra, dict) else {"info": str(extra)})
            try:
                with open(path + ".meta.json", "w", encoding="utf-8") as f:
                    json.dump(sidecar, f, ensure_ascii=False, indent=2)
            except Exception:
                # Non bloccare il training se il sidecar fallisce
                pass

    def load(self, path: str, map_location=None):
        """
        Carica un checkpoint full-state (consigliato) o, in fallback,
        uno state_dict puro del q_network.
        """
        ckpt = torch.load(path, map_location=self.device if map_location is None else map_location)

        if isinstance(ckpt, dict) and "q_network_state_dict" in ckpt:
            state = ckpt["q_network_state_dict"]
            self.q_network.load_state_dict(state, strict=False)

            # target network
            if "target_network_state_dict" in ckpt and ckpt["target_network_state_dict"] is not None:
                self.target_network.load_state_dict(ckpt["target_network_state_dict"], strict=False)
            else:
                self.target_network.load_state_dict(self.q_network.state_dict(), strict=False)

            # optimizer (best effort)
            if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    pass

            # iperparametri
            if "gamma" in ckpt:
                self.gamma = float(ckpt["gamma"])
            if "tau" in ckpt:
                self.tau = float(ckpt["tau"])

            self.q_network.to(self.device).eval()
            self.target_network.to(self.device).eval()

            ep = ckpt.get("episode", 0) or 0
            print(f"📥 Modello caricato da: {path} (episode={ep})")
            return int(ep)

        # fallback: trattalo come puro state_dict
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            # caso "old style": qualcuno ha salvato {"state_dict": ...}
            self.q_network.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            self.q_network.load_state_dict(ckpt, strict=False)

        self.q_network.to(self.device).eval()
        self.target_network.load_state_dict(self.q_network.state_dict(), strict=False)
        self.target_network.to(self.device).eval()

        print(f"📥 Modello (state_dict) caricato da: {path}")
        return 0

    def soft_update_target_network(self):
        tau = self.target_tau
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    # ---------------- Valutazione ----------------
    def evaluate(self, model, dataloader, device, epsilon_eval: float = 0.0):
        """
        Valuta la policy greedy sull'ambiente discreto (grid 10x10).
        Ritorna metriche RL + liste per MAE/CS@5.
        """
        if model is not None:
            model.eval()

        env = RLEnvironment()

        # Conta esempi per decade (serve all'ambiente)
        all_ys = torch.cat([age.view(-1).cpu() for _, age in dataloader])
        age_groups = (all_ys.numpy() // 10).astype(int).tolist()
        counts = Counter(age_groups)
        env.group_counts = dict(counts)
        env.majority_class_count = max(counts.values()) if counts else 1

        total_reward = 0.0
        total_correct = 0
        total_correct_decade = 0
        only_row_correct = 0
        total_samples = 0

        accuracy_list = []
        decade_accuracy_list = []
        true_labels, predicted_labels = [], []
        true_ages, predicted_ages = [], []

        with torch.no_grad():
            for batch_idx, (embedding, age) in enumerate(dataloader):
                x = embedding.to(self.device, dtype=torch.float32)           # [B, 512]
                y_true = age.to(self.device).float()     # [B]
                B = x.size(0)

                for i in range(B):
                    xi = x[i].unsqueeze(0)               # [1, 512]
                    yi = y_true[i]                       # scalar
                    gi = int(yi.item()) // 10
                    ei = int(yi.item()) % 10

                    # Stima iniziale decade + colonna dal classifier (se presente)
                    if self.classifier is not None:
                        # porta xi sul device del classifier in modo robusto
                        cls_dev = next(self.classifier.parameters()).device
                        xi_cls = xi.to(cls_dev, dtype=torch.float32)
                        logits = self.classifier(xi_cls)                 # [1, C]
                        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                        C = probs.shape[0]

                        # midpoint generico: 5,15,25,...
                        mid = np.arange(5, 5 + 10*C, 10, dtype=float)

                        pred_age_soft = float((probs * mid).sum())
                        start_row = int(pred_age_soft // 10)
                        start_col = int(round(pred_age_soft % 10))
                    else:
                        start_row = gi
                        start_col = 0  # oppure ei, se vuoi partire dalla colonna vera

                    # clip nei limiti della griglia (default 10x10)
                    n_rows = getattr(env, "n_rows", 10)
                    n_cols = getattr(env, "n_cols", 10)
                    start_row = max(0, min(n_rows - 1, start_row))
                    start_col = max(0, min(n_cols - 1, start_col))

                    # calcola l'imbalance PRIMA del reset
                    Ngi = env.group_counts.get(gi, 1)
                    NgM = env.majority_class_count
                    imbalance_ratio = min(NgM / max(Ngi, 1), 1e5)

                    # unico reset dell'ambiente
                    env.reset(
                        x=xi,
                        target_row=gi,
                        target_col=ei,
                        imbalance_ratio=imbalance_ratio,
                        actual_age=float(yi.item()),
                        start_row=start_row,
                    )
                    # se reset non accetta start_col, impostalo manualmente
                    if hasattr(env, "c"):
                        env.c = start_col

                    reached_target = False
                    episode_return = 0.0

                    # Stato = [embedding, pos_r/10, pos_c/10] → 512+2 = 514
                    max_steps = 100
                    for step in range(max_steps):
                        pos_tensor = torch.tensor(
                            [[env.r / 10.0, env.c / 10.0]],
                            device=self.device,
                            dtype=torch.float32
                        )
                        # xi è già sul device dell'agente (self.device) — se serve, porta xi lì
                        xi_on_agent = xi.to(self.device, dtype=torch.float32)
                        state_tensor = torch.cat([xi_on_agent, pos_tensor], dim=1)

                        if step == 0 and state_tensor.shape[1] != self.q_network.fc1.in_features:
                            raise RuntimeError(
                                f"State dim mismatch: {state_tensor.shape[1]} vs {self.q_network.fc1.in_features}"
                            )

                        prev_r, prev_c = env.r, env.c
                        action = self.select_action(state_tensor, epsilon=epsilon_eval)
                        _, reward, done = env.step(action)

                        # shaping distanza
                        dist_prev = abs(prev_r - gi) + abs(prev_c - ei)
                        dist_now  = abs(env.r - gi) + abs(env.c - ei)
                        reward += (dist_prev - dist_now) * 0.5

                        episode_return += float(reward)

                        if env.r == gi and env.c == ei:
                            episode_return += 20.0
                            reached_target = True
                            break

                    # metriche
                    predicted_decade = env.r
                    actual_decade = gi
                    if predicted_decade == actual_decade:
                        total_correct_decade += 1
                        decade_accuracy_list.append(1)
                    else:
                        decade_accuracy_list.append(0)

                    if reached_target:
                        total_correct += 1
                        accuracy_list.append(1)
                    else:
                        accuracy_list.append(0)

                    # per MAE/CS@5
                    true_labels.append(gi)
                    predicted_labels.append(env.r)
                    true_ages.append(float(yi.item()))
                    predicted_ages.append(env.r * 10 + env.c)

                    total_samples += 1
                    total_reward += episode_return

                    if env.r == gi and env.c != ei:
                        only_row_correct += 1

        print(f"🔢 Samples: {total_samples}")
        print(f"✅ Target esatti: {total_correct}")
        print(f"🔹 Solo riga corretta: {only_row_correct}")
        print(
            f"📊 RL → Accuracy: {100.0 * total_correct / max(total_samples, 1):.2f}% | "
            f"Decade: {100.0 * total_correct_decade / max(total_samples, 1):.2f}% | "
            f"AvgReward: {total_reward / max(total_samples, 1):.4f}"
        )

        return {
            "accuracy": 100.0 * total_correct / max(total_samples, 1),
            "decade_accuracy": 100.0 * total_correct_decade / max(total_samples, 1),
            "only_row_accuracy": 100.0 * only_row_correct / max(total_samples, 1),
            "avg_reward": total_reward / max(total_samples, 1),
            "accuracy_list": accuracy_list,
            "decade_accuracy_list": decade_accuracy_list,
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
            "true_ages": true_ages,
            "predicted_ages": predicted_ages,
        }