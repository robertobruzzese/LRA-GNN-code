import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from collections import Counter
from training.rl_environment import RLEnvironment
import matplotlib.pyplot as plt
from models.classifier import AgeGroupClassifier
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn import GCNConv



class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.output = nn.Linear(hidden_dim, action_dim)

        self.apply(self.init_weights)  # inizializza i pesi

    def forward(self, state):
        x = F.leaky_relu(self.norm1(self.fc1(state)))
        x = F.leaky_relu(self.norm2(self.fc2(x)))
        x = F.leaky_relu(self.norm3(self.fc3(x)))
        return self.output(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0)
    
class PRLAELoss(nn.Module):
    def __init__(self, eta=0.5, tau=1.3):
        super().__init__()
        self.eta = eta
        self.tau = tau
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        # Assume pred is raw score, apply sigmoid to get probability
        prob = torch.sigmoid(pred)
        focal = -((1 - prob) ** self.tau) * torch.log(prob + 1e-8)
        mae = self.mae(pred, target)
        return self.eta * focal.mean() + (1 - self.eta) * mae

class ProgressiveRLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.00001, gamma=0.99, tau=1.0, device='cpu',            classifier=None):
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = copy.deepcopy(self.q_network).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.tau = tau
        # Usa PRLAELoss invece di MSE
        # self.loss_fn = PRLAELoss(eta=0.8, tau=4.0)
        #self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.MSELoss()
        ########self.loss_fn = nn.SmoothL1Loss()
        # ✅ Loss PRLAE come nel paper
        self.loss_fn = PRLAELoss(eta=0.4, tau=1.3)
        # 🔽 Salva il classificatore passato
        self.classifier = classifier.to(device) if classifier is not None else None

        self.action_dim = action_dim
        self.device = device  # 👉 Salva il device usato
        # 🔁 GNN per estrazione feature da grafi LRC
        self.gnn = GATConv(in_channels=512, out_channels=512)  # se gli embedding sono a 512oppure un GCNConv
    
    def extract_features(self, graphs):
        """
        Estrae le feature da tutti i grafi LRC usando GAT (o GCN), e ritorna la media.
        `graphs` è una lista di oggetti `torch_geometric.data.Data`.
        """
        outputs = []
        for graph in graphs:
            graph = graph.to(self.device)
            # ⚠️ Assicurati che self.gnn sia definito (es. un GAT/GNN nel tuo agente)
            out = self.gnn(graph.x, graph.edge_index)  # (num_nodes, feature_dim)
            pooled = out.mean(dim=0)  # (feature_dim,)
            outputs.append(pooled)
        return torch.stack(outputs, dim=0).mean(dim=0)  # (feature_dim,)

    def select_action(self, state, epsilon):
        print(f"👉 select_action chiamato con epsilon={epsilon:.4f}")
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
            print(f"🎲 Azione casuale scelta (ε-greedy): {action}")
            return np.random.choice(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                # 🔍 Stampa Q-values per debugging
                print(f"   🔢 Q-values: {q_values.cpu().numpy().flatten()}")
                action = torch.argmax(q_values).item()
                print(f"✅ Azione greedy scelta: {action}")
                
                return torch.argmax(q_values).item()
   
    
    def update(self, state, action, reward, next_state, done):
        state = state.float()
        next_state = next_state.float()

        # Copia dei pesi prima dell'update
        #params_after = list(self.q_network.parameters())
        #params_before = copy.deepcopy(list(self.q_network.parameters()))
        q_values = self.q_network(state)
        print(f"📊 Q-values aggiornati: {q_values.detach().cpu().numpy().flatten()}")
        q_value = q_values[0, action]
        with torch.no_grad():
            # Double DQN: seleziona azione migliore con q_network
            next_q_values_online = self.q_network(next_state)
            best_action = torch.argmax(next_q_values_online).item()

                          # Q(s, a)
            # Valore di quella azione dalla target_network
            #next_q_values_target = self.target_network(next_state)
            next_q_values_target = self.target_network(next_state.to(self.q_network.fc1.weight.device))

            next_q_value = next_q_values_target[0, best_action]

            target = reward if done else reward + self.gamma * next_q_value
            target = torch.tensor([target], dtype=torch.float32, device=state.device)

        loss = self.loss_fn(q_value.unsqueeze(0), target)  # matcha shape: [1]
            
        self.optimizer.zero_grad()
        #loss.backward()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        # 👇 Esegui aggiornamento morbido della rete target
        self.soft_update_target_network()

        return loss

    def soft_update_target_network(self):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path=None):
         if path is None:
            raise ValueError("❗ Devi specificare un path per salvare l'agente RL.")
         torch.save(self.q_network.state_dict(), path)
    
    
    def load(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.q_network.to(self.device)  
        self.q_network.eval()
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"📥 Modello caricato da: {path}")


    def evaluate(self, model, dataloader, device):
        print("🧪 evaluate() DEFINITIVA in uso ✅")
        print("\n🧪 Inizio evaluate: nuovo round di valutazione\n")
        cls_only_correct = 0
        cls_only_total = 0
        if model is not None:
            model.eval()
        #model.eval()
        env = RLEnvironment()
        # 🔢 Estrai tutte le età
        #all_ys = torch.cat([batch.y for batch in dataloader])
        #for i, item in enumerate(dataloader):
        #    print(f"[DEBUG] Item {i}: {item}")
        #    if i == 2:
        #        break
        #all_ys = torch.cat([torch.tensor([age], dtype=torch.float32) for _, age in dataloader])
        #all_ys = torch.cat([torch.tensor([item['age']], dtype=torch.float32) for item in dataloader])
        #all_ys = torch.cat([torch.tensor([age], dtype=torch.float32) for _, age in dataloader])
        #all_ys = torch.tensor([age for _, age in dataloader], dtype=torch.float32)
        #all_ys = torch.tensor([float(age) for _, age in dataloader], dtype=torch.float32)
        #all_ys = torch.tensor([age for _, age in dataloader]).float()
        #all_ys = torch.tensor([float(age) for _, age in dataloader], dtype=torch.float32)
        #all_ys = torch.cat([age for _, age in dataloader]).float()
        all_ys = torch.tensor([
            (sample["age"] if "age" in sample else sample["label"]).item()
            if isinstance(sample, dict) else sample[1].item()
            for sample in dataloader.dataset
        ])
        # ✅ Stampa primi 10 valori veri per controllo leakage
        
        age_groups = [int(y.item()) // 10 for y in all_ys]
        counts = Counter(age_groups)

        # 📊 Salva nel tuo ambiente RL
        env.group_counts = dict(counts)
        env.majority_class_count = max(counts.values())

        total_reward = 0
        total_correct = 0
        total_samples = 0
        only_row_correct = 0 
        total_correct_decade = 0
        accuracy_list = []
        decade_accuracy_list = []
        true_labels = []
        predicted_labels = []

        true_ages = []  # ✅ aggiunto
        predicted_ages = []  # ✅ aggiunto



        with torch.no_grad():

            #for batch_idx, batch in enumerate(dataloader):
            #    print(f"📦 Batch {batch_idx + 1}")

             #   batch = batch.to(device)
             #   x = model(batch, return_features=True)
             #   print(f"📦 Batch size corrente: {x.size(0)}")

              #  y_true = batch.y.float()
            for batch_idx, sample in enumerate(dataloader):
                if isinstance(sample, dict):
                    x = sample["embedding"].to(device)
                    #y_true = sample["label"].to(device).float()
                            # 🔧 Pooling per garantire shape [1, 512]
                    if x.dim() == 2:
                        x = x.mean(dim=0, keepdim=True)  # [N, 512] → [1, 512]
                    elif x.dim() == 1:
                        x = x.unsqueeze(0)               # [512] → [1, 512]
                    elif x.dim() == 3:
                        x = x.mean(dim=1)                # [1, N, 512] → [1, 512]
                    else:
                        raise ValueError(f"❌ Shape embedding non supportata: {x.shape}")
                    if "label" in sample:
                        y_true = sample["label"].to(device).float()
                    else:
                        y_true = sample["age"].to(device).float()
                elif isinstance(sample, (tuple, list)):
                    x = sample[0].to(device)
                    y_true = sample[1].to(device).float()
                else:
                    raise ValueError(f"Formato sample non riconosciuto: {type(sample)}")

                for i in range(x.size(0)):  

                    xi = x[i].unsqueeze(0)
                    yi = y_true[i]
                    gi = int(yi.item()) // 10
                    ei = int(yi.item()) % 10

                    Ngi = env.group_counts[gi]
                    NgM = env.majority_class_count
                    imbalance_ratio = min(NgM / max(Ngi, 1), 1e5)

                    # === UTKFACE-safe: separa RAW per RL e SCALATO per classifier ===
                    xi_raw = xi.clone()     # quello che userà SEMPRE la Q-network e il reset
                    xi_cls = xi             # quello che (eventualmente scalato) usa SOLO il classifier

                    if hasattr(self, "scaler") and (self.scaler is not None):
                        xi_np  = xi.detach().cpu().numpy()        # [1, 512]
                        xi_np  = self.scaler.transform(xi_np)     # standardizzazione come nel training del classifier
                        xi_cls = torch.tensor(xi_np, dtype=xi.dtype, device=xi.device)

                    # 1) decade iniziale dal classifier (sul dato SCALATO)
                    with torch.no_grad():
                        start_row = torch.argmax(self.classifier(xi_cls), dim=1).item()

                    # (facoltativo) mini sanity sui primi 64 sample
                    if batch_idx < 64:
                        cls_only_correct += int(start_row == gi)
                        cls_only_total   += 1

                    # 2) reset env con l'EMBEDDING RAW (come visto in training dall'RL)
                    env.reset(
                        x=xi_raw,
                        target_row=gi,
                        target_col=ei,
                        imbalance_ratio=imbalance_ratio,
                        actual_age=yi.item()
                    )

                    # 3) forza lo start stimato dal classifier: r = decade, c = 0
                    env.r = int(start_row)
                    env.c = 0
                    lock_row = True  # <-- per UTKFACE blocchiamo la decade fornita dal classifier

                    reached_target = False

                    for step in range(100):  # o max_steps
                        #pos_encoding = torch.tensor([[env.r / 10, env.c / 10]], device=device)
                        #target_encoding = torch.tensor([[gi / 10, ei / 10]], device=device)
                        target_tensor = torch.tensor([[gi / 10, ei / 10]], device=xi.device)
                        delta_x = torch.tensor([[(gi - env.r) / 10.0]], device=device)
                        delta_y = torch.tensor([[(ei - env.c) / 10.0]], device=device)
                        pos_tensor = torch.tensor([[env.r / 10.0, env.c / 10.0]], device=device)
                        print(f"xi shape: {xi.shape}")
                        print(f"delta_x shape: {delta_x.shape}")
                        print(f"delta_y shape: {delta_y.shape}")
                        print(f"pos_tensor shape: {pos_tensor.shape}")
                        print(f"target_tensor shape: {target_tensor.shape}")
                        # Se xi è 3D, riducilo a 2D
                        if xi_raw.dim() == 3:
                              xi_raw = xi_raw.mean(dim=1)
                        state_tensor = torch.cat([xi_raw, delta_x, delta_y, pos_tensor, target_tensor], dim=1)



                        #state_tensor = torch.cat([xi, pos_encoding, target_encoding], dim=1)
                        # Salva la posizione precedente
                        prev_r, prev_c = env.r, env.c
                        action = self.select_action(state_tensor, epsilon=0.0)
                        _, reward, done = env.step(action)
                        # 🔒 blocca SEMPRE la riga alla decade del classifier
                        if lock_row:
                            env.r = int(start_row)

                         #📍 Reward intermedio: distanza manhattan
                        dist_prev = abs(prev_r - gi) + abs(prev_c - ei)
                        dist_now = abs(env.r - gi) + abs(env.c - ei)
                        delta_dist = dist_prev - dist_now
                        reward += delta_dist * 0.5 

                        if env.r == gi and env.c == ei:
                            reward += 20.0
                            reached_target = True
                            break  # ✅ uscita anticipata se il target è raggiunto
                    
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
                       desc = "✅ CORRETTO"
                    else:
                       accuracy_list.append(0)
                       desc = "❌ ERRATO"
                       print(f"❌ Predicted: ({env.r}, {env.c}) | GT: ({gi}, {ei})")
                    # Manca questa riga importante nel loop interno
                    true_labels.append(gi)
                    predicted_labels.append(env.r)

                    true_ages.append(yi.item())  # ✅ aggiunto
                    predicted_ages.append(env.r * 10 + env.c)  # ✅ aggiunto



                    total_samples += 1

                        # 🔹 Nuova metrica: riga corretta, colonna errata
                    if env.r == gi and env.c != ei:
                        only_row_correct += 1
                        print(f"🔵 Solo RIGA corretta: pred ({env.r}, {env.c}) vs target ({gi}, {ei})")

             # 📊 RISULTATI FINALI
            print(f"🔢 Numero totale di sample valutati: {total_samples}")
            print(f"✅ Target esatti raggiunti: {total_correct}")
            print(f"🔹 Solo RIGA corretta (decade): {only_row_correct}")
            print(f"📊 RL Evaluation → Accuracy Finale: {(100.0 * total_correct / total_samples):.2f}%, Avg Reward: {total_reward / total_samples:.4f}")
            print(f"📊 RL Evaluation → Accuracy sulla decade: {(100.0 * total_correct_decade / total_samples):.2f}%")
            print(f"📊 RL Evaluation → Accuracy SOLO decade (riga corretta, colonna errata): {(100.0 * only_row_correct / total_samples):.2f}%")

            # 📈 Accuracy per sample
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(accuracy_list)), accuracy_list,
                    color=['green' if a == 1 else 'red' for a in accuracy_list])
            plt.title("Accuratezza per campione")
            plt.xlabel("Campione")
            plt.ylabel("Esito (1 = corretto, 0 = errato)")
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.xticks(range(len(accuracy_list)), [f"#{i+1}" for i in range(len(accuracy_list))], rotation=45)
            plt.show()

            # 📈 Accuracy per decade
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(decade_accuracy_list)), decade_accuracy_list,
                    color=['blue' if a == 1 else 'orange' for a in decade_accuracy_list])
            plt.title("Accuratezza per decade (0 = errato, 1 = corretto)")
            plt.xlabel("Campione")
            plt.ylabel("Esito Decade")
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.xticks(range(len(decade_accuracy_list)), [f"#{i+1}" for i in range(len(decade_accuracy_list))], rotation=45)
            plt.show()
            if cls_only_total > 0:
                print(f"🧪 Classifier-only decade acc (first {cls_only_total}): {100.0 * cls_only_correct / cls_only_total:.2f}%")
            print("👉 True ages:", true_ages[:10])
            print("👉 Predicted ages:", predicted_ages[:10])
            print("🧪 Check Leakage → True ages:")
            print([age.item() if torch.is_tensor(age) else age for _, age in list(dataloader)[:10]])

            for i in range(10):
                print(f"[{i}] True: {all_ys[i].item()}  |  Predicted: {predicted_ages[i]}")

            return {
                "accuracy": 100.0 * total_correct / total_samples,
                "decade_accuracy": 100.0 * total_correct_decade / total_samples,
                "only_row_accuracy": 100.0 * only_row_correct / total_samples,
                "avg_reward": total_reward / total_samples,
                "accuracy_list": accuracy_list,
                "decade_accuracy_list": decade_accuracy_list,
                "true_labels": true_labels,

                "predicted_labels": predicted_labels,
                "true_ages": true_ages,  # ✅ restituito
                "predicted_ages": predicted_ages  # ✅ restituito
            }

