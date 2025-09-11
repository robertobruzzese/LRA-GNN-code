import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.embedding_dataset_ablation_prlae import EmbeddingDataset
from dataset.embedding_dataset_ablation_prlae_x_train_classifier import EmbeddingDatasetPRLAEXClassifier

from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# ⚙️ GCN Encoder per grafi LRC
class GraphEncoder(nn.Module):
    def __init__(self, in_channels=512, hidden_dim=512):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return self.pool(x, batch)
    
# 📦 MLP Classifier
class AgeGroupClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 🧠 Training
def train_classifier(embeddings, ages, input_dim, device='cpu', epochs=700, batch_size=32):
    targets = torch.tensor([int(age.item()) // 10 for age in ages], dtype=torch.long)

    dataset = TensorDataset(embeddings, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AgeGroupClassifier(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == yb).sum().item()

        accuracy = 100 * correct / len(dataset)
        print(f"📚 Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Accuracy: {accuracy:.2f}%")

    return model

def extract_embeddings_and_labels(dataloader, device):
    X_list, y_list = [], []
    for sample in dataloader:
        if isinstance(sample, dict):  # caso DFE
            embeddings = sample["embedding"].to(device)
            ages = sample["label"].to(device)
        elif isinstance(sample, (list, tuple)):  # caso LRC
            embeddings, ages = sample
            embeddings = embeddings.to(device)
            ages = ages.to(device)
        else:
            print("⚠️  Sample non valido:", type(sample))
            continue

        #X_list.append(embeddings)
        embeddings = embeddings.view(-1, embeddings.shape[-1])  # garantisce shape [1, 256]
        X_list.append(embeddings)
        y_list.append(ages)

    if len(X_list) == 0:
        raise ValueError("❌ Nessun embedding valido trovato.")

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y

# 🚀 Entry Point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset da usare: MORPH, FGNET, UTKFACE, CLAP2016")
    parser.add_argument("--exp_name", type=str, required=True, choices=["lrc", "dfe"],help="Nome esperimento (es. dfe, lrc, lrc_dfe, no_lrc_no_dfe)")
    args = parser.parse_args()

    dataset_name = args.dataset.upper()
    exp_name = args.exp_name

     # ⬇️ Determina directory e flag
    if exp_name == "lrc":
        embedding_dir = f"embeddings_ablation_{dataset_name.lower()}_lrc_no_dfe/train"
        enable_lrc = True
        enable_dfe = False
    elif exp_name == "dfe":
        embedding_dir = f"embeddings_ablation_{dataset_name.lower()}_no_lrc_dfe/train"
        enable_lrc = False
        enable_dfe = True

    
    
    #checkpoint_dir = os.path.join("checkpoints_ablation", dataset_name.lower(), f"prlae_{exp_name}")
    if enable_lrc and not enable_dfe:
        exp_tag = "prlae_lrc_no_dfe"
    elif not enable_lrc and enable_dfe:
        exp_tag = "prlae_no_lrc_dfe"
    elif enable_lrc and enable_dfe:
        exp_tag = "prlae_lrc_dfe"
    else:
        exp_tag = "prlae_no_lrc_no_dfe"

    checkpoint_dir = os.path.join("checkpoints_ablation", dataset_name.lower(), exp_tag)
    # ⚙️ Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 📥 Carica dataset
    # 📥 Dataset
   # 📥 Carica dataset
    if exp_name == "lrc":
        encoder = GraphEncoder().to(device)
        embedding_dataset = EmbeddingDatasetPRLAEXClassifier(
            embeddings_dir=embedding_dir,
            dataset_name=dataset_name,
            encoder=encoder,
            device=device
        )
    elif exp_name == "dfe":
        embedding_dataset = EmbeddingDataset(
            embedding_dir,
            dataset_name,
            return_dict=True,
            enable_lrc=False,
            enable_dfe=True
        )
    else:
        raise ValueError("❌ Esperimento non valido: deve essere 'lrc' o 'dfe'")
    embedding_loader = DataLoader(embedding_dataset, batch_size=1, shuffle=False)

    # 📊 Estrai embedding filtrando quelli errati
    X_real, y_real = extract_embeddings_and_labels(embedding_loader, device)

    # 🧠 Addestramento
    #model = train_classifier(X_real, y_real, input_dim=X_real.shape[1], device=device)
    # 🔍 Determina input_dim dinamicamente dal primo embedding
    # 🔍 Forza dimensione corretta se LRC
    if exp_name == "lrc":
        input_dim = 512
    else:
        input_dim = X_real.shape[1]

    print(f"ℹ️ Dimensione embedding rilevata: {input_dim}")

    # 🧠 Addestramento
    model = train_classifier(X_real, y_real, input_dim=input_dim, device=device)
    # 💾 Salvataggio nella cartella ablation corretta
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "classifier.pth"))
    print(f"✅ Classificatore salvato in {checkpoint_dir}/classifier.pth")