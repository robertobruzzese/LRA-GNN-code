import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.embedding_dataset import EmbeddingDataset 

 # importa dal file dove lo hai definito
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
def train_classifier(embeddings, ages, input_dim, device='cpu', epochs=5500, batch_size=32):
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

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/classifier.pth")
    print("✅ Classificatore salvato in checkpoints/classifier.pth")
    return model

# 🔄 Estrazione
def extract_embeddings_and_labels(dataloader, device):
    X_list, y_list = [], []
    for embedding, age in dataloader:
        X_list.append(embedding.to(device))
        y_list.append(age.to(device))
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y

# 🚀 Entry Point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MORPH", help="Dataset da usare: MORPH o FGNET")
    args = parser.parse_args()
    dataset_name = args.dataset.upper()
    

    # 📁 Directory corretta
   # embedding_dir = "embeddings/train" if dataset_name == "MORPH" else "embeddings_FGNET/train"
    # 📁 Directory corretta
    if dataset_name == "MORPH":
        embedding_dir = "embeddings_morph/train"
    elif dataset_name == "FGNET":
        embedding_dir = "embeddings_FGNET/train"
    elif dataset_name == "UTKFACE":
        embedding_dir = "embeddings_utkface/train"
    elif dataset_name == "CLAP2016":
        embedding_dir = "embeddings_clap2016/train"
    else:
        raise ValueError(f"❌ Dataset non riconosciuto: {dataset_name}")
    checkpoint_dir = os.path.join("checkpoints", dataset_name)

    # ⚙️ Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 📥 Carica dataset
    embedding_dataset = EmbeddingDataset(embedding_dir)
    embedding_loader = DataLoader(embedding_dataset, batch_size=1, shuffle=False)

    # 📊 Estrai embedding
    dataloader = DataLoader(EmbeddingDataset(embedding_dir), batch_size=1, shuffle=False)
    X_real, y_real = extract_embeddings_and_labels(dataloader, device)

    # 🧠 Addestramento
    model = train_classifier(X_real, y_real, input_dim=X_real.shape[1], device=device)


    # 💾 Salvataggio nella cartella corretta
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "classifier.pth"))
    print(f"✅ Classificatore salvato in {checkpoint_dir}/classifier.pth")
   