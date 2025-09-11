import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.embedding_dataset import EmbeddingDataset
from models.classifier import AgeGroupClassifier
import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#si lancia così PYTORCH_ENABLE_MPS_FALLBACK=1 python -m utils.evaluate_classifier --dataset MORPH

#output atteso ✅ Accuracy del classificatore su MORPH: 84.23% (842/1000)

# Argomenti da linea di comando
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="MORPH", help="Dataset (MORPH o FGNET)")
args = parser.parse_args()
dataset_name = args.dataset.upper()

# Dispositivo
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Dataset e DataLoader
embedding_dir = "embeddings/train" if dataset_name == "MORPH" else "embeddings_FGNET/train"
embedding_dataset = EmbeddingDataset(embedding_dir)
embedding_loader = DataLoader(embedding_dataset, batch_size=1, shuffle=False)

# Carica modello
model = AgeGroupClassifier(input_dim=embedding_dataset[0][0].shape[0]).to(device)
model_path = os.path.join("checkpoints", dataset_name, "classifier.pth")

if not os.path.exists(model_path):
    print(f"❌ Classificatore non trovato: {model_path}")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Valutazione
correct = 0
total = 0
with torch.no_grad():
    for embedding, age in embedding_loader:
        embedding = embedding.to(device)
        age = age.to(device)
        target_group = (age.item() // 10)
        
        logits = model(embedding)
        pred_group = torch.argmax(F.softmax(logits, dim=1), dim=1).item()

        if pred_group == target_group:
            correct += 1
        total += 1

accuracy = 100 * correct / total if total > 0 else 0
print(f"\n✅ Accuracy del classificatore su {dataset_name}: {accuracy:.2f}% ({correct}/{total})")
