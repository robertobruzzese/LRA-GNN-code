import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.embedding_dataset import EmbeddingDataset
from models.progressive_rl import ProgressiveRLAgent
from models.classifier import AgeGroupClassifier
from sklearn.metrics import mean_absolute_error
import pandas as pd

# === Configurazione ===
embedding_dir = "embeddings_morph/val"
checkpoint_dir = "checkpoints/MORPH"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# === Carica dataset
dataset = EmbeddingDataset(embedding_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# === Carica dimensione embedding dinamicamente
embedding_dim = next(iter(dataloader))[0].shape[1]

# === Inizializza agente e classifier
agent = ProgressiveRLAgent(state_dim=134, action_dim=5)
agent_path = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("best_agent_")])[-1]
agent.q_network.load_state_dict(torch.load(os.path.join(checkpoint_dir, agent_path), map_location=device))
agent.q_network.to(device)

classifier = AgeGroupClassifier(input_dim=embedding_dim).to(device)
classifier.load_state_dict(torch.load(os.path.join(checkpoint_dir, "classifier.pth"), map_location=device))
classifier.eval()
agent.classifier = classifier

# === Valutazione
results = agent.evaluate(model=None, dataloader=dataloader, device=device)
true_ages = np.array(results["true_ages"])
predicted_ages = np.array(results["predicted_ages"])

# === Fasce di età
bins = [(15, 25), (26, 35), (36, 45), (46, 55), (56, 65), (66, 100)]
labels = [f"{low}-{high}" for (low, high) in bins]

# === Calcola MAE, CS@5 per fascia
rows = []
for (low, high), label in zip(bins, labels):
    mask = (true_ages >= low) & (true_ages <= high)
    if np.any(mask):
        y_true_bin = true_ages[mask]
        y_pred_bin = predicted_ages[mask]
        mae = mean_absolute_error(y_true_bin, y_pred_bin)
        cs5 = np.mean(np.abs(y_true_bin - y_pred_bin) <= 5) * 100
        count = len(y_true_bin)
        rows.append((label, mae, cs5, count))

# === Risultati
df = pd.DataFrame(rows, columns=["Age Group", "MAE", "CS@5 (%)", "Count"])
print("\n📊 Risultati per fascia di età:")
print(df.to_string(index=False))

# === Salvataggio opzionale
os.makedirs("output", exist_ok=True)
df.to_csv("output/morph_per_age_group.csv", index=False)
print("✅ Salvato in output/morph_per_age_group.csv")