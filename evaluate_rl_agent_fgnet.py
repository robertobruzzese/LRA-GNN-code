#evaluate_rl_agent_utkface.py
import torch
from torch.utils.data import DataLoader
from dataset.embedding_dataset_utkface import UtkfaceEmbeddingDataset
from dataset.embedding_dataset_clap2016 import Clap2016EmbeddingDataset
from dataset.embedding_dataset import EmbeddingDataset
from models.progressive_rl_no_ablation import ProgressiveRLAgent
from models.classifier import AgeGroupClassifier
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np

# 🔧 Argomenti
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="UTKFACE", help="Dataset: MORPH, FGNET, UTKFACE o CLAP2016")
args = parser.parse_args()
dataset_name = args.dataset.upper()

# 📌 Device
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# 📂 Embeddings dir
if dataset_name == "MORPH":
    embedding_dir = "embeddings_morph/val"
elif dataset_name == "FGNET":
    embedding_dir = "embeddings_FGNET/val"
elif dataset_name == "UTKFACE":
    embedding_dir = "embeddings_utkface/val"
elif dataset_name == "CLAP2016":
    embedding_dir = "embeddings_clap2016/val"
else:
    raise ValueError(f"❌ Dataset sconosciuto: {dataset_name}")

run_name = f"{dataset_name}_clean514"
checkpoint_dir = os.path.join("checkpoints", run_name)

# 🔁 Dataset
# 🔁 Dataset
if dataset_name == "CLAP2016":
    dataset = Clap2016EmbeddingDataset(
        embeddings_dir=embedding_dir,
        dataset_name="CLAP2016",
        clap2016_csv=os.path.join("datasets","data","CLAP2016","CLAP_complete_val.csv")
    )
elif dataset_name == "UTKFACE":
    dataset = UtkfaceEmbeddingDataset(embedding_dir)
else:
    # FGNET e MORPH (o altri) usano il dataset generico di embeddings flat
    dataset = EmbeddingDataset(
        embeddings_dir=embedding_dir,
        dataset_name=dataset_name,
        return_dict=False
    )
if len(dataset) == 0:
    raise RuntimeError(f"❌ Dataset vuoto in {embedding_dir}")

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 🔎 embedding_dim dal primo sample (senza consumare il dataloader)
x0, _age0 = dataset[0]
embedding_dim = x0.shape[-1]
extra_features = 2
state_dim = embedding_dim + extra_features
print(f"🔎 Dimensione embedding rilevata: {embedding_dim} → state_dim: {state_dim}")

# 🔁 Carica l’agente RL (ultimo best)
agent_files = glob.glob(os.path.join(checkpoint_dir, "best_agent_*.pth"))
if not agent_files:
    raise FileNotFoundError("❌ Nessun file best_agent_*.pth trovato in checkpoints!")
agent_files.sort()
agent_path = agent_files[-1]
print(f"📂 File best_agent selezionato: {agent_path}")

agent = ProgressiveRLAgent(state_dim=state_dim, action_dim=5)
checkpoint = torch.load(agent_path, map_location=device)

fc1_weight_shape = checkpoint['fc1.weight'].shape if 'fc1.weight' in checkpoint else None
expected_shape = agent.q_network.fc1.in_features
if fc1_weight_shape and fc1_weight_shape[1] != expected_shape:
    print(f"⚠️ WARNING: QNetwork input dim mismatch: checkpoint={fc1_weight_shape[1]} vs model={expected_shape}")
else:
    agent.q_network.load_state_dict(checkpoint)
agent.q_network.to(device)
agent.q_network.eval()
# Conta i parametri del Q-network (per la tabella)
num_params = sum(p.numel() for p in agent.q_network.parameters() if p.requires_grad)
num_params_m = num_params / 1e6
print(f"📊 Parametri Q-network: {num_params} → {num_params_m:.1f}M")
# ➕ Classifier per decade
#classifier_path = os.path.join(checkpoint_dir, "classifier.pth")
# DOPO (sempre nella cartella base del dataset)
classifier_path = os.path.join("checkpoints", dataset_name, "classifier.pth")
classifier = AgeGroupClassifier(input_dim=embedding_dim).to(device)
classifier.load_state_dict(torch.load(classifier_path, map_location=device))
classifier.eval()
agent.classifier = classifier
# 🔬 Ora puoi usarlo per valutare
results = agent.evaluate(model=None, dataloader=dataloader, device=device)
print(results)
print("📦 Chiavi contenute in results:", results.keys())


# Estrai età vere e predette continue
y_true = np.array(results["true_ages"])
y_pred = np.array(results["predicted_ages"])
# 🛡️ Guard-rail anti-leakage + diagnostica
if y_true.shape != y_pred.shape:
    raise RuntimeError(f"❌ Shape mismatch: y_true{y_true.shape} vs y_pred{y_pred.shape}")

print(
    f"🧪 n={len(y_true)} | "
    f"y_true[min,max]=({y_true.min():.2f},{y_true.max():.2f}) | "
    f"y_pred[min,max]=({y_pred.min():.2f},{y_pred.max():.2f})"
)

if np.allclose(y_true, y_pred, atol=1e-8):
    raise RuntimeError("🛑 Leakage: y_pred coincide con y_true (controlla state_tensor in ProgressiveRLAgent.evaluate).")

def _mae(a, b): 
    return float(np.mean(np.abs(a - b)))

mae_raw = _mae(y_true, y_pred)
mae_shuf = _mae(y_true, y_pred[np.random.permutation(len(y_pred))])
corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
print(f"🔁 MAE raw={mae_raw:.3f} | MAE shuffled={mae_shuf:.3f} | Corr={corr:.4f}")
# Definizione metriche
def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def compute_cs(y_true, y_pred, k=5):
    return np.mean(np.abs(y_true - y_pred) <= k) * 100  # percentuale

def compute_epsilon_error(y_true, y_pred, sigma=5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    squared_diff = (y_true - y_pred) ** 2
    exp_term = np.exp(-squared_diff / (2 * sigma ** 2))
    return 1 - np.mean(exp_term)

# Calcolo metriche
mae = compute_mae(y_true, y_pred)
cs5 = compute_cs(y_true, y_pred, k=5)
eps = compute_epsilon_error(y_true, y_pred)

# Stampa risultati
print(f"\n📐 MAE: {mae:.2f}")
print(f"📊 CS@5: {cs5:.2f}%")
print(f"⚠️ Epsilon-error: {eps:.4f}")
import pandas as pd

if dataset_name == "FGNET":
    paper_entry = {
        "Dataset": "FGNET",
        "Method": "LRA-GNN (Paper)",
        "MAE": 2.14,
        "CS@5 (%)": 91.6,
        "Param.": "13M"
    }
elif dataset_name == "MORPH":
    paper_entry = {
        "Dataset": "MORPH",
        "Method": "LRA-GNN (Paper)",
        "MAE": 2.21,
        "CS@5 (%)": "-",  # Nessun valore CS riportato nel paper per MORPH
        "Param.": "13M"
    }
elif dataset_name == "UTKFACE":
    paper_entry = {
        "Dataset": "UTKFACE",
        "Method": "LRA-GNN (Paper)",
        "MAE": "4.22",  # Se non hai valori dal paper
        "CS@5 (%)": "-",
        "Param.": "13M"
    }
elif dataset_name == "CLAP2016":
    paper_entry = {
        "Dataset": "CLAP2016", 
        "Method": "LRA-GNN (Paper)",
        "MAE": "3.11", "CS@5 (%)": "-", 
        "Param.": "13M"}
else:
    raise ValueError(f"❌ Baseline non disponibile per dataset: {dataset_name}")

# 📊 Risultato del tuo modello
ours_entry = {
    "Dataset": dataset_name,
    "Method": "LRA-GNN (Ours)",
    "MAE": round(mae, 2),
    "CS@5 (%)": round(cs5, 2),
    "ε-error": round(eps, 4),
    "Param.": f"{num_params_m:.1f}M"
}

# 🔁 Crea DataFrame completo con baseline prima
df = pd.DataFrame([paper_entry, ours_entry])

# 📋 Stampa tabella
print(f"\n📋 Tabella comparativa prestazioni su {dataset_name}:")
print(df.to_string(index=False))

# 💾 Salva CSV dinamico
os.makedirs("output", exist_ok=True)
csv_path = f"output/{dataset_name.lower()}_comparison_table.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Salvato in: {csv_path}")


# 📊 Crea immagine PNG della tabella comparativa MAE/CS/Parametri
fig, ax = plt.subplots(figsize=(8, 1.6))
ax.axis('off')
table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 loc='center',
                 cellLoc='center')
for key, cell in table.get_celld().items():
    if cell.get_text() is not None:
        cell.get_text().set_fontname("Courier New")  # oppure "Consolas"
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title(f"📋 Performance Comparison on {dataset_name}", pad=12)
plt.tight_layout()

# 💾 Salva immagine dinamicamente
png_path = f"output/{dataset_name.lower()}_results_table.png"
plt.savefig(png_path, dpi=300)
print(f"✅ Tabella PNG salvata in: {png_path}")
plt.show()


true_labels = results["true_labels"]
predicted_labels = results["predicted_labels"]
acc = accuracy_score(true_labels, predicted_labels) * 100
# Calcola le classi realmente presenti
all_labels = sorted(set(true_labels) | set(predicted_labels))  # unione insiemistica
target_names = [f"{i*10}s" for i in all_labels]
# 📋 Report
print(classification_report(true_labels, predicted_labels, labels=all_labels, target_names=target_names, zero_division=0))
# 📊 Classification Report
report_dict = classification_report(true_labels, predicted_labels, labels=all_labels, target_names=target_names, output_dict=True, zero_division=0)
report_table = pd.DataFrame(report_dict).transpose()

# 🖼️ Stampa in tabella
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')
table = ax.table(cellText=report_table.round(2).values,
                 colLabels=report_table.columns,
                 rowLabels=report_table.index,
                 loc='center',
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title("📊 Classification Report - RL Agent", pad=20)
plt.tight_layout()

# 💾 Salvataggio della tabella in output
os.makedirs("output", exist_ok=True)
plt.savefig("output/classification_report_table.png", dpi=300)

plt.show()

# 🔲 Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix - {dataset_name}")
plt.tight_layout()

# 💾 Salvataggio della Confusion Matrix
os.makedirs("output", exist_ok=True)
plt.savefig("output/confusion_matrix.png", dpi=300)

plt.show()

# Calcola correttezza per campione
# Accuracy per sample, in percentuale
correctness = [int(t == p) * 100 for t, p in zip(true_labels, predicted_labels)]

# Calcola accuracy media mobile
window_size = 20
moving_avg = np.convolve(correctness, np.ones(window_size)/window_size, mode='valid')

# Plot più pulito con media mobile
plt.figure(figsize=(10, 5))
plt.plot(moving_avg, label=f"Moving Average (window={window_size})", color='tab:blue')
plt.axhline(y=acc, color='red', linestyle='--', label=f"Overall Accuracy = {acc:.2f}%")
plt.ylabel("Accuracy (%)")
plt.title("Smoothed Accuracy over Validation Samples")
plt.xlabel("Sample Index")
plt.ylabel("Smoothed Accuracy")
plt.legend()
plt.grid(True)

# Salva
os.makedirs("output", exist_ok=True)
plt.savefig("output/smoothed_accuracy.png", dpi=300)
plt.tight_layout()
plt.show()

# 📌 Stampa età vere e predette, riga per riga
print("\n🧾 Età vere vs predette (prime 30):")
for i, (true, pred) in enumerate(zip(y_true, y_pred)):
    print(f"{i+1:2d}) Età vera: {true:.1f} — Predetta: {pred:.1f}")
    if i >= 29:
        break

# 💾 Salvataggio confronto in CSV
# 💾 Salvataggio confronto in CSV (nome dinamico per dataset)
csv_name = f"output/{dataset_name.lower()}_true_vs_predicted.csv"
df_compare = pd.DataFrame({"True Age": y_true, "Predicted Age": y_pred})
df_compare.to_csv(csv_name, index=False)
print(f"✅ Salvato file {csv_name}")