#evaluate_rl_agent.py
import torch
from torch.utils.data import DataLoader
from dataset.embedding_dataset import EmbeddingDataset
from dataset.embedding_dataset_clap2016 import Clap2016EmbeddingDataset
from models.progressive_rl import ProgressiveRLAgent
from models.lra_gnn import LRA_GNN
from training.rl_environment import RLEnvironment
from models.classifier import AgeGroupClassifier
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np



# 🔧 Parsing argomento --dataset
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="MORPH", help="Dataset: MORPH, FGNET o UTKFACE o CLAP2016")
args = parser.parse_args()
dataset_name = args.dataset.upper()

# 📌 Parametri
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# 📂 Directory embeddings in base al dataset
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
checkpoint_dir = os.path.join("checkpoints", dataset_name)
# 🔍 Cerca tutti i best_agent con timestamp nella cartella
agent_files = glob.glob(os.path.join(checkpoint_dir, "best_agent_*.pth"))

if not agent_files:
    raise FileNotFoundError("❌ Nessun file best_agent_*.pth trovato in checkpoints!")

# 📅 Ordina per timestamp (in base al nome)
agent_files.sort()  # ordine alfabetico equivale a ordine temporale grazie al formato YYYY-MM-DD_HH-MM-SS

# 🆕 Prende l'ultimo
agent_path = agent_files[-1]

print(f"📂 File best_agent selezionato: {agent_path}")

# ⚙️ Hyperparametri coerenti col training

#state_dim = 134  # esempio: embedding (128) + delta_x + delta_y + pos
action_dim = 5   # su, giù, sinistra, destra, resta

# 🔁 Carica il dataset
if dataset_name == "CLAP2016":
    dataset = Clap2016EmbeddingDataset(
        embeddings_dir=embedding_dir,
        dataset_name="CLAP2016",  # <<--- IMPORTANTISSIMO
        # opzionale ma consigliato per essere espliciti:
        clap2016_csv=os.path.join("datasets","data","CLAP2016","CLAP_complete_val.csv")
    )
else:
    dataset = EmbeddingDataset(embedding_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
sample = next(iter(dataloader))
embedding_dim = sample[0].shape[1]
extra_features = 6
state_dim = embedding_dim + extra_features
print(f"🔎 Dimensione embedding rilevata: {embedding_dim} → state_dim: {state_dim}")
# 🔍 Debug dimensione embedding
sample = next(iter(dataloader))
print(f"🔎 Dimensione embedding rilevata: {sample[0].shape[1]}")
# 🎯 Carica il modello LRA-GNN (serve per feature estratte se non già incluse)
model = LRA_GNN()  # se serve per il forward
model.to(device)
model.eval()
# 🔢 Conta il numero di parametri del modello LRA-GNN
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
num_params_m = num_params / 1e6  # converti in milioni
print(f"📊 Parametri totali LRA-GNN: {num_params} → {num_params_m:.1f}M")

# 🔁 Carica l’agente RL
agent = ProgressiveRLAgent(state_dim=state_dim, action_dim=action_dim)
# 📥 Carica il modello salvato
agent_files = glob.glob(os.path.join(checkpoint_dir, "best_agent_*.pth"))
if not agent_files:
    raise FileNotFoundError("❌ Nessun file best_agent_*.pth trovato in checkpoints!")
agent_files.sort()
agent_path = agent_files[-1]
print(f"📂 File best_agent selezionato: {agent_path}")

#agent.q_network.load_state_dict(torch.load(agent_path, map_location=device))
# 📥 Carica il checkpoint del QNetwork
checkpoint = torch.load(agent_path, map_location=device)

# ✅ Controllo di compatibilità (solo warning se mismatch)
fc1_weight_shape = checkpoint['fc1.weight'].shape if 'fc1.weight' in checkpoint else None
expected_shape = agent.q_network.fc1.in_features

if fc1_weight_shape and fc1_weight_shape[1] != expected_shape:
    print(f"⚠️ WARNING: QNetwork input dim mismatch: checkpoint={fc1_weight_shape[1]} vs model={expected_shape}")
    print("ℹ️ Questo potrebbe indicare che il checkpoint è stato salvato con un modello addestrato con dimensione diversa (es. ablation vs full).")
    print("❗ Valuta se rieseguire l'addestramento RL con il nuovo modello.")
else:
    agent.q_network.load_state_dict(checkpoint)  # Carica solo se tutto ok


#agent.q_network.load_state_dict(torch.load("checkpoints/best_agent.pth"))
agent.q_network.to(device)  # 👈 Sposta la rete su MPS o CUDA

print(agent.q_network.fc1.weight[0][:5])  # ad esempio

# ➕ Inizializza il classificatore e assegna all'agente
embedding_dim = next(iter(dataloader))[0].shape[1]
classifier_path = os.path.join(checkpoint_dir, "classifier.pth")
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
df_compare = pd.DataFrame({"True Age": y_true, "Predicted Age": y_pred})
df_compare.to_csv("output/true_vs_predicted_morph.csv", index=False)
print("✅ Salvato file output/true_vs_predicted_morph.csv")