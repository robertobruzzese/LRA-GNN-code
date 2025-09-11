#train_rl_agent_no_ablation.py
import argparse
import os
import torch
from torch.utils.data import DataLoader
from models.progressive_rl_no_ablation import ProgressiveRLAgent
from training import train_rl_flat as rl_flat
from dataset.embedding_dataset import EmbeddingDataset
from dataset.embedding_dataset_utkface import UtkfaceEmbeddingDataset
from dataset.embedding_dataset_clap2016 import Clap2016EmbeddingDataset
from models.classifier import AgeGroupClassifier
from datetime import datetime
import glob
import re

# 🔧 Argomento per scegliere il dataset
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="MORPH", help="Nome del dataset (MORPH o FGNET o UTKFACE o CLAP2016)")
args = parser.parse_args()
dataset_name = args.dataset.upper()

# ⚙️ Dispositivo
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 📂 Seleziona cartella embeddings corretta
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

# 📥 Carica dataset di embedding
if dataset_name == "CLAP2016":
    clap_csv = os.path.join("datasets", "data", "CLAP2016", "CLAP_complete_train.csv")
else:
    clap_csv = None
if dataset_name == "CLAP2016":
    clap_csv = os.path.join("datasets", "data", "CLAP2016", "CLAP_complete_train.csv")
    embedding_dataset = Clap2016EmbeddingDataset(
        embeddings_dir=embedding_dir,
        dataset_name="CLAP2016",
        clap2016_csv=clap_csv
    )
elif dataset_name == "UTKFACE":
    embedding_dataset = UtkfaceEmbeddingDataset(embedding_dir)
else:
    embedding_dataset = EmbeddingDataset(
        embeddings_dir=embedding_dir,
        dataset_name=dataset_name,
        return_dict=False
    )

embedding_loader = DataLoader(embedding_dataset, batch_size=1, shuffle=True)

 
# 🔢 Calcola dinamicamente la dimensione dello stato (stato pulito: pos_r,pos_c)
first_embedding, _ = embedding_dataset[0]
embedding_dim = first_embedding.shape[0]
extra_features = 2
state_dim = embedding_dim + extra_features  # 512 + 2 = 514
action_dim = 5

# 🔍 Classificatore pre-addestrato
classifier = AgeGroupClassifier(input_dim=512).to(device)
classifier_path = os.path.join("checkpoints", dataset_name, "classifier.pth")

if os.path.exists(classifier_path):
    try:
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        print(f"✅ Classificatore caricato da {classifier_path}")
    except RuntimeError as e:
        print("⚠️ Errore nel caricamento del classificatore:", str(e))
        print("👉 Esegui di nuovo `train_classifier.py` per rigenerare il file.")
        exit(1)
else:
    print(f"⚠️ Nessun classificatore trovato in {classifier_path}: esegui prima train_classifier.py")
    exit(1)

# 🤖 Istanzia l’agente RL
agent = ProgressiveRLAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    device=device,
    classifier=classifier
)
agent.q_network.to(device)

# 🔁 Carica automaticamente il checkpoint parziale più recente (ordinato per episodio e mtime)
# 🔁 Cartella dedicata per il run “pulito” 514 (nuovi pesi)
run_name = f"{dataset_name}_clean514"
checkpoint_dir = os.path.join("checkpoints", run_name)
os.makedirs(checkpoint_dir, exist_ok=True)
pattern = os.path.join(checkpoint_dir, "rl_agent_partial_*.pth")
checkpoints = glob.glob(pattern)

def extract_episode_num(path):
    m = re.search(r"rl_agent_partial_(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else -1

# ordina: prima per numero episodio, poi per mtime come tie-break
checkpoints = sorted(checkpoints, key=lambda p: (extract_episode_num(p), os.path.getmtime(p)))

if checkpoints:
    last_checkpoint = checkpoints[-1]
    agent.load(last_checkpoint)  # carica i pesi
    start_step = extract_episode_num(last_checkpoint)
    print(f"📥 Checkpoint caricato: {os.path.basename(last_checkpoint)} → riparto dall’episodio {start_step+1}")
else:
    print("🚀 Nessun checkpoint trovato. Inizio training da zero.")
    start_step = 0

# 🔧 Riallinea l'episodio di ripartenza ai multipli di 50 (0,50,100,150)
#if start_step % 50 != 0:
#    aligned = start_step - (start_step % 50)
#    if aligned != start_step:
#        print(f"ℹ️ Checkpoint a ep {start_step} non allineato: riparto da {aligned}")
#    start_step = aligned


# 🚀 Allena l’agente
if __name__ == "__main__":
    best_accuracy = 0.0
    for step in range(start_step, 200, 50):   # quattro blocchi: 0–49, 50–99, 100–149, 150–199
        best_accuracy = rl_flat.train_prlae(
            agent=agent,
            dataloader=embedding_loader,
            device=device,
            dataset_name=run_name,
            num_episodes=50,          # ogni blocco = 50 episodi
            start_episode=step,       # episodio assoluto di partenza
            save_every=1,            # salva solo a fine blocco
            best_accuracy=best_accuracy
        )

    # 💾 Salvataggio finale del modello
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(checkpoint_dir, f"rl_agent_{timestamp}.pth")
    agent.save(save_path)
    print(f"\n💾 RL agent salvato in: {save_path}")

print("\n🏁 Training RL completato!")
