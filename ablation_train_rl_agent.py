import argparse
import os
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from models.progressive_rl_ablation import ProgressiveRLAgent
from training.train_rl_ablation import train_prlae
from dataset.embedding_dataset_ablation_prlae import EmbeddingDataset
from models.classifier import AgeGroupClassifier as DefaultClassifier
from models.classifier_extended import AgeGroupClassifier as ExtendedClassifier
from models.classifier_deep import AgeGroupClassifier as DeepClassifier
from models.classifier_shallow import AgeGroupClassifier as ShallowClassifier
from datetime import datetime
import glob
import re
from functools import partial

# 🔧 Parser argomenti
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Nome del dataset (MORPH, FGNET, UTKFACE, CLAP2016)")
parser.add_argument("--enable_lrc", action="store_true", help="Abilita LRC")
parser.add_argument("--enable_dfe", action="store_true", help="Abilita DFE")
args = parser.parse_args()
dataset_name = args.dataset.upper()

print("🚀 Launching ablation training with flags:")
print(f"    dataset = {args.dataset}")
print(f"    enable_lrc = {args.enable_lrc}")
print(f"    enable_dfe = {args.enable_dfe}")

# 📁 Directory embeddings ablation
if args.enable_lrc and not args.enable_dfe:
    exp_name = "prlae_lrc_no_dfe"
    embedding_name = "lrc_no_dfe"
elif args.enable_dfe and not args.enable_lrc:
    exp_name = "prlae_no_lrc_dfe"
    embedding_name = "no_lrc_dfe"
else:
    raise ValueError("❌ Devi specificare **solo uno** tra --enable_lrc e --enable_dfe")

embedding_dir = f"embeddings_ablation_{args.dataset.lower()}_{embedding_name}/train"
if not os.path.exists(embedding_dir):
    raise FileNotFoundError(f"❌ Directory embeddings non trovata: {embedding_dir}")

# ⚙️ Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# 📥 Carica dataset di embedding
embedding_dataset = EmbeddingDataset(
    embeddings_dir=embedding_dir,
    dataset_name=args.dataset,
    enable_lrc=args.enable_lrc,
    enable_dfe=args.enable_dfe
)
print(f"📁 Controllo directory embeddings: {embedding_dir}")
print("🧪 Sample count:", len(embedding_dataset))

# 🔧 Collate function per LRC


# 📦 Dataloader
embedding_loader = PyGDataLoader(
    embedding_dataset,
    batch_size=1,
    shuffle=True
)


# 🔢 Dimensione stato
sample = embedding_dataset[0]
print("🔍 Sample type:", type(sample))

# 🔍 DEBUG aggiuntivo
if isinstance(sample, tuple):
    graph, age = sample
    print("📌 graph type:", type(graph))
    print("📌 age:", age)
    if graph is None:
        raise ValueError("❌ Il grafo è None")
else:
    graph = sample
    print("📌 graph type (not tuple):", type(graph))

# Estrai embedding_dim dalla dimensione dei nodi del grafo
if isinstance(sample, tuple):
    graph = sample[0]
else:
    graph = sample

if not hasattr(graph, 'x') or graph.x is None:
    raise ValueError("❌ Il grafo non contiene l'attributo 'x'")

embedding_dim = graph.x.shape[1]
state_dim = embedding_dim + 6


action_dim = 5

# Classificatore
#classifier = AgeGroupClassifier(input_dim=512).to(device)

#classifier = AgeGroupClassifier(input_dim=embedding_dim).to(device)
# Scegli classificatore dinamicamente

use_default = False  # per UTKFACE+DFE NO
use_deep = (
    (dataset_name == "MORPH" and args.enable_lrc) or
    (dataset_name == "CLAP2016" and args.enable_dfe) or
    (dataset_name == "MORPH" and args.enable_dfe)
)
use_shallow = (
    (dataset_name == "FGNET" and args.enable_lrc) or
    (dataset_name == "FGNET" and args.enable_dfe) or
    (dataset_name == "UTKFACE" and args.enable_dfe) or   # <— AGGIUNTO
    (dataset_name == "UTKFACE" and args.enable_lrc)      # se vuoi anche LRC→shallow
)

if use_deep:
    classifier = DeepClassifier(input_dim=embedding_dim).to(device)
elif use_shallow:
    classifier = ShallowClassifier(input_dim=embedding_dim).to(device)
else:
    classifier = DefaultClassifier(input_dim=embedding_dim).to(device)
classifier_path = os.path.join("checkpoints_ablation", args.dataset.lower(), exp_name, "classifier.pth")
if os.path.exists(classifier_path):
    try:
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier = classifier.to(device)  # 🔁 Assicura che TUTTI i pesi siano sul device
        classifier.eval()
        print(f"✅ Classificatore caricato da {classifier_path}")
    except RuntimeError as e:
        print("⚠️ Errore nel caricamento del classificatore:", str(e))
        exit(1)
else:
    print(f"⚠️ Nessun classificatore trovato in {classifier_path}")
    exit(1)

# 🤖 Agente RL
agent = ProgressiveRLAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    device=device,
    classifier=classifier
)
agent.q_network.to(device)

# 🔁 Caricamento checkpoint parziali
checkpoint_dir = os.path.join("checkpoints_ablation", dataset_name.lower(), exp_name)
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoints = glob.glob(os.path.join(checkpoint_dir, "rl_agent_partial_*.pth"))

def extract_episode_num(path):
    match = re.search(r'rl_agent_partial_(\d+)_', path)
    return int(match.group(1)) if match else -1

checkpoints = sorted(checkpoints, key=extract_episode_num)
print("📁 Checkpoints rilevati:", checkpoints)
print(f"📁 Checkpoint trovati in {checkpoint_dir}: {len(checkpoints)} file")
if checkpoints:
    last_checkpoint = checkpoints[-1]
    agent.load(last_checkpoint)
    print(f"📥 Checkpoint caricato da {last_checkpoint}")
    start_step = extract_episode_num(last_checkpoint)
else:
    print("🚀 Nessun checkpoint trovato. Inizio training da zero.")
    start_step = 0
best_model_dir = os.path.join("checkpoints_ablation", dataset_name.lower(), exp_name)
# 🚀 Training RL
if __name__ == "__main__":
    best_accuracy = 0.0

    for step in range(start_step, 200, 50):
        best_accuracy = train_prlae(
            agent=agent,
            dataloader=embedding_loader,
            device=device,
            dataset_name=dataset_name,
            num_episodes=50,
            start_episode=step,
            save_every=10,
            best_accuracy=best_accuracy,
            best_model_dir=best_model_dir,
            classifier=classifier,  # ⬅️ passaggio corretto
            embedding_dim=embedding_dim
        )

       # 🔹 Crea la directory di salvataggio in base al dataset
    #checkpoint_dir = os.path.join("checkpoints_ablation", dataset_name.upper())  # garantisce maiuscolo
    checkpoint_dir = os.path.join("checkpoints_ablation", dataset_name.lower(), exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # 🔹 Costruisci il percorso del file di salvataggio
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(checkpoint_dir, f"rl_agent_{timestamp}.pth")

    # 🔹 Salva l'agente
    agent.save(save_path)
    print(f"\n💾 RL agent salvato in: {save_path}")
print("\n🏁 Training RL completato!")