# train_rl_agent_no_ablation_x_mae.py
import argparse, os, re, glob
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from models.progressive_rl_no_ablation_x_mae import ProgressiveRLAgent
from training import train_rl_flat_x_mae as rl_flat
from dataset.embedding_dataset import EmbeddingDataset
from dataset.embedding_dataset_utkface import UtkfaceEmbeddingDataset
from dataset.embedding_dataset_clap2016 import Clap2016EmbeddingDataset
from models.classifier import AgeGroupClassifier

# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="MORPH",
                    help="MORPH | FGNET | UTKFACE | CLAP2016")
parser.add_argument("--episodes", type=int, default=200,
                    help="Episodi totali (multipli di 50)")
parser.add_argument("--block", type=int, default=50,
                    help="Episodi per blocco di salvataggio")
parser.add_argument("--bs", type=int, default=1,
                    help="Batch size (embedding loader)")
args = parser.parse_args()
dataset_name = args.dataset.upper()

# 🔧 device: cuda -> mps -> cpu
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"💻 Device: {device}")

# --------- Embeddings path (train) ---------
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

# --------- Dataset loader ---------
if dataset_name == "CLAP2016":
    clap_csv = os.path.join("datasets", "data", "CLAP2016", "CLAP_complete_train.csv")
    embedding_dataset = Clap2016EmbeddingDataset(
        embeddings_dir=embedding_dir, dataset_name="CLAP2016", clap2016_csv=clap_csv
    )
elif dataset_name == "UTKFACE":
    embedding_dataset = UtkfaceEmbeddingDataset(embedding_dir)
else:
    embedding_dataset = EmbeddingDataset(
        embeddings_dir=embedding_dir, dataset_name=dataset_name, return_dict=False
    )

embedding_loader = DataLoader(
    embedding_dataset, batch_size=args.bs, shuffle=True,
    num_workers=0, pin_memory=(device.type == "cuda")
)

# --------- embedding_dim robusto ---------
first_item = embedding_dataset[0]
# 🔧 gestisci tuple/dict
if isinstance(first_item, (tuple, list)):
    first_embedding = first_item[0]
elif isinstance(first_item, dict):
    # prova chiavi comuni
    first_embedding = first_item.get("embedding", None)
    if first_embedding is None:
        # fallback grezzo
        first_embedding = next(iter(first_item.values()))
else:
    first_embedding = first_item
# 🔧 usa l'ultima dimensione
embedding_dim = int(first_embedding.shape[-1])
print(f"🔎 embedding_dim={embedding_dim}")

# --------- Stato agent (embedding + pos_r,pos_c) ---------
state_dim  = embedding_dim + 2
action_dim = 5

# --------- Classificatore pretrain ---------
classifier_path = os.path.join("checkpoints", dataset_name, "classifier.pth")
classifier = AgeGroupClassifier(input_dim=embedding_dim).to(device)  # 🔧
if os.path.exists(classifier_path):
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    print(f"✅ Classificatore caricato: {classifier_path}")
else:
    raise FileNotFoundError(f"⚠️ Nessun classifier in {classifier_path} — esegui prima train_classifier.py")

# --------- Agente ---------
agent = ProgressiveRLAgent(
    state_dim=state_dim, action_dim=action_dim, device=device, classifier=classifier
)
agent.q_network.to(device)

# --------- Checkpoint dir ---------
run_name = f"{dataset_name}_clean514_x_mae"
checkpoint_dir = os.path.join("checkpoints", run_name)
os.makedirs(checkpoint_dir, exist_ok=True)

####### --------- Ripristino ultimo parziale (se esiste) ---------
# ---- Resume (caricamento ultimo parziale, robusto) ----
pattern = os.path.join(checkpoint_dir, "rl_agent_partial_*.pth")
checkpoints = glob.glob(pattern)

def extract_episode_num(path):
    m = re.search(r"rl_agent_partial_(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else -1

# ordina per numero episodio (primario) e mtime (tie-breaker)
checkpoints = sorted(checkpoints, key=lambda p: (extract_episode_num(p), os.path.getmtime(p)))

start_step = 0
if checkpoints:
    last_checkpoint = checkpoints[-1]
    loaded_ep = agent.load(last_checkpoint)  # int se full-state, 0 se old-style
    if loaded_ep in (None, 0):
        # fallback: episodio dal nome file
        loaded_ep = extract_episode_num(last_checkpoint)
        if loaded_ep < 0:
            loaded_ep = 0
    start_step = max(start_step, int(loaded_ep))
    print(f"📥 Riparto dall’episodio {start_step+1} (da {os.path.basename(last_checkpoint)})")
else:
    print("🚀 Nessun checkpoint trovato. Inizio training da zero.")
########
# --------- Training a blocchi ---------
# --------- Training a blocchi ---------
total_eps = int(args.episodes)
block     = int(args.block)
assert total_eps % block == 0, "episodes deve essere multiplo di block"

best_accuracy = 0.0
for step in range(start_step, start_step + total_eps, block):  # <-- usa start_step
    print(f"\n🏁 Blocco episodi [{step}..{step+block-1}]")
    best_accuracy = rl_flat.train_prlae(
        agent=agent,
        dataloader=embedding_loader,
        device=device,
        dataset_name=run_name,
        num_episodes=block,      # questo blocco allena 'block' episodi
        start_episode=step,      # episodio assoluto di partenza del blocco
        save_every=1,            # vedi nota sotto
        best_accuracy=best_accuracy
    )

    # Salvataggio parziale alla fine del blocco (puoi tenerlo o delegare al trainer)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    end_ep = step + block - 1
    partial_path = os.path.join(checkpoint_dir, f"rl_agent_partial_{end_ep}_{ts}.pth")
    agent.save(partial_path)
    print(f"💾 Salvato parziale: {os.path.basename(partial_path)}")

# --------- Salvataggio finale (facoltativo) ---------
final_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
final_path = os.path.join(checkpoint_dir, f"rl_agent_{final_ts}.pth")
agent.save(final_path)
print(f"\n💾 RL agent (finale) salvato in: {final_path}")

print("\n🏁 Training RL completato!")