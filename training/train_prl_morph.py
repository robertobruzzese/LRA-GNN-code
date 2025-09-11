# train_prl_morph.py
import  argparse, time, json
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.embedding_dataset import EmbeddingDataset  # <-- usa il tuo loader
from models.progressive_rl_prl import ProgressiveRLAgent
from models.classifier import AgeGroupClassifier

# ----- utils metriche -----
def mae(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))

def cs_at_k(y_true, y_pred, k=5):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float((np.abs(y_true - y_pred) <= k).mean() * 100.0)

def seed_everything(s=42):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# ----- argomenti -----
p = argparse.ArgumentParser()
p.add_argument("--train-dir", required=True, help="es. embeddings_morph/train")
p.add_argument("--val-dir",   required=True, help="es. embeddings_morph/val")
p.add_argument("--run-name",  default="MORPH_prl", help="cartella run in checkpoints/")
p.add_argument("--epochs",    type=int, default=10)
p.add_argument("--batch-size",type=int, default=64)    # batch per update dal replay
p.add_argument("--max-steps", type=int, default=100)   # passi per episodio
p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--lr",        type=float, default=1e-5)
p.add_argument("--hidden",    type=int, default=256)
p.add_argument("--dropout",   type=float, default=0.0)
p.add_argument("--eta",       type=float, default=0.4)  # PRLAE eta (0.5 su MORPH se vuoi)
p.add_argument("--tau-soft",  type=float, default=0.005)
p.add_argument("--eps-decay", type=int,   default=50000)
p.add_argument("--use-classifier-start", action="store_true",
               help="usa il classificatore per scegliere la riga iniziale (allenato SOLO su TRAIN)")
p.add_argument("--classifier-ckpt", default="", help="es. checkpoints/MORPH/classifier.pth")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

seed_everything(args.seed)
device = torch.device(args.device)

# ----- dataset & loader -----
train_set = EmbeddingDataset(args.train_dir, dataset_name="MORPH", return_dict=False)
val_set   = EmbeddingDataset(args.val_dir,   dataset_name="MORPH", return_dict=False)

if len(train_set) == 0 or len(val_set) == 0:
    raise RuntimeError("Dataset train/val vuoti. Controlla i percorsi agli embeddings.")

train_loader = DataLoader(train_set, batch_size=1, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False, num_workers=0)

# ----- dimensione stato -----
x0, _ = train_set[0]
embedding_dim = int(x0.shape[-1])
state_dim = embedding_dim + 2
action_dim = 5

# ----- agente PRL -----
agent = ProgressiveRLAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=args.hidden,
    dropout=args.dropout,
    learning_rate=args.lr,
    tau_soft=args.tau_soft,
    loss_eta=args.eta,
    device=device,
)

# ----- classifier (opzionale) SOLO per start row -----
if args.use_classifier_start:
    if not args.classifier_ckpt or not os.path.isfile(args.classifier_ckpt):
        print("⚠️  --use-classifier-start attivo ma --classifier-ckpt mancante: userò start random.")
        classifier = None
    else:
        classifier = AgeGroupClassifier(input_dim=embedding_dim).to(device)
        classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
        classifier.eval()
        agent.classifier = classifier

# ----- conteggi decade SOLO su TRAIN -----
counts_train = ProgressiveRLAgent.compute_counts_from_loader(train_loader)
agent.set_class_counts(counts_train)
print(f"📊 Class counts (train): {counts_train} | majority={agent.majority_class_count}")

# ----- checkpoints -----
ckpt_dir = os.path.join("checkpoints", args.run_name)
os.makedirs(ckpt_dir, exist_ok=True)

best_mae, best_path = float("inf"), None

for ep in range(1, args.epochs + 1):
    print(f"\n===== Epoch {ep}/{args.epochs} =====")

    # --- TRAIN: un'epoca di interazione + update dal replay ---
    mean_loss, n_upd = agent.train_one_epoch(
        dataloader=train_loader,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=args.eps_decay,
        updates_per_step=1,
        use_classifier_start=args.use_classifier_start,
        clip_reward=10.0,   # piccolo clip per stabilità; puoi alzarlo/abbassarlo
    )
    print(f"🛠️  train: mean_loss={mean_loss:.4f} | updates={n_upd}")

    # --- VALIDATION: eval pulita (greedy) ---
    with torch.no_grad():
        res = agent.evaluate(
            model=None,
            dataloader=val_loader,
            device=device,
            max_steps=args.max_steps,
            start_mode="classifier" if args.use_classifier_start and agent.classifier is not None else "random",
        )
    y_true = np.array(res["true_ages"], float)
    y_pred = np.array(res["predicted_ages"], float)
    val_mae = mae(y_true, y_pred)
    val_cs5 = cs_at_k(y_true, y_pred, k=5)
    print(f"🔍 val: MAE={val_mae:.2f} | CS@5={val_cs5:.2f}% | decade_acc={res['decade_accuracy']:.2f}%")

    # --- salva checkpoint di epoca ---
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ep_ckpt = os.path.join(ckpt_dir, f"rl_agent_ep{ep:03d}_{ts}.pth")
    agent.save(ep_ckpt)
    print(f"💾 salvato: {ep_ckpt}")

    # --- aggiorna best per MAE su val ---
    if val_mae < best_mae:
        best_mae, best_path = val_mae, ep_ckpt
        best_alias = os.path.join(ckpt_dir, "best_agent_mae.pth")
        torch.save(torch.load(ep_ckpt, map_location="cpu"), best_alias)
        with open(os.path.join(ckpt_dir, "BEST_SELECTED.json"), "w") as f:
            json.dump({"best_path": best_path, "best_mae": float(best_mae)}, f, indent=2)
        print(f"🏆 nuovo best (MAE {best_mae:.2f}) → {best_alias}")

print("\n✅ Training completato.")
if best_path:
    print(f"📌 Best MAE su val = {best_mae:.2f} | file: {best_path}")