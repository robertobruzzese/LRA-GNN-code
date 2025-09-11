# eval_prl_morph.py
import os, argparse, json, glob
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.embedding_dataset import EmbeddingDataset
from models.progressive_rl_prl import ProgressiveRLAgent
from models.classifier import AgeGroupClassifier

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))

def cs_at_k(y_true, y_pred, k=5):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float((np.abs(y_true - y_pred) <= k).mean() * 100.0)

p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True, help="es. embeddings_morph/val oppure .../test")
p.add_argument("--run-name", default="MORPH_prl", help="cartella run in checkpoints/")
p.add_argument("--ckpt", default="", help="percorso .pth (default: checkpoints/<run-name>/best_agent_mae.pth)")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--max-steps", type=int, default=100)
p.add_argument("--use-classifier-start", action="store_true")
p.add_argument("--classifier-ckpt", default="")
args = p.parse_args()

device = torch.device(args.device)

# ----- dataset/loader -----
dataset = EmbeddingDataset(args.data_dir, dataset_name="MORPH", return_dict=False)
loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
if len(dataset) == 0:
    raise RuntimeError("Dataset vuoto. Controlla --data-dir.")

# ----- dimensione stato -----
x0, _ = dataset[0]
embedding_dim = int(x0.shape[-1])
state_dim = embedding_dim + 2
action_dim = 5

# ----- agente -----
agent = ProgressiveRLAgent(state_dim=state_dim, action_dim=action_dim, device=device)

# ----- classifier opzionale per start row -----
if args.use_classifier_start:
    if not args.classifier_ckpt or not os.path.isfile(args.classifier_ckpt):
        print("⚠️  --use-classifier-start attivo ma --classifier-ckpt mancante: userò start random.")
    else:
        classifier = AgeGroupClassifier(input_dim=embedding_dim).to(device)
        classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
        classifier.eval()
        agent.classifier = classifier

# ----- checkpoint -----
ckpt = args.ckpt
if not ckpt:
    ckpt = os.path.join("checkpoints", args.run_name, "best_agent_mae.pth")
if not os.path.isfile(ckpt):
    # prova l'ultimo salvato
    cand = sorted(glob.glob(os.path.join("checkpoints", args.run_name, "rl_agent_ep*.pth")))
    if cand: ckpt = cand[-1]
if not os.path.isfile(ckpt):
    raise FileNotFoundError(f"Checkpoint non trovato: {ckpt}")

agent.load(ckpt)
print(f"📂 Caricato checkpoint: {ckpt}")

# (Opzionale, ma pulito): imposta qui counts se vuoi mantenerli “da train”
# agent.set_class_counts({...})

# ----- evaluation -----
with torch.no_grad():
    res = agent.evaluate(
        model=None,
        dataloader=loader,
        device=device,
        max_steps=args.max_steps,
        start_mode="classifier" if args.use_classifier_start and agent.classifier is not None else "random",
    )

y_true = np.array(res["true_ages"], float)
y_pred = np.array(res["predicted_ages"], float)
print(f"\n📊 Results on {args.data_dir}")
print(f"   MAE   = {mae(y_true, y_pred):.2f}")
print(f"   CS@5  = {cs_at_k(y_true, y_pred, k=5):.2f}%")
print(f"   Decade Acc = {res['decade_accuracy']:.2f}%")