# evaluate_rl_agent_no_ablation_fgnet.py
import os
import sys
import re
import json
import glob
import time
import random
import argparse
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)

# ==== Dataset loaders ====
from dataset.embedding_dataset import EmbeddingDataset               # per MORPH/FGNET flat
from dataset.embedding_dataset_utkface import UtkfaceEmbeddingDataset
from dataset.embedding_dataset_clap2016 import Clap2016EmbeddingDataset

# ==== Agent & classifier ====
from models.progressive_rl_no_ablation import ProgressiveRLAgent      # <-- NO-ABLATION (2 feature)
from models.classifier import AgeGroupClassifier


# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="FGNET",
                    help="Dataset: MORPH, FGNET, UTKFACE o CLAP2016")
parser.add_argument("--quick-n", type=int, default=200,
                    help="Campioni usati per la valutazione rapida")
parser.add_argument("--top-k", type=int, default=3,
                    help="Quanti checkpoint migliori rivalutare sul val completo (usa -1 per TUTTI)")
parser.add_argument("--quick-mode", type=str, default="uniform",
                    choices=["uniform", "random", "head", "stratified"],
                    help="Criterio di scelta del sottoinsieme rapido")
parser.add_argument("--quick-seed", type=int, default=42,
                    help="Seed per la selezione random/stratified")
parser.add_argument("--eval-only", action="store_true",
                    help="Stampa MAE/accuracy e termina (senza grafici).")
args = parser.parse_args()

EVAL_ONLY = bool(args.eval_only)
if EVAL_ONLY:
    import matplotlib
    matplotlib.use("Agg")
    plt.ioff()

dataset_name = args.dataset.upper()

# --------------- Utils ---------------
def pause_and_exit():
    try:
        input("\n⏸  Premi INVIO per uscire...")
    except (EOFError, KeyboardInterrupt):
        pass
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

def pause(msg="\n⏸  Premi INVIO per continuare..."):
    sys.stdout.flush()
    try:
        if sys.stdin.isatty():
            input(msg)
        else:
            time.sleep(3)
    except (EOFError, KeyboardInterrupt):
        time.sleep(3)

def show_if_needed():
    if not EVAL_ONLY:
        plt.show()
    else:
        plt.close('all')

def last_linear(m):
    return [x for x in m.modules() if isinstance(x, nn.Linear)][-1]

def midpoints_for(dataset, C):
    # midpoints robusti rispetto #classi
    start_decade = 0
    if dataset == "MORPH":
        start_decade = 0 if C >= 9 else 10
    return np.arange(start_decade + 5, start_decade + 5 + 10*C, 10, dtype=float)

def mae_from_results(res, dataset_name):
    y_true = np.array(res["true_ages"], dtype=float)
    y_pred = np.array(res.get("predicted_ages", []), dtype=float)

    if y_pred.size == 0 and "pred_rows" in res and "pred_cols" in res:
        y_pred = (np.array(res["pred_rows"], dtype=int) * 10
                  + np.array(res["pred_cols"], dtype=int))
    if y_pred.size == 0 and "predicted_labels" in res and len(res["predicted_labels"]) > 0:
        labs = np.array(res["predicted_labels"], dtype=int)
        C_rl = int(labs.max()) + 1
        mid_rl = midpoints_for(dataset_name, C_rl)
        y_pred = mid_rl[labs]
    if y_pred.size == 0:
        raise RuntimeError("Impossibile calcolare MAE: nessuna predizione continua disponibile.")
    return float(np.mean(np.abs(y_true - y_pred)))

def extract_y_true_pred(res, dataset_name):
    y_true = np.array(res["true_ages"], dtype=float)
    y_pred = np.array(res.get("predicted_ages", []), dtype=float)
    if y_pred.size == 0 and "pred_rows" in res and "pred_cols" in res:
        y_pred = (np.array(res["pred_rows"], dtype=int) * 10
                  + np.array(res["pred_cols"], dtype=int))
    if y_pred.size == 0 and "predicted_labels" in res and len(res["predicted_labels"]) > 0:
        labs = np.array(res["predicted_labels"], dtype=int)
        C_rl = int(labs.max()) + 1
        mid_rl = midpoints_for(dataset_name, C_rl)
        y_pred = mid_rl[labs]
    return y_true, y_pred

def compute_cs(y_true, y_pred, k=5):
    return float((np.abs(y_true - y_pred) <= k).mean() * 100.0)

def compute_epsilon_error(y_true, y_pred, sigma=5):
    sq = (y_true - y_pred) ** 2
    return float(1.0 - np.exp(-sq / (2.0 * sigma ** 2)).mean())


# --------------- Device ---------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# --------- Embeddings & checkpoints ---------
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

# ✅ RL ckpt nella cartella del run “pulito 514”
run_name = f"{dataset_name}_clean514"
checkpoint_dir = os.path.join("checkpoints", run_name)

# --------------- Dataset loader ---------------
if dataset_name == "CLAP2016":
    dataset = Clap2016EmbeddingDataset(
        embeddings_dir=embedding_dir,
        dataset_name="CLAP2016",
        clap2016_csv=os.path.join("datasets", "data", "CLAP2016", "CLAP_complete_val.csv")
    )
elif dataset_name == "UTKFACE":
    dataset = UtkfaceEmbeddingDataset(embedding_dir)
else:  # MORPH, FGNET → embeddings piatti
    dataset = EmbeddingDataset(
        embeddings_dir=embedding_dir,
        dataset_name=dataset_name,
        return_dict=False
    )

if len(dataset) == 0:
    raise RuntimeError(f"❌ Dataset vuoto in {embedding_dir}")

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ---- State dim (2 feature: pos_r, pos_c) ----
x0, _age0 = dataset[0]
embedding_dim = x0.shape[-1]
extra_features = 2
state_dim = embedding_dim + extra_features
action_dim = 5
print(f"🔎 Dimensione embedding: {embedding_dim} → state_dim: {state_dim} (2 feature)")

# --------------- Agent & Classifier ---------------
agent = ProgressiveRLAgent(state_dim=state_dim, action_dim=action_dim)

# classifier SEMPRE nella cartella base del dataset
classifier_path = os.path.join("checkpoints", dataset_name, "classifier.pth")
classifier = AgeGroupClassifier(input_dim=embedding_dim).to(device)
classifier.load_state_dict(torch.load(classifier_path, map_location=device))
classifier.eval()
agent.classifier = classifier

agent.q_network.to(device)
agent.q_network.eval()

# --------------- Selezione checkpoint ---------------
def _ts_from_name(path):
    m = re.search(r"best_agent_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.pth$", os.path.basename(path))
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return datetime.min

all_ckpts = [p for p in glob.glob(os.path.join(checkpoint_dir, "best_agent_*.pth")) if os.path.isfile(p)]
all_ckpts_sorted = sorted(all_ckpts, key=_ts_from_name)
RECENT_K = 15
agent_files = all_ckpts_sorted[-RECENT_K:] if len(all_ckpts_sorted) > RECENT_K else all_ckpts_sorted

print(f"🔎 Trovati {len(all_ckpts)} checkpoint in {checkpoint_dir} — uso gli ultimi {len(agent_files)}:")
print("   ", [os.path.basename(p) for p in agent_files])
if not agent_files:
    raise FileNotFoundError(f"Nessun best_agent_*.pth in {checkpoint_dir}")

# --------------- Quick subset ---------------
total_n = len(dataset)
quick_n = min(args.quick_n, total_n)
np.random.seed(args.quick_seed)
random.seed(args.quick_seed)
torch.manual_seed(args.quick_seed)

def _infer_age_from_item(item):
    if isinstance(item, (list, tuple)) and len(item) > 1:
        meta = item[1]
        if isinstance(meta, dict) and "age" in meta:
            a = meta["age"]
            if torch.is_tensor(a):
                return float(a.item())
            try:
                return float(a)
            except Exception:
                return None
        elif torch.is_tensor(meta):
            return float(meta.item()) if meta.numel() == 1 else float(meta[0].item())
        elif isinstance(meta, (int, float, np.number)):
            return float(meta)
    return None

rng = np.random.default_rng(args.quick_seed)

if args.quick_mode == "head":
    quick_idx = list(range(quick_n))
elif args.quick_mode == "uniform":
    quick_idx = np.linspace(0, total_n - 1, quick_n, dtype=int).tolist()
elif args.quick_mode == "random":
    quick_idx = rng.choice(total_n, size=quick_n, replace=False).tolist()
elif args.quick_mode == "stratified":
    decade_bins = defaultdict(list)
    for i in range(total_n):
        try:
            age = _infer_age_from_item(dataset[i])
        except Exception:
            age = None
        if age is None:
            continue
        decade_bins[int(age // 10)].append(i)
    if not decade_bins:
        quick_idx = rng.choice(total_n, size=quick_n, replace=False).tolist()
    else:
        per_bin = max(1, quick_n // len(decade_bins))
        quick_idx = []
        for _, idxs in sorted(decade_bins.items()):
            k = min(per_bin, len(idxs))
            if k > 0:
                pick = rng.choice(len(idxs), size=k, replace=False)
                quick_idx.extend([idxs[j] for j in pick])
        remaining = quick_n - len(quick_idx)
        if remaining > 0:
            pool = list(set(range(total_n)) - set(quick_idx))
            add = rng.choice(len(pool), size=min(remaining, len(pool)), replace=False)
            quick_idx.extend([pool[j] for j in add])
else:
    quick_idx = list(range(quick_n))

quick_idx = sorted(set(int(i) for i in quick_idx))
quick_subset = Subset(dataset, quick_idx)
quick_loader = DataLoader(quick_subset, batch_size=1, shuffle=False)
print(f"⚡ Quick subset: {len(quick_idx)}/{total_n} campioni ({args.quick_mode})")

# --------------- Fase 1: quick eval ---------------
scores = []  # (mae_q, acc_dec_q, ckpt_path)
for p in agent_files:
    state_raw = torch.load(p, map_location=device)
    state = state_raw.get("q_network_state_dict", state_raw.get("state_dict", state_raw))

    clean_state = { (k[len("q_network."):] if k.startswith("q_network.") else k): v
                    for k, v in state.items() }

    if "fc1.weight" in clean_state and clean_state["fc1.weight"].shape[1] != agent.q_network.fc1.in_features:
        print(f"⏭️ Skip {os.path.basename(p)}: input dim mismatch "
              f"({clean_state['fc1.weight'].shape[1]} vs {agent.q_network.fc1.in_features})")
        continue

    agent.q_network.load_state_dict(clean_state, strict=False)
    agent.q_network.to(device); agent.q_network.eval()

    print(f"🔎 (quick) Valuto ckpt: {os.path.basename(p)}")
    pause("⏸  (quick) Premi INVIO per valutare il prossimo checkpoint...")

    try:
        ll = last_linear(agent.q_network)
        if ll.bias is not None:
            print("bias_mean:", float(ll.bias.detach().mean()))
    except IndexError:
        pass

    res_q = agent.evaluate(model=None, dataloader=quick_loader, device=device)

    acc_dec_q = float(res_q.get("decade_accuracy", 0))
    if acc_dec_q < 1:
        acc_dec_q *= 100

    mae_q = mae_from_results(res_q, dataset_name)
    print(f"   → (quick) {os.path.basename(p)}: acc = {acc_dec_q:.2f}% | MAE = {mae_q:.2f}")
    scores.append((mae_q, acc_dec_q, p))

if not scores:
    raise RuntimeError("Nessun checkpoint valido dopo la shortlist.")

scores.sort(key=lambda x: (x[0], -x[1]))  # MAE crescente, tie: acc decrescente

FULL_K = args.top_k if args.top_k != -1 else len(scores)
raw_candidates = scores[:min(FULL_K, len(scores))]
candidates = [p for (_, _, p) in raw_candidates if os.path.isfile(p)]
missing = [p for (_, _, p) in raw_candidates if not os.path.isfile(p)]
if missing:
    print("⚠️ Skip perché mancanti:", [os.path.basename(x) for x in missing])

print("🏁 Candidati FULL:", [os.path.basename(x) for x in candidates])

# --------------- Fase 2: full eval ---------------
best_path, best_mae, best_acc = None, float("inf"), -1.0
best_cs5, best_eps = -1.0, float("inf")
for p in candidates:
    state_raw = torch.load(p, map_location=device)
    state = state_raw.get("q_network_state_dict", state_raw.get("state_dict", state_raw))
    clean_state = { (k[len("q_network."):] if k.startswith("q_network.") else k): v for k, v in state.items() }

    agent.q_network.load_state_dict(clean_state, strict=False)
    agent.q_network.to(device); agent.q_network.eval()

    print(f"🔎 (full) Valuto ckpt: {os.path.basename(p)}")
    try:
        ll = last_linear(agent.q_network)
        if ll.bias is not None:
            print("bias_mean:", float(ll.bias.detach().mean()))
    except IndexError:
        pass

    res = agent.evaluate(model=None, dataloader=dataloader, device=device)

    acc_dec = float(res.get("decade_accuracy", 0))
    if acc_dec < 1:
        acc_dec *= 100

    mae = mae_from_results(res, dataset_name)
    y_true_full, y_pred_full = extract_y_true_pred(res, dataset_name)
    cs5 = compute_cs(y_true_full, y_pred_full, k=5)
    eps = compute_epsilon_error(y_true_full, y_pred_full, sigma=5)

    print(f"   → (full) {os.path.basename(p)}: acc = {acc_dec:.2f}% | MAE = {mae:.2f}")

    if (mae < best_mae) or (mae == best_mae and acc_dec > best_acc):
        best_mae, best_acc, best_path = mae, acc_dec, p
        best_cs5, best_eps = cs5, eps

print(
    f"✅ Miglior checkpoint (per MAE): {os.path.basename(best_path)} — "
    f"MAE = {best_mae:.2f} | CS@5 = {best_cs5:.2f}% | ε-error = {best_eps:.4f} | "
    f"decade acc = {best_acc:.2f}%"
)

if best_path is None:
    raise RuntimeError("Nessun checkpoint selezionato come best.")

meta = {"best_path": best_path, "mae": float(best_mae), "decade_acc_pct": float(best_acc)}
with open(os.path.join(checkpoint_dir, "BEST_SELECTED.json"), "w") as f:
    json.dump(meta, f, indent=2)

if EVAL_ONLY:
    pause("\n⏸  Modalità eval-only: premi INVIO per uscire...")
    raise SystemExit(0)

import shutil
dst = os.path.join(checkpoint_dir, "best_agent_SELECTED.pth")
try:
    shutil.copy2(best_path, dst)
    print(f"💾 Copiato il best in: {dst}")
except Exception as e:
    print(f"⚠️ Copia best fallita: {e}")

# ricarica best definitivo
best_raw = torch.load(best_path, map_location=device)
best_state = best_raw.get("q_network_state_dict", best_raw.get("state_dict", best_raw))
best_clean = { (k[len("q_network."):] if k.startswith("q_network.") else k): v for k, v in best_state.items() }
agent.q_network.load_state_dict(best_clean, strict=False)
agent.q_network.to(device)
agent.q_network.eval()

# ---- Param count (Q + classifier) ----
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
q_params  = count_parameters(agent.q_network)
cls_params = count_parameters(classifier)
num_params_m = (q_params + cls_params) / 1e6
print(f"📊 Parametri totali: Q-Network={q_params:,} + Classifier={cls_params:,} → {num_params_m:.1f}M")

# ================= Baseline classifier =================
clf_true_labels, clf_pred_labels = [], []
clf_true_ages,  clf_pred_ages  = [], []

classifier.eval()
with torch.no_grad():
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            emb = batch[0].to(device)
            meta = batch[1] if len(batch) > 1 else None
        else:
            emb = batch.to(device)
            meta = None

        logits = classifier(emb)                 # [1, C]
        pred_lab = logits.argmax(dim=1).item()

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        C = probs.shape[0]
        midpoints = midpoints_for(dataset_name, C)
        pred_age_soft = float((probs * midpoints).sum())

        true_age = None
        if isinstance(meta, dict) and "age" in meta:
            ta = meta["age"]
            true_age = float(ta.item() if torch.is_tensor(ta) else ta)
        elif torch.is_tensor(meta):
            true_age = float(meta.item()) if meta.numel() == 1 else float(meta[0].item())
        if true_age is None:
            continue

        true_lab = int(true_age // 10)
        clf_true_labels.append(true_lab)
        clf_pred_labels.append(pred_lab)
        clf_true_ages.append(true_age)
        clf_pred_ages.append(pred_age_soft)

clf_acc = accuracy_score(clf_true_labels, clf_pred_labels) * 100
baseline_mae = np.mean(np.abs(np.array(clf_true_ages) - np.array(clf_pred_ages)))
baseline_cs5 = np.mean(np.abs(np.array(clf_true_ages) - np.array(clf_pred_ages)) <= 5) * 100
print(f"🔎 Baseline Classifier → Accuracy decade: {clf_acc:.2f}% | MAE: {baseline_mae:.2f} | CS@5: {baseline_cs5:.2f}%")

# ================= RL evaluation =================
results = agent.evaluate(model=None, dataloader=dataloader, device=device)
print("📦 Chiavi results:", results.keys())

predicted_labels_rl = np.array(results.get("predicted_labels", []), dtype=int)

rl_decade_acc = float(results.get("decade_accuracy", 0))
if rl_decade_acc < 1:
    rl_decade_acc *= 100
print(f"🆚 Confronto (decade): Baseline = {clf_acc:.2f}%  |  RL = {rl_decade_acc:.2f}%")

y_true = np.array(results["true_ages"], dtype=float)
y_pred = np.array(results.get("predicted_ages", []), dtype=float)
if y_pred.size == 0 and "pred_rows" in results and "pred_cols" in results:
    y_pred = np.array(results["pred_rows"], dtype=int) * 10 + np.array(results["pred_cols"], dtype=int)
if y_pred.size == 0 and predicted_labels_rl.size:
    C_rl = int(predicted_labels_rl.max()) + 1
    mid_rl = midpoints_for(dataset_name, C_rl)
    y_pred = mid_rl[predicted_labels_rl]

# per-sample CSV
os.makedirs("output", exist_ok=True)
n = min(len(y_true), len(clf_pred_ages), len(predicted_labels_rl))
df_cmp = pd.DataFrame({
    "true_age": y_true[:n],
    "baseline_age_soft": np.array(clf_pred_ages, dtype=float)[:n],
    "rl_age": y_pred[:n],
    "baseline_label": np.array(clf_pred_labels, dtype=int)[:n],
    "rl_label": predicted_labels_rl[:n],
    "baseline_correct_decade": (np.array(clf_true_labels[:n]) == np.array(clf_pred_labels[:n])).astype(int),
    "rl_correct_decade": (np.array(results["true_labels"][:n]) == predicted_labels_rl[:n]).astype(int),
})
csv_cmp = f"output/{dataset_name.lower()}_baseline_vs_rl_per_sample.csv"
df_cmp.to_csv(csv_cmp, index=False)
print(f"✅ Salvato {csv_cmp}")

def compute_mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
def compute_cs(y_true, y_pred, k=5): return np.mean(np.abs(y_true - y_pred) <= k) * 100
def compute_epsilon_error(y_true, y_pred, sigma=5):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return 1 - np.exp(-((y_true - y_pred)**2) / (2 * sigma**2)).mean()

mae = compute_mae(y_true, y_pred)
cs5 = compute_cs(y_true, y_pred, k=5)
eps = compute_epsilon_error(y_true, y_pred)
print(f"\n📐 MAE: {mae:.2f}")
print(f"📊 CS@5: {cs5:.2f}%")
print(f"⚠️ Epsilon-error: {eps:.4f}")

# ---- Tabella confronto paper vs ours ----
if dataset_name == "FGNET":
    paper_entry = {"Dataset": "FGNET", "Method": "LRA-GNN (Paper)", "MAE": 2.14, "CS@5 (%)": 91.6, "Param.": "13M"}
elif dataset_name == "MORPH":
    paper_entry = {"Dataset": "MORPH", "Method": "LRA-GNN (Paper)", "MAE": 2.21, "CS@5 (%)": "-", "Param.": "13M"}
elif dataset_name == "UTKFACE":
    paper_entry = {"Dataset": "UTKFACE", "Method": "LRA-GNN (Paper)", "MAE": "4.22", "CS@5 (%)": "-", "Param.": "13M"}
elif dataset_name == "CLAP2016":
    paper_entry = {"Dataset": "CLAP2016", "Method": "LRA-GNN (Paper)", "MAE": "3.11", "CS@5 (%)": "-", "Param.": "13M"}
else:
    raise ValueError(f"❌ Baseline non disponibile: {dataset_name}")

ours_entry = {
    "Dataset": dataset_name,
    "Method": "LRA-GNN (Ours)",
    "MAE": round(mae, 2),
    "CS@5 (%)": round(cs5, 2),
    "ε-error": round(eps, 4),
    "Param.": f"{(count_parameters(agent.q_network)+count_parameters(classifier))/1e6:.1f}M"
}
df = pd.DataFrame([paper_entry, ours_entry])
print(f"\n📋 Tabella comparativa prestazioni su {dataset_name}:")
print(df.to_string(index=False))

os.makedirs("output", exist_ok=True)
csv_path = f"output/{dataset_name.lower()}_comparison_table.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Salvato in: {csv_path}")

fig, ax = plt.subplots(figsize=(8, 1.6)); ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.5)
plt.title(f"📋 Performance Comparison on {dataset_name}", pad=12)
plt.tight_layout()
png_path = f"output/{dataset_name.lower()}_results_table.png"
plt.savefig(png_path, dpi=300)
print(f"✅ Tabella PNG salvata in: {png_path}")
show_if_needed()

# ---- Report & Confusion matrix ----
true_labels = results["true_labels"]
predicted_labels = results["predicted_labels"]
acc = accuracy_score(true_labels, predicted_labels) * 100
all_labels = sorted(set(true_labels) | set(predicted_labels))
target_names = [f"{i*10}s" for i in all_labels]

print(classification_report(true_labels, predicted_labels, labels=all_labels,
                            target_names=target_names, zero_division=0))

report_dict = classification_report(true_labels, predicted_labels, labels=all_labels,
                                    target_names=target_names, output_dict=True, zero_division=0)
report_table = pd.DataFrame(report_dict).transpose()
fig, ax = plt.subplots(figsize=(12, 5)); ax.axis('off')
t2 = ax.table(cellText=report_table.round(2).values,
              colLabels=report_table.columns,
              rowLabels=report_table.index,
              loc='center', cellLoc='center')
t2.auto_set_font_size(False); t2.set_fontsize(10); t2.scale(1.2, 1.2)
plt.title("📊 Classification Report - RL Agent", pad=20)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/classification_report_table.png", dpi=300)
show_if_needed()

cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix - {dataset_name}")
plt.tight_layout()
plt.savefig("output/confusion_matrix.png", dpi=300)
show_if_needed()

# ---- Smoothed accuracy ----
correctness = [int(t == p) * 100 for t, p in zip(true_labels, predicted_labels)]
window_size = 20
moving_avg = np.convolve(correctness, np.ones(window_size)/window_size, mode='valid')
plt.figure(figsize=(10, 5))
plt.plot(moving_avg, label=f"Moving Average (window={window_size})")
plt.axhline(y=acc, linestyle='--', label=f"Overall Accuracy = {acc:.2f}%")
plt.ylabel("Accuracy (%)"); plt.title("Smoothed Accuracy over Validation Samples")
plt.xlabel("Sample Index"); plt.legend(); plt.grid(True)
os.makedirs("output", exist_ok=True)
plt.savefig("output/smoothed_accuracy.png", dpi=300)
plt.tight_layout()
show_if_needed()

# ---- True vs Pred (CSV) ----
print("\n🧾 Età vere vs predette (prime 30):")
for i, (true, pred) in enumerate(zip(y_true, y_pred)):
    print(f"{i+1:2d}) Età vera: {true:.1f} — Predetta: {pred:.1f}")
    if i >= 29: break

df_compare = pd.DataFrame({"True Age": y_true, "Predicted Age": y_pred})
csv_name = f"output/{dataset_name.lower()}_true_vs_predicted.csv"
df_compare.to_csv(csv_name, index=False)
print(f"✅ Salvato file {csv_name}")