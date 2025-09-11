# evaluate_rl_last15_x_mae.py
import os, sys, re, json, glob, shutil, argparse, time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)

# ==== Dataset loaders ====
from dataset.embedding_dataset import EmbeddingDataset               # MORPH/FGNET flat
from dataset.embedding_dataset_utkface import UtkfaceEmbeddingDataset
from dataset.embedding_dataset_clap2016 import Clap2016EmbeddingDataset

# ==== Agent & classifier (NO-ABLATION, 2 feature) ====
#from models.progressive_rl_no_ablation import ProgressiveRLAgent
from models.progressive_rl_no_ablation_x_mae import ProgressiveRLAgent
from models.classifier import AgeGroupClassifier


# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset: MORPH, FGNET, UTKFACE, CLAP2016")
parser.add_argument("--recent-k", type=int, default=15,
                    help="Quanti checkpoint più recenti valutare (default: 15)")
parser.add_argument("--sigma", type=float, default=5.0,
                    help="Sigma per epsilon-error (default: 5)")
parser.add_argument("--out-dir", type=str, default="output",
                    help="Cartella artefatti (default: output)")
parser.add_argument("--no-pause", action="store_true",
                    help="Non fermarti dopo ogni checkpoint")
parser.add_argument("--epsilon-eval", type=float, default=0.0,
                    help="Epsilon (esplorazione) usato in evaluate; es. 0.65 per imitare il training")
args = parser.parse_args()

dataset_name = args.dataset.upper()
RECENT_K = int(args.recent_k)
SIGMA = float(args.sigma)
OUTDIR = args.out_dir
EPS_EVAL = float(args.epsilon_eval)
os.makedirs(OUTDIR, exist_ok=True)


# --------------- Util ----------------
def midpoints_for(dataset, C):
    # Midpoint robusti rispetto al #classi
    if dataset == "MORPH":
        start_decade = 0 if C >= 9 else 10
    else:
        start_decade = 0
    return np.arange(start_decade + 5, start_decade + 5 + 10*C, 10, dtype=float)

def _pause_ckpt():
    """Pausa dopo la stampa dei risultati del checkpoint corrente."""
    try:
        sys.stdout.flush()
        if sys.stdin.isatty():
            input("\n⏸  Premi INVIO per passare al prossimo checkpoint...")
        else:
            time.sleep(3)
    except (EOFError, KeyboardInterrupt):
        pass

def extract_y_true_pred(res, dataset):
    y_true = np.array(res["true_ages"], dtype=float)
    y_pred = np.array(res.get("predicted_ages", []), dtype=float)
    # fallback: row/col → anni
    if y_pred.size == 0 and "pred_rows" in res and "pred_cols" in res:
        y_pred = (np.array(res["pred_rows"], dtype=int) * 10
                  + np.array(res["pred_cols"], dtype=int))
    # fallback: label → midpoint
    if y_pred.size == 0 and "predicted_labels" in res and len(res["predicted_labels"]) > 0:
        labs = np.array(res["predicted_labels"], dtype=int)
        C_rl = int(labs.max()) + 1
        mid_rl = midpoints_for(dataset, C_rl)
        y_pred = mid_rl[labs]
    return y_true, y_pred

def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def compute_cs(y_true, y_pred, k=5):
    return float((np.abs(y_true - y_pred) <= k).mean() * 100.0)

def compute_epsilon_error(y_true, y_pred, sigma=5.0):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(1.0 - np.exp(-((y_true - y_pred) ** 2) / (2.0 * sigma ** 2)).mean())

def _ts_from_name(path):
    # best_agent_YYYY-MM-DD_HH-MM-SS.pth → datetime
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

def clean_and_load_q_state(agent, state_raw, device):
    """
    Supporta:
      - puro state_dict (chiavi 'fc1.weight', ...)
      - {'q_network_state_dict': ...}, {'state_dict': ...}, {'q_network': ...}
      - chiavi prefissate 'q_network.'
    Carica SEMPRE su CPU e POI sposta il modello su device (MPS/cuda/cpu).
    """
    # 1) estrai lo state_dict del Q
    if isinstance(state_raw, dict):
        state = None
        for key in ("q_network_state_dict", "state_dict", "q_network"):
            if key in state_raw and isinstance(state_raw[key], dict):
                state = state_raw[key]
                break
        if state is None:
            if "fc1.weight" in state_raw:   # il dict è già uno state_dict
                state = state_raw
            else:
                print("⏭️  Skip ckpt: dict non riconosciuto come state_dict del Q.")
                return False
    else:
        state = state_raw

    # 2) rimuovi prefisso 'q_network.'
    state = {
        (k[len("q_network."):] if k.startswith("q_network.") else k): v
        for k, v in state.items()
    }

    # 3) controllo dimensione input
    if "fc1.weight" in state:
        in_features_ckpt = state["fc1.weight"].shape[1]
        if in_features_ckpt != agent.q_network.fc1.in_features:
            print(f"⏭️  Skip ckpt: input dim mismatch ({in_features_ckpt} vs {agent.q_network.fc1.in_features})")
            return False

    # 4) *** carica SU CPU ***
    with torch.no_grad():
        agent.q_network.to("cpu")
        state_cpu = {k: v.detach().to("cpu").contiguous() for k, v in state.items()}
        agent.q_network.load_state_dict(state_cpu, strict=False)
        agent.q_network.eval()

        # 5) *** POI sposta su device (MPS/cuda/cpu) ***
        agent.q_network.to(device)
        agent.q_network.eval()

    return True


# --------------- Device ---------------
device = torch.device("cpu")

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

run_name = f"{dataset_name}_clean514_x_mae"    # cartella dei ckpt “puliti 514”
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
else:  # MORPH, FGNET
    dataset = EmbeddingDataset(
        embeddings_dir=embedding_dir,
        dataset_name=dataset_name,
        return_dict=False
    )
if len(dataset) == 0:
    raise RuntimeError(f"❌ Dataset vuoto in {embedding_dir}")

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ---- State dim (2 feature: pos_r, pos_c) ----
x0, _ = dataset[0]
embedding_dim = int(x0.shape[-1])
state_dim = embedding_dim + 2
action_dim = 5
print(f"🔎 Embedding dim: {embedding_dim} → state_dim={state_dim} (2 feature)")

# --------------- Agent & Classifier ---------------
agent = ProgressiveRLAgent(state_dim=state_dim, action_dim=action_dim)

classifier_path = os.path.join("checkpoints", dataset_name, "classifier.pth")
classifier = AgeGroupClassifier(input_dim=embedding_dim)          # crea su CPU
sd = torch.load(classifier_path, map_location="cpu")              # carica su CPU
classifier.load_state_dict(sd)
classifier.eval()                                                 # 👈 resta su CPU

agent.classifier = classifier
agent.q_network.to(device)
agent.q_network.eval()

# #######--------------- Prendi gli ultimi K checkpoint ---------------
# --------------- Selezione e ordinamento checkpoint (rigorosa) ---------------
# --------------- Selezione e ordinamento degli ULTIMI 15 rl_agent_partial_* ---------------
# Esempi attesi:
#   rl_agent_partial_74_2025-08-27_22-47-50.pth
#   rl_agent_partial_51_2025-08-27_05-17-54.pth
pattern_partial = re.compile(r"^rl_agent_partial_\d+_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.pth$")

all_files = glob.glob(os.path.join(checkpoint_dir, "rl_agent_partial_*.pth"))
if not all_files:
    raise FileNotFoundError(f"❌ Nessun rl_agent_partial_*.pth in {checkpoint_dir}")

# Escludi eventuali file “alias” tipo *_SELECTED.pth
exclude_names = {"rl_agent_partial_SELECTED.pth"}
all_files = [p for p in all_files if os.path.basename(p) not in exclude_names]

candidates, skipped = [], []
for p in all_files:
    base = os.path.basename(p)
    m = pattern_partial.match(base)
    if not m:
        skipped.append(base)  # non ha timestamp nel formato atteso → escludo
        continue
    ts = datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")
    candidates.append((p, ts, base))

if not candidates:
    raise FileNotFoundError("❌ Nessun checkpoint rl_agent_partial_* con timestamp valido nel nome.")

# Ordine globale: dal più vecchio al più nuovo
candidates.sort(key=lambda x: x[1])

# Preselezione: ultimi K partial
sel = candidates[-RECENT_K:] if len(candidates) > RECENT_K else candidates
ckpt_list = [p for (p, _ts, _base) in sel]

# ==== PRIORITÀ AI best_agent_mae_* ====
best_mae_files = sorted(
    glob.glob(os.path.join(checkpoint_dir, "best_agent_mae_*.pth")),
    key=os.path.getmtime
)
using_best = False
if best_mae_files:
    print(f"🔍 Trovati {len(best_mae_files)} best_agent_mae_* → li valuto in priorità.")
    # Se vuoi limitarli agli ultimi K, usa: best_mae_files = best_mae_files[-RECENT_K:]
    ckpt_list = best_mae_files
    using_best = True
# =====================================

# Stampa ordine coerente con la sorgente scelta
if using_best:
    print(f"📦 Valuto {len(ckpt_list)} checkpoint best_agent_mae_* (by mtime, oldest → newest):")
    for i, p in enumerate(ckpt_list, 1):
        ts = datetime.fromtimestamp(os.path.getmtime(p))
        print(f"   {i:2d}. {os.path.basename(p)}  |  ts={ts}")
else:
    print(f"📦 Partial con timestamp: {len(candidates)} — ne valuto {len(ckpt_list)} (ultimi).")
    print("   Ordine di valutazione (oldest → newest tra gli ultimi K):")
    for i, (p, ts, base) in enumerate(sel, 1):
        print(f"   {i:2d}. {base}  |  ts={ts}")

if skipped:
    print("ℹ️  Esclusi (nome non conforme):", skipped)

# --------------- Valutazione sequenziale (con pausa opzionale) ---------------
rows = []
best_mae, best_acc, best_path = float("inf"), -1.0, None
best_cache_results = None  # (res, y_true, y_pred)

for idx, p in enumerate(ckpt_list, 1):
    ckpt_name = os.path.basename(p)
    print(f"\n[{idx}/{len(ckpt_list)}] 🔎 Valutazione ckpt: {ckpt_name}")
    state_raw = torch.load(p, map_location="cpu")   # evita placeholder su MPS
    ok = clean_and_load_q_state(agent, state_raw, device)
    if not ok:
        continue

    #res = agent.evaluate(model=None, dataloader=dataloader, device=device)
    res = agent.evaluate(model=None, dataloader=dataloader, device=device, epsilon_eval=EPS_EVAL)
    y_true, y_pred = extract_y_true_pred(res, dataset_name)
    if y_pred.size == 0:
        print("⚠️  Nessuna predizione continua disponibile, skip.")
        continue

    mae = compute_mae(y_true, y_pred)
    cs5 = compute_cs(y_true, y_pred, k=5)
    eps = compute_epsilon_error(y_true, y_pred, sigma=SIGMA)

    # ⬇️ INSERIRE QUI IL BLOCCO DI CORREZIONE PER-DECADE
    dec_true = (y_true // 10).astype(int)
    err = y_pred - y_true
    bias_per_dec = {d: np.median(err[dec_true==d]) for d in np.unique(dec_true)}

    y_pred_corr = np.array([yp - bias_per_dec.get(int(yp//10), 0.0) for yp in y_pred])
    mae_corr = np.mean(np.abs(y_true - y_pred_corr))
    cs5_corr = np.mean(np.abs(y_true - y_pred_corr) <= 5)*100
    print(f"   → (post-hoc per-decade) MAE={mae_corr:.2f} | CS@5={cs5_corr:.2f}%")
    # ⬆️ FINE BLOCCO
    # --- ⬇️ QUI INSERIRE IL POST-HOC LINEARE ---
    A = np.vstack([y_pred, np.ones_like(y_pred)]).T
    a, b = np.linalg.lstsq(A, y_true, rcond=None)[0]
    y_pred_lin = a * y_pred + b
    mae_lin = np.mean(np.abs(y_true - y_pred_lin))
    cs5_lin = np.mean(np.abs(y_true - y_pred_lin) <= 5) * 100
    print(f"   → (post-hoc lineare) MAE={mae_lin:.2f} | CS@5={cs5_lin:.2f}%  | a={a:.3f}, b={b:.3f}")
    # --- ⬆️ FINE BLOCCO ---

    dec_acc = float(res.get("decade_accuracy", np.nan))
    if dec_acc < 1.0:
        dec_acc *= 100.0

    print(f"   → ckpt={ckpt_name} | MAE={mae:.2f} | CS@5={cs5:.2f}% | ε={eps:.4f} | decade_acc={dec_acc:.2f}%")

    rows.append({
        "checkpoint": ckpt_name,
        "path": p,
        "mae": mae,
        "cs5": cs5,
        "epsilon": eps,
        "decade_acc_pct": dec_acc,
    })

    if (mae < best_mae) or (mae == best_mae and dec_acc > best_acc):
        best_mae, best_acc, best_path = mae, dec_acc, p
        best_cache_results = (res, y_true, y_pred)

    if not args.no_pause:
        _pause_ckpt()

# Salva riepilogo per-ckpt
try:
    os.makedirs(OUTDIR, exist_ok=True)
    csv_summary = os.path.join(OUTDIR, f"{dataset_name.lower()}_ckpt_summary.csv")
    pd.DataFrame(rows).sort_values(["mae", "decade_acc_pct"], ascending=[True, False]).to_csv(csv_summary, index=False)
    print(f"💾 Riepilogo per-ckpt salvato in {csv_summary}")
except Exception as e:
    print("⚠️ Salvataggio riepilogo per-ckpt fallito:", e)

# Verifica best
if best_path is None:
    raise RuntimeError("❌ Nessun checkpoint valido selezionato.")

print(f"\n🏆 Best (per MAE): {os.path.basename(best_path)} — MAE={best_mae:.2f} | decade_acc={best_acc:.2f}%")

# Persisti scelta + copia best
meta = {"best_path": best_path, "mae": float(best_mae), "decade_acc_pct": float(best_acc)}
with open(os.path.join(checkpoint_dir, "BEST_SELECTED.json"), "w") as f:
    json.dump(meta, f, indent=2)
dst = os.path.join(checkpoint_dir, "best_agent_SELECTED.pth")
try:
    shutil.copy2(best_path, dst)
    print(f"💾 Copiato il best in: {dst}")
except Exception as e:
    print(f"⚠️ Copia best fallita: {e}")

# --------------- Confronto con baseline classifier ---------------
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

clf_true_labels, clf_pred_labels = [], []
clf_true_ages,  clf_pred_ages  = [], []

with torch.no_grad():
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            emb = batch[0].to(device)
            meta = batch[1] if len(batch) > 1 else None
        else:
            emb = batch.to(device); meta = None

        logits = classifier(emb)                # [1, C]
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_lab = int(probs.argmax())
        C = probs.shape[0]
        mid = midpoints_for(dataset_name, C)
        pred_age_soft = float((probs * mid).sum())

        # true age
        true_age = None
        if isinstance(meta, dict) and "age" in meta:
            ta = meta["age"]; true_age = float(ta.item() if torch.is_tensor(ta) else ta)
        elif torch.is_tensor(meta):
            true_age = float(meta.item()) if meta.numel() == 1 else float(meta[0].item())
        if true_age is None:
            continue

        true_lab = int(true_age // 10)
        clf_true_labels.append(true_lab)
        clf_pred_labels.append(pred_lab)
        clf_true_ages.append(true_age)
        clf_pred_ages.append(pred_age_soft)

clf_acc = accuracy_score(clf_true_labels, clf_pred_labels) * 100.0
baseline_mae = compute_mae(np.array(clf_true_ages), np.array(clf_pred_ages))
baseline_cs5 = compute_cs(np.array(clf_true_ages), np.array(clf_pred_ages), k=5)
baseline_eps = compute_epsilon_error(np.array(clf_true_ages), np.array(clf_pred_ages), sigma=SIGMA)

print("\n🆚 Confronto BASELINE vs RL (best)")
print(f"   Baseline → decade_acc={clf_acc:.2f}% | MAE={baseline_mae:.2f} | CS@5={baseline_cs5:.2f}% | ε={baseline_eps:.4f}")
res_best, y_true_best, y_pred_best = best_cache_results
rl_dec = float(res_best.get("decade_accuracy", np.nan))
if rl_dec < 1.0: rl_dec *= 100.0
print(f"   RL best  → decade_acc={rl_dec:.2f}% | MAE={best_mae:.2f} | CS@5={compute_cs(y_true_best, y_pred_best):.2f}% | ε={compute_epsilon_error(y_true_best, y_pred_best, sigma=SIGMA):.4f}")

# --------------- Artefatti del best ---------------
# Tabella paper vs ours
if dataset_name == "FGNET":
    paper_entry = {"Dataset": "FGNET", "Method": "LRA-GNN (Paper)", "MAE": 2.14, "CS@5 (%)": 91.6, "Param.": "13M"}
elif dataset_name == "MORPH":
    paper_entry = {"Dataset": "MORPH", "Method": "LRA-GNN (Paper)", "MAE": 2.21, "CS@5 (%)": "-", "Param.": "13M"}
elif dataset_name == "UTKFACE":
    paper_entry = {"Dataset": "UTKFACE", "Method": "LRA-GNN (Paper)", "MAE": "4.22", "CS@5 (%)": "-", "Param.": "13M"}
elif dataset_name == "CLAP2016":
    paper_entry = {"Dataset": "CLAP2016", "Method": "LRA-GNN (Paper)", "MAE": "3.11", "CS@5 (%)": "-", "Param.": "13M"}
else:
    raise ValueError(f"❌ Baseline non disponibile per dataset: {dataset_name}")

q_params  = count_parameters(agent.q_network)
cls_params = count_parameters(classifier)
ours_entry = {
    "Dataset": dataset_name,
    "Method": "LRA-GNN (Ours)",
    "MAE": round(best_mae, 2),
    "CS@5 (%)": round(compute_cs(y_true_best, y_pred_best), 2),
    "ε-error": round(compute_epsilon_error(y_true_best, y_pred_best, sigma=SIGMA), 4),
    "Param.": f"{(q_params+cls_params)/1e6:.1f}M"
}
df_comp = pd.DataFrame([paper_entry, ours_entry])
csv_comp = os.path.join(OUTDIR, f"{dataset_name.lower()}_comparison_table.csv")
df_comp.to_csv(csv_comp, index=False)
print(f"📄 Tabella comparativa salvata: {csv_comp}")

# PNG della tabella (niente emoji, per evitare warning font)
fig, ax = plt.subplots(figsize=(8, 1.6)); ax.axis('off')
tbl = ax.table(cellText=df_comp.values, colLabels=df_comp.columns, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.5)
plt.title(f"Performance Comparison on {dataset_name}", pad=12)
plt.tight_layout()
png_comp = os.path.join(OUTDIR, f"{dataset_name.lower()}_results_table.png")
plt.savefig(png_comp, dpi=300); plt.close()
print(f"🖼️ PNG tabella salvata: {png_comp}")

# Report & Confusion matrix
true_labels = res_best["true_labels"]
predicted_labels = res_best["predicted_labels"]
acc_dec = accuracy_score(true_labels, predicted_labels) * 100.0
all_labels = sorted(set(true_labels) | set(predicted_labels))
target_names = [f"{i*10}s" for i in all_labels]

rep_text = classification_report(true_labels, predicted_labels, labels=all_labels,
                                 target_names=target_names, zero_division=0)
print("\n📋 Classification report (best RL):\n", rep_text)

report_dict = classification_report(true_labels, predicted_labels, labels=all_labels,
                                    target_names=target_names, output_dict=True, zero_division=0)
report_table = pd.DataFrame(report_dict).transpose()
fig, ax = plt.subplots(figsize=(12, 5)); ax.axis('off')
tbl2 = ax.table(cellText=report_table.round(2).values,
                colLabels=report_table.columns, rowLabels=report_table.index,
                loc='center', cellLoc='center')
tbl2.auto_set_font_size(False); tbl2.set_fontsize(10); tbl2.scale(1.2, 1.2)
plt.title("Classification Report - RL Best", pad=20)
plt.tight_layout()
png_rep = os.path.join(OUTDIR, "classification_report_table.png")
plt.savefig(png_rep, dpi=300); plt.close()
print(f"🖼️ Report PNG salvato: {png_rep}")

cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix - {dataset_name}")
plt.tight_layout()
png_cm = os.path.join(OUTDIR, "confusion_matrix.png")
plt.savefig(png_cm, dpi=300); plt.close()
print(f"🧊 Confusion matrix PNG salvata: {png_cm}")

# Smoothed accuracy
correctness = [int(t == p) * 100 for t, p in zip(true_labels, predicted_labels)]
window = 20
moving_avg = np.convolve(correctness, np.ones(window)/window, mode='valid')
plt.figure(figsize=(10, 5))
plt.plot(moving_avg, label=f"Moving Average (window={window})")
plt.axhline(y=acc_dec, linestyle='--', label=f"Overall Accuracy = {acc_dec:.2f}%")
plt.ylabel("Accuracy (%)"); plt.title("Smoothed Accuracy over Validation Samples")
plt.xlabel("Sample Index"); plt.legend(); plt.grid(True)
plt.tight_layout()
png_sm = os.path.join(OUTDIR, "smoothed_accuracy.png")
plt.savefig(png_sm, dpi=300); plt.close()
print(f"📈 Smoothed accuracy PNG salvata: {png_sm}")

# True vs Pred (CSV)
df_compare = pd.DataFrame({"True Age": y_true_best, "Predicted Age": y_pred_best})
csv_tvsp = os.path.join(OUTDIR, f"{dataset_name.lower()}_true_vs_predicted.csv")
df_compare.to_csv(csv_tvsp, index=False)
print(f"📑 True-vs-Pred CSV salvato: {csv_tvsp}")

print("\n✅ Completato.")