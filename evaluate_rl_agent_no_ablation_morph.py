
#evaluate_rl_agent_no_ablation_morph.py
#ha leakage
import torch
from torch.utils.data import DataLoader
from dataset.embedding_dataset import EmbeddingDataset
from models.progressive_rl import ProgressiveRLAgent

import torch.nn as nn
from collections import defaultdict
from models.classifier import AgeGroupClassifier
import os
import sys
import argparse
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import glob

import numpy as np
import random
import time
from torch.utils.data import Subset



# 🔧 Parsing argomento --dataset
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="MORPH", help="Dataset: MORPH, FGNET o UTKFACE o CLAP2016")
parser.add_argument("--quick-n", type=int, default=200, help="Campioni usati per la valutazione rapida")
parser.add_argument("--top-k", type=int, default=3, help="Quanti checkpoint migliori rivalutare sul val completo (usa -1 per TUTTI)")
parser.add_argument("--quick-mode", type=str, default="uniform",
                    choices=["uniform", "random", "head", "stratified"],
                    help="Criterio di scelta del sottoinsieme rapido")
parser.add_argument("--quick-seed", type=int, default=42, help="Seed per la selezione random/stratified")
parser.add_argument("--eval-only", action="store_true",
                    help="Stampa MAE/accuracy per modello e termina, senza grafici né report")

args = parser.parse_args()
EVAL_ONLY = bool(args.eval_only)
if EVAL_ONLY:
    import matplotlib
    matplotlib.use("Agg")
    plt.ioff()

dataset_name = args.dataset.upper()

# --- pausa bloccante finché non premi INVIO, poi esci ---
def pause_and_exit():
    try:
        input("\n⏸  Premi INVIO per uscire...")
    except (EOFError, KeyboardInterrupt):
        # se lo stdin non è interattivo, non possiamo aspettare → fallback
        pass
    try:
        import sys
        sys.exit(0)
    except SystemExit:
        import os
        os._exit(0)

def pause(msg="\n⏸  Premi INVIO per continuare..."):
    import sys, time
    sys.stdout.flush()
    try:
        # se c'è una TTY, aspetta invio
        if sys.stdin.isatty():
            input(msg)
        else:
            # in ambiente non interattivo fai una breve pausa visibile
            time.sleep(3)
    except (EOFError, KeyboardInterrupt):
        time.sleep(3)

def show_if_needed():
    if not EVAL_ONLY:
        plt.show()
    else:
        plt.close('all')
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

def last_linear(m):
    # prende l'ultimo Linear del Q-network
    return [x for x in m.modules() if isinstance(x, nn.Linear)][-1]
# ⚙️ Hyperparametri coerenti col training

#state_dim = 134  # esempio: embedding (128) + delta_x + delta_y + pos
action_dim = 5   # su, giù, sinistra, destra, resta

# 🔁 Carica il dataset
dataset = EmbeddingDataset(embedding_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
sample = next(iter(dataloader))
embedding_dim = sample[0].shape[1]
extra_features = 6
state_dim = embedding_dim + extra_features
print(f"🔎 Dimensione embedding rilevata: {embedding_dim} → state_dim: {state_dim}")
# 🔍 Debug dimensione embedding
print(f"🔎 Dimensione embedding rilevata: {sample[0].shape[1]}")
# ---- Costruzione quick_loader in base a --quick-mode ----


total_n = len(dataset)
quick_n = min(args.quick_n, total_n)

# seeding per riproducibilità
np.random.seed(args.quick_seed)
random.seed(args.quick_seed)
torch.manual_seed(args.quick_seed)

def _infer_age_from_item(item):
    # item può essere (emb, meta) oppure (emb, age)
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
    # prova a stratificare per decade dell'età; fallback a random se età non disponibile
    decade_bins = defaultdict(list)
    for i in range(total_n):
        try:
            age = _infer_age_from_item(dataset[i])
        except Exception:
            age = None
        if age is None:
            continue
        decade_bins[int(age // 10)].append(i)

    if not decade_bins:  # nessuna età disponibile → random
        quick_idx = rng.choice(total_n, size=quick_n, replace=False).tolist()
    else:
        per_bin = max(1, quick_n // len(decade_bins))
        quick_idx = []
        for _, idxs in sorted(decade_bins.items()):
            k = min(per_bin, len(idxs))
            if k > 0:
                pick = rng.choice(len(idxs), size=k, replace=False)
                quick_idx.extend([idxs[j] for j in pick])
        # riempi se mancano elementi
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
def midpoints_for(dataset, C):
    # Restituisce i midpoints delle decadi in base al dataset e al #classi C
    # Esempi:
    #  - FGNET/UTKFACE: di solito classi 0..6 → 5,15,25,35,45,55,65
    #  - MORPH: se il classifier ha 9 classi (0..80s) start=0; altrimenti 10..70 → start=10
    if dataset == "MORPH":
        start_decade = 0 if C >= 9 else 10
    else:
        start_decade = 0
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
    # Estrae y_true e y_pred continui con gli stessi fallback usati per il MAE
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
# 🎯 Carica il modello LRA-GNN (serve per feature estratte se non già incluse)
#model = LRA_GNN()  # se serve per il forward
#model.to(device)
#model.eval()
# 🔢 Conta il numero di parametri del modello LRA-GNN
# ✅ Parametri: Q-Network + Classifier


# 🔁 Carica l’agente RL
agent = ProgressiveRLAgent(state_dim=state_dim, action_dim=action_dim)
# 📥 Carica il modello salvato
classifier_path = os.path.join(checkpoint_dir, "classifier.pth")
classifier = AgeGroupClassifier(input_dim=embedding_dim).to(device)
classifier.load_state_dict(torch.load(classifier_path, map_location=device))
classifier.eval()
agent.classifier = classifier
# dopo agent.classifier = classifier   ⬅️ QUI

agent.q_network.to(device)
agent.q_network.eval()

import re
from datetime import datetime

def _ts_from_name(path):
    # Prova a estrarre il timestamp dal nome file: best_agent_YYYY-MM-DD_HH-MM-SS.pth
    m = re.search(r"best_agent_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.pth$", os.path.basename(path))
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            pass
    # Fallback all’mtime del file
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return datetime.min

all_ckpts = [p for p in glob.glob(os.path.join(checkpoint_dir, "best_agent_*.pth")) if os.path.isfile(p)]
all_ckpts_sorted = sorted(all_ckpts, key=_ts_from_name)  # dal più vecchio al più nuovo
RECENT_K = 15
agent_files = all_ckpts_sorted[-RECENT_K:] if len(all_ckpts_sorted) > RECENT_K else all_ckpts_sorted

print(f"🔎 Trovati {len(all_ckpts)} checkpoint in {checkpoint_dir} — uso solo gli ultimi {len(agent_files)}:")
print("   ", [os.path.basename(p) for p in agent_files])

if not agent_files:
    raise FileNotFoundError(f"Nessun best_agent_*.pth in {checkpoint_dir}")
# ---- Fase 1: valutazione rapida su subset (quick_loader) ----
scores = []  # (mae_q, acc_dec_q, ckpt_path)
for p in agent_files:
    state_raw = torch.load(p, map_location=device)

    # estrai lo state_dict vero e proprio
    if isinstance(state_raw, dict) and ("state_dict" in state_raw or "q_network_state_dict" in state_raw):
        state = state_raw.get("q_network_state_dict", state_raw.get("state_dict"))
    else:
        state = state_raw

    # rimuovi eventuale prefisso "q_network."
    clean_state = { (k[len("q_network."):] if k.startswith("q_network.") else k): v
                    for k, v in state.items() }

    # (opzionale) skip se dimensioni non compatibili
    if "fc1.weight" in clean_state and clean_state["fc1.weight"].shape[1] != agent.q_network.fc1.in_features:
        print(f"⏭️ Skip {os.path.basename(p)}: input dim mismatch "
              f"({clean_state['fc1.weight'].shape[1]} vs {agent.q_network.fc1.in_features})")
        continue

    agent.q_network.load_state_dict(clean_state, strict=False)
    agent.q_network.to(device); agent.q_network.eval()

    # info checkpoint corrente
    print(f"🔎 (quick) Valuto ckpt: {os.path.basename(p)}")
    pause("⏸  (quick) Premi INVIO per valutare il prossimo checkpoint...")
    try:
        last = last_linear(agent.q_network)  # o agent.q_network.fc_out
        if last.bias is not None:
            print("bias_mean:", float(last.bias.detach().mean()))
    except IndexError:
        pass

    # quick eval sul sottoinsieme
    res_q = agent.evaluate(model=None, dataloader=quick_loader, device=device)

    acc_dec_q = float(res_q.get("decade_accuracy", 0))
    if acc_dec_q < 1:
        acc_dec_q *= 100

    mae_q = mae_from_results(res_q, dataset_name)

    print(f"   → (quick) {os.path.basename(p)}: acc = {acc_dec_q:.2f}% | MAE = {mae_q:.2f}")
    scores.append((mae_q, acc_dec_q, p))

if not scores:
    raise RuntimeError("Nessun checkpoint valido dopo la shortlist.")

# Ordina per MAE crescente (migliore), tie-breaker: accuracy decrescente
# Ordina per MAE crescente (migliore), tie-breaker: accuracy decrescente
scores.sort(key=lambda x: (x[0], -x[1]))

# Se --top-k = -1, valuta TUTTI i checkpoint nella full-eval
# Ordina per MAE crescente (migliore), tie-breaker: accuracy decrescente
scores.sort(key=lambda x: (x[0], -x[1]))

# ⚙️ full-eval SOLO sui migliori 3 del quick
FULL_K = 3
raw_candidates = scores[:min(FULL_K, len(scores))]

# Filtra eventuali file mancanti per robustezza
candidates = [p for (_, _, p) in raw_candidates if os.path.isfile(p)]
missing = [p for (_, _, p) in raw_candidates if not os.path.isfile(p)]
if missing:
    print("⚠️ Skip perché mancanti:", [os.path.basename(x) for x in missing])

print("🏁 Candidati FULL (migliori dal quick sui 10 più recenti):", [os.path.basename(x) for x in candidates])

# ---- Fase 2: full eval sul validation completo ----
# ---- Fase 2: full eval sul validation completo ----
best_path, best_mae, best_acc = None, float("inf"), -1.0
best_cs5, best_eps = -1.0, float("inf")
for p in candidates:
    state_raw = torch.load(p, map_location=device)
    if isinstance(state_raw, dict) and ("state_dict" in state_raw or "q_network_state_dict" in state_raw):
        state = state_raw.get("q_network_state_dict", state_raw.get("state_dict"))
    else:
        state = state_raw

    clean_state = { (k[len("q_network."):] if k.startswith("q_network.") else k): v
                    for k, v in state.items() }

    agent.q_network.load_state_dict(clean_state, strict=False)
    agent.q_network.to(device); agent.q_network.eval()

    print(f"🔎 (full) Valuto ckpt: {os.path.basename(p)}")
    try:
        last = last_linear(agent.q_network)
        if last.bias is not None:
            print("bias_mean:", float(last.bias.detach().mean()))
    except IndexError:
        pass

    # full eval sul dataloader completo
    res = agent.evaluate(model=None, dataloader=dataloader, device=device)

    acc_dec = float(res.get("decade_accuracy", 0))
    if acc_dec < 1:
        acc_dec *= 100

    mae = mae_from_results(res, dataset_name)  # deve essere definita sopra
    y_true_full, y_pred_full = extract_y_true_pred(res, dataset_name)
    cs5 = compute_cs(y_true_full, y_pred_full, k=5)
    eps = compute_epsilon_error(y_true_full, y_pred_full, sigma=5)

    print(f"   → (full) {os.path.basename(p)}: acc = {acc_dec:.2f}% | MAE = {mae:.2f}")

    # criterio: minimizza MAE; a parità di MAE massimizza accuracy
    if (mae < best_mae) or (mae == best_mae and acc_dec > best_acc):
        best_mae, best_acc, best_path = mae, acc_dec, p
        best_cs5, best_eps = cs5, eps
#########
print(
    f"✅ Miglior checkpoint (per MAE): {os.path.basename(best_path)} — "
    f"MAE = {best_mae:.2f} | CS@5 = {best_cs5:.2f}% | ε-error = {best_eps:.4f} | "
    f"decade acc = {best_acc:.2f}%"
)

# --- Persisti la scelta del best (criterio MAE) ---
import json, shutil
if best_path is None:
    raise RuntimeError("Nessun checkpoint selezionato come best.")

meta = {"best_path": best_path, "mae": float(best_mae), "decade_acc_pct": float(best_acc)}
with open(os.path.join(checkpoint_dir, "BEST_SELECTED.json"), "w") as f:
    json.dump(meta, f, indent=2)
if EVAL_ONLY:
    pause("\n⏸  Modalità eval-only: premi INVIO per uscire...")
    raise SystemExit(0)
dst = os.path.join(checkpoint_dir, "best_agent_SELECTED.pth")
try:
    shutil.copy2(best_path, dst)
    print(f"💾 Copiato il best in: {dst}")
except Exception as e:
    print(f"⚠️ Copia best fallita: {e}")

# 👉 SOLO MAE/accuracy: blocca finché non premi INVIO, poi esci
if EVAL_ONLY:
    print("🛑 Modalità --eval-only attiva: stop dopo stampa MAE/accuracy. Nessun grafico creato.")
    pause_and_exit()

# === Da qui in giù SOLO se NON --eval-only ===
# ricarica definitivamente il migliore (se vuoi proseguire con report/grafici)
best_raw = torch.load(best_path, map_location=device)
if isinstance(best_raw, dict) and ("state_dict" in best_raw or "q_network_state_dict" in best_raw):
    best_state = best_raw.get("q_network_state_dict", best_raw.get("state_dict"))
else:
    best_state = best_raw
best_clean = { (k[len("q_network."):] if k.startswith("q_network.") else k): v for k, v in best_state.items() }
agent.q_network.load_state_dict(best_clean, strict=False)
agent.q_network.to(device)
agent.q_network.eval()
##########
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

q_params = count_parameters(agent.q_network)
cls_params = count_parameters(classifier)
num_params_m = (q_params + cls_params) / 1e6
print(f"📊 Parametri totali: Q-Network={q_params:,} + Classifier={cls_params:,} → {num_params_m:.1f}M")

# ================== BASELINE CLASSIFIER (età "soft" da probabilità) ==================
from sklearn.metrics import accuracy_score

clf_true_labels, clf_pred_labels = [], []
clf_true_ages,  clf_pred_ages  = [], []

classifier.eval()
with torch.no_grad():
    for batch in dataloader:
        # Il tuo EmbeddingDataset può restituire (emb, age) o (emb, meta)
        if isinstance(batch, (list, tuple)):
            emb = batch[0].to(device)
            meta = batch[1] if len(batch) > 1 else None
        else:
            emb = batch.to(device)
            meta = None

        # ---- forward classificatore ----
        logits = classifier(emb)                 # [1, C]
        pred_lab = logits.argmax(dim=1).item()   # per accuracy di decade

        # ---- età continua "soft" da probabilità ----
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # [C]
        C = probs.shape[0]

        # offset delle decadi (robusto a #classi)
        midpoints = midpoints_for(dataset_name, C)  # ⬅️ usa la funzione
        pred_age_soft = float((probs * midpoints).sum())
        # ---- età reale ----
        true_age = None
        if isinstance(meta, dict) and "age" in meta:
            ta = meta["age"]
            true_age = float(ta.item() if torch.is_tensor(ta) else ta)
        elif torch.is_tensor(meta):
            true_age = float(meta.item()) if meta.numel() == 1 else float(meta[0].item())
        if true_age is None:
            continue

        true_lab = int(true_age // 10)

        # ---- accumula metriche baseline ----
        clf_true_labels.append(true_lab)
        clf_pred_labels.append(pred_lab)
        clf_true_ages.append(true_age)
        clf_pred_ages.append(pred_age_soft)

# Stampa metriche baseline (decade + MAE/CS@5 continui)
clf_acc = accuracy_score(clf_true_labels, clf_pred_labels) * 100
baseline_mae = np.mean(np.abs(np.array(clf_true_ages) - np.array(clf_pred_ages)))
baseline_cs5 = np.mean(np.abs(np.array(clf_true_ages) - np.array(clf_pred_ages)) <= 5) * 100
print(f"🔎 Baseline Classifier → Accuracy decade: {clf_acc:.2f}% | MAE: {baseline_mae:.2f} | CS@5: {baseline_cs5:.2f}%")
# ======================================================================
# 🔬 Ora puoi usarlo per valutare
results = agent.evaluate(model=None, dataloader=dataloader, device=device)
print(results)
print("📦 Chiavi contenute in results:", results.keys())
predicted_labels_rl = np.array(results.get("predicted_labels", []), dtype=int)
# ✨ Confronto aggregato tra baseline e RL
rl_decade_acc = float(results.get("decade_accuracy", 0))
if rl_decade_acc < 1:   # se è in [0,1], portalo in percentuale
    rl_decade_acc *= 100
print(f"🆚 Confronto (decade): Baseline = {clf_acc:.2f}%  |  RL = {rl_decade_acc:.2f}%")

# Estrai età vere e predette continue
# Estrai età vere e rimappa le predizioni RL con gli stessi midpoints
# Estrai età vere
y_true = np.array(results["true_ages"], dtype=float)

# 1) usa direttamente le età continue dell’RL se presenti
y_pred = np.array(results.get("predicted_ages", []), dtype=float)

# 2) fallback: ricostruisci da row/col (se il tuo env le salva)
if y_pred.size == 0 and "pred_rows" in results and "pred_cols" in results:
    y_pred = np.array(results["pred_rows"], dtype=int) * 10 + np.array(results["pred_cols"], dtype=int)


# 3) fallback finale: usa i midpoint delle decadi dalle label
if y_pred.size == 0:
    if predicted_labels_rl.size:
        C_rl = int(predicted_labels_rl.max()) + 1
        mid_rl = midpoints_for(dataset_name, C_rl)
        y_pred = mid_rl[predicted_labels_rl]
    else:
        raise RuntimeError("Né predicted_ages né predicted_labels presenti nei results: non posso calcolare MAE/CS.")
# --- Confronto per campione (CSV) ---    
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
csv_cmp = "output/baseline_vs_rl_per_sample.csv"
df_cmp.to_csv(csv_cmp, index=False)
print(f"✅ Salvato {csv_cmp}")

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
show_if_needed()


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

show_if_needed()

# 🔲 Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix - {dataset_name}")
plt.tight_layout()

# 💾 Salvataggio della Confusion Matrix
os.makedirs("output", exist_ok=True)
plt.savefig("output/confusion_matrix.png", dpi=300)

show_if_needed()

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
show_if_needed()

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