# utils/ablation_train_classifier_fgnet.py
import argparse, os, sys, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns, matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric
from torch.utils.data import get_worker_info

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.embedding_dataset_ablation_prlae import EmbeddingDataset
from dataset.embedding_dataset_ablation_prlae_x_train_classifier import EmbeddingDatasetPRLAEXClassifier
import random

# --- SEED E DETERMINISMO ---
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)
# ---------- utils ----------
def decade_tensor_from_ages(t):
    if t.dim() > 1:
        t = t.view(-1)
    return torch.tensor([int(float(a.item())) // 10 for a in t], dtype=torch.long, device=t.device)

def print_class_dist(y, name):
    cls = [int(float(a.item())) // 10 for a in y]
    cnt = Counter(cls)
    print(f"\n📊 Distribuzione classi in {name}:")
    for c in sorted(cnt):
        print(f"  {c*10:02d}-{c*10+9:02d}: {cnt[c]}")
    return np.array(cls)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.g = gamma
        self.a = alpha
        self.red = reduction
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.a)
        pt = torch.exp(-ce)
        fl = ((1-pt)**self.g) * ce
        return fl.mean() if self.red == 'mean' else fl

# ---------- modelli ----------
def make_mlp_fgnet(input_dim=512):
    # più semplice per evitare collapse su dataset piccoli
    return nn.Sequential(
        nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(64, 10)
    )

class DFEGraphClassifier(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, num_classes=10):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin   = nn.Linear(hidden_dim, num_classes)
    def forward(self, data):
        x = data.x
        ei = data.edge_index.long()
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        else:
            batch = batch.long()
        x = F.relu(self.conv1(x, ei))
        x = self.conv2(x, ei)
        g = global_mean_pool(x, batch)
        return self.lin(g)

# ---------- estrazione (per LRC) ----------
def extract_Xy_from_loader(loader, device):
    X_rows, y_rows = [], []
    for batch in loader:
        sample = batch[0]
        if isinstance(sample, dict):
            emb = sample["embedding"].to(device)
            age = sample["age"].to(device)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            elif emb.dim() > 2:
                print(f"❌ Embedding dim >2: {emb.shape} (skip)")
                continue
        elif isinstance(sample, torch_geometric.data.Data):
            if not hasattr(sample, "x"):
                print("❌ Data senza x, skip")
                continue
            emb = sample.x.to(device)
            if emb.dim() != 2 or emb.size(1) != 512:
                print(f"❌ x shape anomala: {emb.shape}, skip")
                continue
            emb = emb.mean(0, keepdim=True)  # [1,512]
            age = sample.y.to(device)
            if age.dim() == 0:
                age = age.unsqueeze(0)
        elif isinstance(sample, (list, tuple)):
            emb, age = sample
            emb = emb.to(device); age = age.to(device)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            if emb.dim() != 2 or emb.size(1) != 512:
                print(f"❌ tuple emb shape anomala: {emb.shape}, skip")
                continue
        else:
            print(f"⚠️ Sample ignorato: {type(sample)}")
            continue
        X_rows.append(emb)
        y_rows.append(age.view(-1))
    if not X_rows:
        raise ValueError("❌ Nessun embedding valido estratto.")
    return torch.cat(X_rows, 0), torch.cat(y_rows, 0)

# ======================= MAIN =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["FGNET"])
    parser.add_argument("--exp_name", type=str, required=True, choices=["lrc", "dfe"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    dataset_name = "FGNET"
    device = torch.device(args.device) if args.device else \
             (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    if args.exp_name == "lrc":
        # --- dataset LRC: embeddings mediati [N,512] ---
        train_dir = "embeddings_ablation_fgnet_lrc_no_dfe/train"
        val_dir   = "embeddings_ablation_fgnet_lrc_no_dfe/val"
        train_ds = EmbeddingDatasetPRLAEXClassifier(train_dir, dataset_name, encoder=None, device=device)
        val_ds   = EmbeddingDatasetPRLAEXClassifier(val_dir,   dataset_name, encoder=None, device=device)

        raw_train = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=lambda x: x)
        raw_val   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=lambda x: x)

        X_train, y_train = extract_Xy_from_loader(raw_train, device)
        X_val,   y_val   = extract_Xy_from_loader(raw_val,   device)

        print("DEBUG shapes pre-scale:", X_train.shape, X_val.shape)
        # (non serve guard, sono già [N,512])
        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train.cpu().numpy()), dtype=torch.float, device=device)
        X_val   = torch.tensor(scaler.transform(X_val.cpu().numpy()),      dtype=torch.float, device=device)

        # classi e pesi + debug distribuzioni
       # --- distribuzione (debug) ---
        ytr_dec_raw = print_class_dist(y_train, "train/")
        yva_dec_raw = print_class_dist(y_val,   "val/")

        # --- rimappa alle sole classi presenti: 0..K-1 ---
        classes = np.unique(ytr_dec_raw)                 # es. array([0,1,2,3,4,5,6])
        cls2idx = {c: i for i, c in enumerate(classes)}

        def remap(arr):
            return np.array([cls2idx[c] for c in arr], dtype=np.int64)

        ytr_dec = remap(ytr_dec_raw)                     # shape [Ntrain]
        yva_dec = remap(yva_dec_raw)                     # shape [Nval]
        num_classes = len(classes)                        # K (per FGNET: 7)

        # --- tensori target mappati ---
        ytr = torch.tensor(ytr_dec, dtype=torch.long, device=device)
        yva = torch.tensor(yva_dec, dtype=torch.long, device=device)
        # --- logit adjustment SOLO in validazione (calcolato una volta sui target di train) ---
        
        # --- pesi classi su K ---
        cw = compute_class_weight(class_weight='balanced',
                                classes=np.arange(num_classes),
                                y=ytr_dec).astype(np.float32)
        cw[cw > 2.0] *= 1.5
        cw_t = torch.tensor(cw, dtype=torch.float, device=device)

        # --- sampler bilanciato ---
        sample_w = cw_t[ytr].detach().cpu().numpy()
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        train_loader = DataLoader(
            TensorDataset(X_train, ytr),
            batch_size=32,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, yva),
            batch_size=32,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g,
        )
       

        # --- MLP con out_dim = K ---
        model = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        ).to(device)

        # 🔁 FocalLoss + pesi (niente label smoothing)
        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, alpha=None):
                super().__init__()
                self.gamma = gamma
                self.alpha = alpha
            def forward(self, logits, targets):
                ce = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
                pt = torch.exp(-ce)
                return ((1-pt)**self.gamma * ce).mean()

        crit = FocalLoss(gamma=2.0, alpha=cw_t)
        opt  = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
        sch  = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

        # --- logit adjustment SOLO in validazione ---
        # --- logit adjustment SOLO in validazione (safe fp32 per MPS) ---
        # --- logit adjustment SOLO in validazione (safe fp32 per MPS) ---
        priors = np.bincount(ytr_dec, minlength=num_classes).astype(np.float32)
        priors /= max(len(ytr_dec), 1)
        log_prior = torch.tensor(priors, dtype=torch.float32, device=device).clamp_min(1e-12).log()
        tau = 0.7  # prova 0.5–1.0

        best_acc, best_state = 0.0, None
        patience, wait = 50, 0
        warmup = 25
        with torch.no_grad():
            model[-1].bias.copy_(-log_prior)  # usa log_prior già sul device
        for ep in range(1, 501):
            # train (NO adjustment qui)
            model.train(); tot_loss = 0.0; corr = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss   = crit(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                tot_loss += loss.item()
                corr += (logits.argmax(1) == yb).sum().item()
            tr_acc = 100 * corr / len(train_loader.dataset)

            # val (con adjustment)
            model.eval(); vc = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb) - tau * log_prior.unsqueeze(0)
                    vc += (out.argmax(1) == yb).sum().item()
            va_acc = 100 * vc / len(val_loader.dataset)

            # early stopping
            improved = va_acc > best_acc + 1e-4
            if improved:
                best_acc = va_acc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if ep >= warmup and wait >= patience:
                    print(f"🛑 Early stopping (FGNET LRC) ep={ep}, wait={wait}, best={best_acc:.2f}%")
                    break

            if ep == 1 or ep % 25 == 0:
                print(f"[DBG] ep={ep}  train={tr_acc:.2f}%  val={va_acc:.2f}%  best={best_acc:.2f}%  wait={wait}/{patience}")
            sch.step()

        if best_state:
            model.load_state_dict(best_state)
        ckpt = os.path.join("checkpoints_ablation", "fgnet", "prlae_lrc_no_dfe")
        os.makedirs(ckpt, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt, "classifier.pth"))
        print(f"✅ Salvato MLP in {ckpt}/classifier.pth")

        # --- valutazione finale / confusion matrix su K classi ---

         
# --- dopo early stopping ---
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb) - tau * log_prior.unsqueeze(0)   # stesso adjustment della validazione
                pr = out.argmax(1).cpu().numpy()
                all_p.extend(pr)
                all_t.extend(yb.cpu().numpy())

        cm = confusion_matrix(all_t, all_p, labels=range(num_classes))

        plt.figure(figsize=(9, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title("FGNET LRC — Confusion Matrix"); plt.xlabel("Pred"); plt.ylabel("True"); plt.show()

        macro_f1 = f1_score(all_t, all_p, average='macro') * 100
        acc = accuracy_score(all_t, all_p) * 100
        print(f"🎯 Acc: {acc:.2f}%  |  Macro-F1: {macro_f1:.2f}%")
        print(classification_report(all_t, all_p, digits=2)) 

    else:  # DFE
        train_dir = "embeddings_ablation_fgnet_no_lrc_dfe/train"
        val_dir   = "embeddings_ablation_fgnet_no_lrc_dfe/val"
        train_ds  = EmbeddingDataset(train_dir, "FGNET", return_dict=False, enable_lrc=False, enable_dfe=True)
        val_ds    = EmbeddingDataset(val_dir,   "FGNET", return_dict=False, enable_lrc=False, enable_dfe=True)

        train_loader = GeoDataLoader(
            train_ds, batch_size=32, shuffle=True,
            worker_init_fn=seed_worker, generator=g
        )
        val_loader = GeoDataLoader(
            val_ds, batch_size=32, shuffle=False,
            worker_init_fn=seed_worker, generator=g
        )

        # class weights
        all_dec = []
        for b in GeoDataLoader(train_ds, batch_size=128, shuffle=False):
            all_dec.append(decade_tensor_from_ages(b.y))
        ytr_np = torch.cat(all_dec, 0).cpu().numpy()
        # --- logit adjustment ---
        eps = 1e-12
        #priors = np.bincount(ytr_np, minlength=10) / max(len(ytr_np), 1)
       # log_prior = torch.log(torch.tensor(priors + eps, device=device))  # shape [10]
       # --- logit adjustment SOLO in validazione ---
        # --- logit adjustment (DFE): usa ytr_np e fissa K=10 ---
        
        num_classes = 10
        priors = np.bincount(ytr_np, minlength=num_classes).astype(np.float32)
        priors /= max(len(ytr_np), 1)
        log_prior = torch.tensor(priors, dtype=torch.float32, device=device).clamp_min(1e-12).log()

        tau = 0.0   # <-- per ora zero, così verifichiamo che la pipeline “regge”

        present = np.unique(ytr_np)
        cw = compute_class_weight('balanced', classes=present, y=ytr_np)
        cw_full = np.zeros(10, dtype=np.float32); cw_full[present] = cw; cw_full[cw_full > 2.0] *= 1.5
        cw_t = torch.tensor(cw_full, dtype=torch.float, device=device)

        model = DFEGraphClassifier(in_dim=512, hidden_dim=256, num_classes=10).to(device)
        with torch.no_grad():
            model.lin.bias.copy_(-log_prior)   # stessa idea del ramo LRC
        # crit = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=0.1)
        crit = nn.CrossEntropyLoss(weight=cw_t)   # oppure: FocalLoss(alpha=cw_t)
        opt  = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        sch   = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
        
        
        best_acc, best_state = 0.0, None
        best_score, wait, patience = 0.0, 0, 50
        for ep in range(1, 501):
            model.train(); tot, corr, n = 0.0, 0, 0
            for batch in train_loader:
                batch = batch.to(device)
                yb = decade_tensor_from_ages(batch.y)

                # forward: niente logit adjustment in TRAIN
                logits = model(batch)

                loss = crit(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()

                tot += loss.item()
                corr += (logits.argmax(1) == yb).sum().item()
                n += yb.size(0)

            tr_acc = 100 * corr / max(n, 1)

            model.eval(); 
            # --- VALIDATION (DURANTE TRAINING) ---
            preds, targs = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    yb = decade_tensor_from_ages(batch.y)

                    logits = model(batch)           # con tau=0.0 nessun adjustment
                    # logits = logits - tau * log_prior.unsqueeze(0)  # se vuoi riattivarlo

                    preds.extend(logits.argmax(1).cpu().numpy())
                    targs.extend(yb.cpu().numpy())

            va_acc = (np.array(preds) == np.array(targs)).mean() * 100
            val_macro_f1 = f1_score(targs, preds, average="macro") * 100

            if val_macro_f1 > best_score + 1e-4:
                best_score = val_macro_f1
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if va_acc > best_acc:
                best_acc, best_state = va_acc, {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if ep == 1 or ep % 50 == 0:
                print(f"Epoch {ep}/500  Loss {tot:.4f}  Train {tr_acc:.2f}%  Val {va_acc:.2f}%")
            sch.step()

        if best_state:
            model.load_state_dict(best_state)
        ckpt = os.path.join("checkpoints_ablation", "fgnet", "prlae_no_lrc_dfe")
        os.makedirs(ckpt, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt, "classifier.pth"))
        print(f"✅ Salvato GCN (DFE) in {ckpt}/classifier.pth")

        # valutazione
        model.eval(); 
       
# --- EVAL FINALE / CM ---
        all_p, all_t = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                logits = model(batch)                         # niente adjustment con tau=0.0
                # oppure: logits = logits - tau * log_prior.unsqueeze(0)

                pr = logits.argmax(1).cpu()
                tg = decade_tensor_from_ages(batch.y).cpu()
                all_p.extend(pr.numpy()); all_t.extend(tg.numpy())
        cm = confusion_matrix(all_t, all_p, labels=range(num_classes))  # num_classes = 10
        plt.figure(figsize=(9, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title("FGNET DFE — Confusion Matrix"); plt.xlabel("Pred"); plt.ylabel("True"); plt.show()
        print(f"🎯 Acc: {accuracy_score(all_t, all_p)*100:.2f}%  |  Macro-F1: {f1_score(all_t, all_p, average='macro')*100:.2f}%")
        print(classification_report(all_t, all_p, digits=2))