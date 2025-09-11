import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import  argparse, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict

# tuoi dataset
from dataset.embedding_dataset_ablation_prlae import EmbeddingDataset            # DFE
from dataset.embedding_dataset_ablation_prlae_x_train_classifier import EmbeddingDatasetPRLAEXClassifier  # LRC

def as_vec(sample, device):
    """
    Restituisce un vettore [512] da un sample dei tuoi dataset:
    - DFE: dict {"embedding": [N,512] o [512]} -> media sui N se serve
    - LRC: tuple (X, age) dove X può essere [N,512] -> media
    """
    if isinstance(sample, dict):
        x = sample["embedding"]
        if x.dim() == 2:        # [N,512]
            x = x.mean(dim=0)
        elif x.dim() == 1:      # [512]
            pass
        else:
            raise ValueError(f"Embedding DFE non atteso: {x.shape}")
        return x.detach().cpu().float().numpy()
    elif isinstance(sample, (tuple, list)):
        x = sample[0]
        if x.dim() == 2:        # [N,512]
            x = x.mean(dim=0)
        elif x.dim() == 1:      # [512]
            pass
        else:
            raise ValueError(f"Embedding LRC non atteso: {x.shape}")
        return x.detach().cpu().float().numpy()
    else:
        raise ValueError(f"Tipo sample non gestito: {type(sample)}")

def get_age(sample):
    if isinstance(sample, dict):
        return float(sample.get("age", sample.get("label")))
    else:
        return float(sample[1])

def load_split(dataset_name, mode, device):
    """
    mode ∈ {'lrc','dfe'}
    Restituisce:
        X  -> np.array [N,512]
        y  -> np.array [N]
        ids -> lista indici (0..N-1)
    """
    ds_dir = f"embeddings_ablation_{dataset_name.lower()}_{'lrc_no_dfe' if mode=='lrc' else 'no_lrc_dfe'}"
    split_dir_train = os.path.join(ds_dir, "train")
    split_dir_val   = os.path.join(ds_dir, "val")

    if mode == "lrc":
        DS = EmbeddingDatasetPRLAEXClassifier
        ds_train = DS(split_dir_train, dataset_name, encoder=None, device=device)
        ds_val   = DS(split_dir_val,   dataset_name, encoder=None, device=device)
    else:
        DS = EmbeddingDataset
        ds_train = DS(split_dir_train, dataset_name, enable_lrc=False, enable_dfe=True, return_dict=True)
        ds_val   = DS(split_dir_val,   dataset_name, enable_lrc=False, enable_dfe=True, return_dict=True)

    def to_arrays(ds):
        X, y = [], []
        for i in range(len(ds)):
            s = ds[i]
            X.append(as_vec(s, device))
            y.append(get_age(s))
        return np.stack(X, axis=0), np.array(y, dtype=float)

    Xtr, ytr = to_arrays(ds_train)
    Xva, yva = to_arrays(ds_val)
    return Xtr, ytr, Xva, yva

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="UTKFACE", choices=["UTKFACE","FGNET","MORPH","CLAP2016"])
    ap.add_argument("--mode", default="lrc", choices=["lrc","dfe"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    Xtr, ytr, Xva, yva = load_split(args.dataset, args.mode, device)

    print(f"Train: X={Xtr.shape}, Val: X={Xva.shape}")

    # 1) Statistiche grezze
    def stats(X, name):
        m = X.mean(axis=0).mean()
        s = X.std(axis=0).mean()
        nans = np.isnan(X).sum()
        infs = np.isinf(X).sum()
        l2 = np.linalg.norm(X, axis=1).mean()
        print(f"[{name}] mean(mean)={m:.4f}  mean(std)={s:.4f}  "
              f"NaN={nans}  Inf={infs}  mean||x||2={l2:.2f}")

    stats(Xtr, "TRAIN")
    stats(Xva, "VAL")

    # 2) Drift fra train e val (per feature, media/std)
    mu_tr, sd_tr = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    mu_va, sd_va = Xva.mean(axis=0), Xva.std(axis=0) + 1e-9
    z_mean_drift = np.abs((mu_va - mu_tr) / sd_tr).mean()
    z_std_ratio  = (sd_va / sd_tr).mean()
    print(f"[DRIFT] |(μ_val-μ_tr)/σ_tr| medio = {z_mean_drift:.3f}   (σ_val/σ_tr) medio = {z_std_ratio:.3f}")

    # 3) Duplicati (approssimati): quanti vettori identici fino a 1e-6
    Xva_r = np.round(Xva, 6)
    uniq = np.unique(Xva_r, axis=0).shape[0]
    dup_ratio = 1 - uniq / Xva.shape[0]
    print(f"[VAL] duplicati ~ {dup_ratio*100:.2f}%")

    # 4) Centroidi per decade e nearest–centroid acc (grezza, senza classifier)
    def decade(a): return int(a) // 10
    tr_idx_by_dec = defaultdict(list)
    for i, a in enumerate(ytr): tr_idx_by_dec[decade(a)].append(i)

    centroids = {}
    for d, idxs in tr_idx_by_dec.items():
        centroids[d] = Xtr[idxs].mean(axis=0)

    def nearest_centroid(x):
        best_d, best_cos = None, -1.0
        for d, c in centroids.items():
            cs = float(np.dot(x, c) / (np.linalg.norm(x)*np.linalg.norm(c)+1e-9))
            if cs > best_cos:
                best_cos, best_d = cs, d
        return best_d, best_cos

    correct = 0
    cos_list = []
    for i, x in enumerate(Xva):
        d_hat, cs = nearest_centroid(x)
        d_true = decade(yva[i])
        cos_list.append(cs)
        correct += int(d_hat == d_true)
    print(f"[VAL] nearest-centroid decade acc = {correct/len(Xva)*100:.2f}%   "
          f"mean cosine to chosen centroid = {np.mean(cos_list):.3f}")

    # 5) “inizio vs fine” set: accuratezza grezza
    K = min(80, len(Xva)//2)
    beg_acc = sum(int(nearest_centroid(Xva[i])[0] == decade(yva[i])) for i in range(K)) / K * 100
    end_acc = sum(int(nearest_centroid(Xva[-i-1])[0] == decade(yva[-i-1])) for i in range(K)) / K * 100
    print(f"[SLICES] first {K} acc = {beg_acc:.2f}%   last {K} acc = {end_acc:.2f}%")

    # 6) PCA giusto per farti un’idea (stampa varianza spiegata)
    pca = PCA(n_components=2, random_state=0)
    pca.fit(Xtr)
    Xva_p = pca.transform(Xva)
    print(f"[PCA] varianza spiegata (2D) = {pca.explained_variance_ratio_.sum():.3f}")

if __name__ == "__main__":
    main()