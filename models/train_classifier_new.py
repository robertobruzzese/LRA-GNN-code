#!/usr/bin/env python3
#train_classifier_new.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse, re, torch
import torch.nn as nn
import torch.optim as optim
from dataset.embedding_dataset import EmbeddingDataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from models.classifier import AgeGroupClassifier  # ✅ usa la classe “ufficiale”

# ---------- Dataset per EMBEDDING PIATTI (512D) ----------
class FlatEmbeddingDataset(Dataset):
    def __init__(self, root_dir: str, dataset_name: str, csv_path: str = None):
        self.root_dir = root_dir
        self.ds = dataset_name.upper()
        self.paths = sorted(
            [os.path.join(root_dir, f) for f in os.listdir(root_dir)
             if f.endswith(".pt") and os.path.isfile(os.path.join(root_dir, f))]
        )
        if self.ds == "CLAP2016":
            if not csv_path or not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Per CLAP2016 serve --csv che punti al metadata (es. CLAP_complete_train.csv): {csv_path}"
                )
            import pandas as pd
            df = pd.read_csv(csv_path)
            self.age_map = {str(r["image"]).replace(".jpg", ""): float(r["mean"]) for _, r in df.iterrows()}
        else:
            self.age_map = None

    def __len__(self): 
        return len(self.paths)

    def _parse_age_from_name(self, stem: str) -> float:
        if self.ds == "MORPH":
            m = re.search(r"[FfMm](\d{1,3})", stem);  return float(int(m.group(1))) if m else None
        if self.ds == "FGNET":
            m = re.search(r"[Aa](\d{1,3})", stem);    return float(int(m.group(1))) if m else None
        if self.ds == "UTKFACE":
            try: return float(int(stem.split("_")[0]))
            except: return None
        if self.ds == "CLAP2016":
            return float(self.age_map.get(stem))
        return None

    def __getitem__(self, idx):
        p = self.paths[idx]
        emb = torch.load(p, map_location="cpu")
        if not isinstance(emb, torch.Tensor) or emb.ndim != 1:
            raise ValueError(f"{os.path.basename(p)}: atteso Tensor 1D (512,), trovato {type(emb)} {getattr(emb,'shape',None)}")
        stem = os.path.splitext(os.path.basename(p))[0]
        age = self._parse_age_from_name(stem)
        if age is None:
            raise ValueError(f"Impossibile ricavare l'età da '{stem}' per dataset {self.ds}")
        return emb, torch.tensor(age, dtype=torch.float32)

# ---------- Train ----------
def train_classifier(embeddings, ages, input_dim, device='cpu', epochs=100, batch_size=64, lr=1e-3):
    # target a decadi
    targets = (ages.long() // 10).clamp(min=0, max=9)
    dl = DataLoader(TensorDataset(embeddings, targets), batch_size=batch_size, shuffle=True)

    # ✅ usa l’input_dim passato alla funzione
    model = AgeGroupClassifier(input_dim=input_dim).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs+1):
        tot_loss = corr = n = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += float(loss) * xb.size(0)
            corr += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)
        if ep % max(1, epochs//10) == 0 or ep == 1:
            print(f"📚 Epoch {ep}/{epochs} - loss={tot_loss/n:.4f} - acc={100*corr/n:.2f}%")
    return model

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["MORPH","FGNET","UTKFACE","CLAP2016"])
    ap.add_argument("--split", choices=["train","val"], default="train")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--csv", type=str, default=None, help="Solo CLAP2016: path CSV (CLAP_complete_<split>.csv)")
    args = ap.parse_args()

    base = {
        "MORPH":"embeddings_morph",
        "FGNET":"embeddings_FGNET",
        "UTKFACE":"embeddings_utkface",
        "CLAP2016":"embeddings_clap2016",
    }[args.dataset]
    split_dir = os.path.join(base, args.split)

    if args.dataset == "CLAP2016" and args.csv is None:
        args.csv = os.path.join("datasets","data","CLAP2016", f"CLAP_complete_{args.split}.csv")

    embedding_dir = split_dir
    dataset_name = args.dataset.upper()

    ds = EmbeddingDataset(
        embeddings_dir=embedding_dir,
        dataset_name=dataset_name,
        return_dict=False,
        clap2016_csv=(args.csv if dataset_name == "CLAP2016" else None),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    Xs, Ys = [], []
    for xb, yb in dl:
        Xs.append(xb)
        Ys.append(yb)
    X = torch.cat(Xs, 0)
    y = torch.cat(Ys, 0)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"✅ Caricati {len(ds)} embedding da {split_dir} | device={device} | dim={tuple(X.shape)}")

    model = train_classifier(X, y, input_dim=X.shape[1], device=device, epochs=args.epochs,
                             batch_size=args.batch_size, lr=args.lr)

    out_dir = os.path.join("checkpoints", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"classifier.pth")
    torch.save(model.state_dict(), out_path)
    print(f"💾 Salvato: {out_path}")

if __name__ == "__main__":
    main()