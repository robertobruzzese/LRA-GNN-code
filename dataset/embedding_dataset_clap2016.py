# dataset/embedding_dataset_clap2016.py
import os, re, torch
from torch.utils.data import Dataset

class Clap2016EmbeddingDataset(Dataset):
    def __init__(self, embeddings_dir, dataset_name="MORPH", return_dict=False, clap2016_csv=None):
        self.embeddings_dir = embeddings_dir
        self.dataset_name = dataset_name.upper()
        self.return_dict = return_dict  # ignorato: torniamo sempre (embedding, age) per RL
        self.files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith(".pt")])

        # Per CLAP2016: opzionale CSV per mappa immagine->età
        self.age_map = None
        if self.dataset_name == "CLAP2016":
            if clap2016_csv is None:
                split = "val" if "val" in embeddings_dir.lower() else "train"
                clap2016_csv = os.path.join("datasets","data","CLAP2016", f"CLAP_complete_{split}.csv")
            import pandas as pd
            df = pd.read_csv(clap2016_csv)
            self.age_map = {str(r["image"]).replace(".jpg",""): float(r["mean"]) for _, r in df.iterrows()}

    def __len__(self): return len(self.files)

    def _parse_age_from_name(self, stem: str):
        ds = self.dataset_name
        if ds == "MORPH":
            m = re.search(r"[FfMm](\d{1,3})", stem);  return float(int(m.group(1))) if m else None
        if ds == "FGNET":
            m = re.search(r"[Aa](\d{1,3})", stem);    return float(int(m.group(1))) if m else None
        if ds == "UTKFACE":
            try: return float(int(stem.split("_")[0]))
            except: return None
        if ds == "CLAP2016":
            if self.age_map: return self.age_map.get(stem)
        return None

    def __getitem__(self, idx):
        fname = self.files[idx]
        path  = os.path.join(self.embeddings_dir, fname)
        data  = torch.load(path, map_location="cpu")

        # 1) Se è un dict {'embedding':..., 'age'/ 'label': ...}
        if isinstance(data, dict):
            emb = data.get("embedding", None)
            age = data.get("age", data.get("label", None))
            if emb is None or age is None:
                raise ValueError(f"{fname}: dict senza 'embedding' o 'age/label'")
        # 2) Se è un tensore piatto 512D
        elif torch.is_tensor(data):
            if data.ndim == 2 and data.shape[0] > 1:  # safety
                data = data.mean(dim=0)
            if data.ndim != 1:
                raise ValueError(f"{fname}: atteso Tensor 1D, trovato {tuple(data.shape)}")
            emb = data
            stem = os.path.splitext(fname)[0]
            # prova sidecar .age.txt
            sidecar = os.path.join(self.embeddings_dir, f"{stem}.age.txt")
            if os.path.exists(sidecar):
                with open(sidecar) as f: age = float(f.read().strip())
            else:
                age = self._parse_age_from_name(stem)
            if age is None:
                raise ValueError(f"Impossibile dedurre l'età per {fname} (no sidecar/parse).")
        else:
            raise TypeError(f"{fname}: tipo non supportato {type(data)}")

        # output sempre come tupla per compatibilità RL
        return emb.to(torch.float32), torch.tensor(age, dtype=torch.float32)