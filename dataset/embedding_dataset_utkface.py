# dataset/embedding_dataset.py
import os, re, torch
from torch.utils.data import Dataset

class UtkfaceEmbeddingDataset(Dataset):
    def __init__(self, embeddings_dir, dataset_name="MORPH", return_dict=False, clap2016_csv=None):
        self.embeddings_dir = embeddings_dir
        self.dataset_name = dataset_name.upper()
        self.return_dict = return_dict  # ignorato: torniamo sempre (embedding, age) per RL
        self.files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith(".pt")])

        # Per CLAP2016: opzionale CSV per mappa immagine->età
        self.age_map = None
        if self.dataset_name == "CLAP2016" and clap2016_csv and os.path.exists(clap2016_csv):
            import pandas as pd
            df = pd.read_csv(clap2016_csv)
            self.age_map = {str(r["image"]).replace(".jpg",""): float(r["mean"]) for _, r in df.iterrows()}

    def __len__(self): return len(self.files)

    def _parse_age_from_name(self, stem: str):
        # UTKFACE: "age_gender_race_timestamp"
        # es. "10_0_2_20170110224230094" -> age = 10
        try:
            tok = str(stem).split('_', 1)[0]
            if tok.isdigit():
                return float(int(tok))
        except Exception:
            pass
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
            if data.ndim == 2 and data.shape[0] > 1:  # safety: fai mean se [N,512]
                data = data.mean(dim=0)
            if data.ndim != 1:
                raise ValueError(f"{fname}: atteso Tensor 1D, trovato {tuple(data.shape)}")

            emb = data
            stem = os.path.splitext(fname)[0]

            # prova sidecar .age.txt
            age = None
            sidecar = os.path.join(self.embeddings_dir, f"{stem}.age.txt")
            if os.path.exists(sidecar):
                try:
                    with open(sidecar) as f:
                        age = float(f.read().strip())
                except Exception:
                    age = None

            # prova parser esistente
            if age is None:
                try:
                    age = self._parse_age_from_name(stem)
                except Exception:
                    age = None

            # ✅ Fallback UTKFACE: "age_gender_race_timestamp"
            if age is None and getattr(self, "dataset_name", "").upper() == "UTKFACE":
                tok = stem.split('_', 1)[0]
                if tok.isdigit():
                    age = float(int(tok))

            if age is None:
                raise ValueError(f"Impossibile dedurre l'età per {fname} (no sidecar/parse).")

        else:
            raise TypeError(f"{fname}: tipo non supportato {type(data)}")

        # output sempre come tupla per compatibilità RL
        return emb.to(torch.float32), torch.tensor(float(age), dtype=torch.float32)