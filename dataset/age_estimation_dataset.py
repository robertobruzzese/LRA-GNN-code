# dataset/age_estimation_dataset.py
import os
import re
import torch
from typing import Optional, Dict
from torch_geometric.data import Data, Dataset  # usa il Dataset di PyG

class AgeEstimationDataset(Dataset):
    """
    Dataset FULL (no ablation).
    Legge le cartelle di embedding (root_dir = .../train oppure .../val).
    Per ogni immagine:
      - Preferisce x = deep_features_from_rw.pt (se presente) altrimenti deep_features.pt
      - Se nessuno dei due esiste, usa x presente in graph_rw.pt (fallback)
      - y = età dedotta dal nome cartella (MORPH/FGNET/UTKFACE) o da CSV (CLAP2016)

    Richiede sempre graph_rw.pt per topologia (edge_index/edge_attr).
    """

    def __init__(self, root_dir: str, dataset_name: str = "MORPH",
                 embedding_split: str = "train", transform=None):
        super().__init__()
        self.dataset_name = dataset_name.upper()
        self.root_dir = root_dir  # ✅ usa sempre il path passato
        self.transform = transform
        self.samples = []  # list[dict(image_name, age)]

        # Parser età / mapping da CSV (solo CLAP2016 in full-pipeline)
        self._image_to_age: Optional[Dict[str, float]] = None
        if self.dataset_name == "CLAP2016":
            import pandas as pd
            csv_path = os.path.join("datasets", "data", "CLAP2016",
                                    f"CLAP_complete_{embedding_split}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"❌ CSV non trovato: {csv_path}")
            df = pd.read_csv(csv_path)
            # la colonna 'image' è senza .jpg? Normalizziamo togliendo l'estensione se presente
            img_col = df["image"].astype(str).str.replace(".jpg", "", regex=False)
            self._image_to_age = dict(zip(img_col, df["mean"]))

        # Regex per gli altri dataset
        morph_re = re.compile(r'[MF](\d{1,3})')
        fgnet_re = re.compile(r'[Aa](\d{1,3})')

        for fname in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, fname)
            if not os.path.isdir(folder_path):
                continue

            try:
                # --- estrazione età ---
                if self.dataset_name == "MORPH":
                    m = morph_re.search(fname)
                    if not m:
                        # log soft e salta
                        print(f"⚠️ Età non trovata in {fname}")
                        continue
                    age = float(int(m.group(1)))

                elif self.dataset_name == "FGNET":
                    m = fgnet_re.search(fname)
                    if not m:
                        print(f"⚠️ Età non trovata in {fname}")
                        continue
                    age = float(int(m.group(1)))

                elif self.dataset_name == "UTKFACE":
                    # tipico: '<age>_<gender>_<race>_<date>...'
                    try:
                        age = float(int(fname.split('_')[0]))
                    except Exception:
                        print(f"⚠️ Formato non valido per UTKFACE: {fname}")
                        continue

                elif self.dataset_name == "CLAP2016":
                    key = fname  # la cartella è già il basename dell'immagine senza .jpg
                    if self._image_to_age is None or key not in self._image_to_age:
                        print(f"⚠️ {fname}: età non trovata nel CSV")
                        continue
                    age = float(self._image_to_age[key])

                else:
                    raise ValueError(f"Dataset sconosciuto: {self.dataset_name}")

                # --- verifica esistenza graph_rw.pt ---
                graph_path = os.path.join(folder_path, "graph_rw.pt")
                if not os.path.exists(graph_path):
                    # niente topologia → skip
                    continue

                # possiamo anche scartare grafi con edge_index vuoto
                try:
                    gtmp = torch.load(graph_path, map_location="cpu")
                    ei = getattr(gtmp, "edge_index", None)
                    if not (torch.is_tensor(ei) and ei.numel() > 0):
                        print(f"⚠️ Ignoro grafo vuoto: {fname}")
                        os.makedirs("logs", exist_ok=True)
                        with open("logs/empty_graphs.log", "a") as f:
                            f.write(f"{fname}\n")
                        continue
                except Exception as e:
                    print(f"❌ Errore caricamento grafo {fname}: {e}")
                    continue

                self.samples.append({"image_name": fname, "age": age})

            except Exception as e:
                print(f"⚠️ Errore parsing {fname}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_name = sample["image_name"]
        age = sample["age"]

        folder = os.path.join(self.root_dir, image_name)

        # --- carica topologia ---
        graph_rw_path = os.path.join(folder, "graph_rw.pt")
        g_rw: Data = torch.load(graph_rw_path, map_location="cpu")

        # --- scegli le feature per-nodo ---
        feat_path = None
        for fn in ("deep_features_from_rw.pt", "deep_features.pt"):
            p = os.path.join(folder, fn)
            if os.path.exists(p):
                feat_path = p
                break

        if feat_path is not None:
            x = torch.load(feat_path, map_location="cpu")
            # 👇👇👇 NEW: bypass campione con feature vuote
            if (not torch.is_tensor(x)) or (x.ndim == 0) or (x.numel() == 0) or (x.size(0) == 0):
                print(f"⚠️ Skip zero-feature sample: {image_name}")
                return self.__getitem__((idx + 1) % len(self))
            # 👆👆👆
            if not (isinstance(x, torch.Tensor) and x.ndim == 2):
                raise ValueError(f"{image_name}: deep_features non Tensor 2D (trovato {type(x)} shape={getattr(x,'shape',None)})")
            x = x.detach().to(torch.float32); x.requires_grad_(False)

            # sanity N vs edge_index
            # sanity N vs edge_index (skip diretto se incoerente)
            ei = getattr(g_rw, "edge_index", None)
            eattr = getattr(g_rw, "edge_attr", None)

            if not (torch.is_tensor(ei) and ei.numel() > 0):
                print(f"⚠️ {image_name}: edge_index assente o vuoto → skip")
                return self.__getitem__((idx + 1) % len(self))

            N = x.shape[0]
            # Tieni solo archi con entrambe le estremità in [0, N-1]
            mask = (ei[0] >= 0) & (ei[0] < N) & (ei[1] >= 0) & (ei[1] < N)
            invalid = int((~mask).sum().item())
            if invalid > 0:
                print(f"⚠️ {image_name}: filtrati {invalid} archi out-of-range (max_idx={int(ei.max().item())}, N={N})")
                ei = ei[:, mask]
                if eattr is not None and eattr.size(0) == mask.size(0):
                    eattr = eattr[mask]

            if ei.numel() == 0:
                print(f"⚠️ {image_name}: tutti gli archi invalidi → skip")
                return self.__getitem__((idx + 1) % len(self))

            data = Data(
                x=x,
                edge_index=ei,
                edge_attr=eattr,
                y=torch.tensor([age], dtype=torch.float32)
            )
        else:
            # fallback: usa graph_rw così com’è (assumendo che .x sia consistente)
            data = g_rw
            data.y = torch.tensor([age], dtype=torch.float32)

        # utile per save_embeddings*
        data.image_name = image_name
        return data