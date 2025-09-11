# dataset/age_estimation_dataset_lrc_dfe.py
import os
import re
import glob
import torch
from typing import List, Optional, Dict
from torch.utils.data import Dataset
from torch_geometric.data import Data

def _freeze_graph(g: Data) -> Data:
    # Disattiva il grad su tensori “di dataset”
    for k in ("x", "edge_attr", "y"):
        v = getattr(g, k, None)
        if torch.is_tensor(v):
            v.requires_grad_(False)
    # (opzionale ma utile) edge_index in long
    ei = getattr(g, "edge_index", None)
    if torch.is_tensor(ei):
        g.edge_index = ei.long()
    return g

class AgeEstimationDatasetLrcDfe(Dataset):
    """
    Dataset per l'ablation **LRC+DFE** (senza PRLAE).

    Per ogni immagine ritorna una LISTA di grafi:
      [ graph_lrc_0.pt, ..., graph_lrc_7.pt,  graph_rw + deep_features ]
    dove il grafo DFE ha x = deep_features per-nodo (allineate a graph_rw).

    Struttura attesa:
      embeddings_ablation_<dataset>_lrc_dfe/
        ├─ train/
        │   └─ <ID>/
        │       ├─ graph_lrc_0.pt ... graph_lrc_7.pt
        │       ├─ graph_rw.pt
        │       └─ deep_features.pt   (oppure deep_features_from_rw.pt)
        └─ val/
            └─ <ID>/ ...

    Args:
        root_dir: path allo split (es. "..._lrc_dfe/val")
        dataset_name: "MORPH" | "FGNET" | "UTKFACE" | "CLAP2016"
        split: solo informativo ("train"|"val")
        strict: se True, errore su sample incompleti; se False, li salta
        prefer_from_rw: se True, usa deep_features_from_rw.pt se presente
        set_target_on_each_graph: se True, imposta y su ogni grafo
        clap2016_csv: path al metadata.csv (CLAP2016). Se None, prova a cercarlo nella dir padre di root_dir
    """

    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        split: str = "val",
        strict: bool = True,
        prefer_from_rw: bool = True,
        set_target_on_each_graph: bool = True,
        clap2016_csv: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.dataset_name = dataset_name.upper()
        self.split = split
        self.strict = strict
        self.prefer_from_rw = prefer_from_rw
        self.set_target_on_each_graph = set_target_on_each_graph

        # Dizionario età per CLAP2016
        self.age_dict: Optional[Dict[str, float]] = None
        if self.dataset_name == "CLAP2016":
            if clap2016_csv is None:
                parent = os.path.abspath(os.path.join(self.root_dir, os.pardir))
                clap2016_csv = os.path.join(parent, "metadata.csv")
            if not os.path.exists(clap2016_csv):
                raise FileNotFoundError(f"CSV mancante per CLAP2016: {clap2016_csv}")
            import pandas as pd
            df = pd.read_csv(clap2016_csv)
            self.age_dict = {str(row["image"]).split(".")[0]: float(row["mean"]) for _, row in df.iterrows()}

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Split directory non trovata: {self.root_dir}")

        # Scansione sample
        self.samples: List[str] = []
        for entry in sorted(os.listdir(self.root_dir)):
            d = os.path.join(self.root_dir, entry)
            if not os.path.isdir(d):
                continue

            lrc_list = sorted(glob.glob(os.path.join(d, "graph_lrc_*.pt")))
            has_rw = os.path.exists(os.path.join(d, "graph_rw.pt"))

            deep_from_rw = os.path.join(d, "deep_features_from_rw.pt")
            deep_std = os.path.join(d, "deep_features.pt")
            has_deep = os.path.exists(deep_from_rw) or os.path.exists(deep_std)

            if len(lrc_list) >= 8 and has_rw and has_deep:
                self.samples.append(entry)
            elif self.strict:
                raise FileNotFoundError(
                    f"Sample incompleto {entry}: LRC={len(lrc_list)}/8, "
                    f"RW={int(has_rw)}/1, deep={int(has_deep)}/1"
                )
            # se non strict, semplicemente non aggiunge il sample

    def __len__(self) -> int:
        return len(self.samples)

    # ---- parsing età per dataset ----
    def _parse_age(self, sample_id: str) -> float:
        if self.dataset_name == "MORPH":
            m = re.search(r"[FfMm](\d{1,3})", sample_id)
            if not m:
                raise ValueError(f"Età non trovata in MORPH id: {sample_id}")
            return float(int(m.group(1)))

        if self.dataset_name == "FGNET":
            m = re.search(r"[Aa](\d{1,3})", sample_id)
            if not m:
                raise ValueError(f"Età non trovata in FGNET id: {sample_id}")
            return float(int(m.group(1)))

        if self.dataset_name == "UTKFACE":
            # tipicamente "age_gender_race_date..." → prende age
            try:
                return float(int(sample_id.split("_")[0]))
            except Exception:
                raise ValueError(f"Formato UTKFACE inatteso per id: {sample_id}")

        if self.dataset_name == "CLAP2016":
            if self.age_dict is None:
                raise RuntimeError("age_dict non caricato per CLAP2016")
            if sample_id not in self.age_dict:
                raise ValueError(f"Età non trovata in CSV per: {sample_id}")
            return float(self.age_dict[sample_id])

        raise ValueError(f"Dataset non supportato: {self.dataset_name}")

    

    def __getitem__(self, idx: int):
        sample_id = self.samples[idx]
        sample_dir = os.path.join(self.root_dir, sample_id)
        age_val = torch.tensor(self._parse_age(sample_id), dtype=torch.float32)

        # 1) Carica gli 8 grafi LRC (se ce ne sono di più, prende i primi 8 ordinati)
        lrc_paths = sorted(glob.glob(os.path.join(sample_dir, "graph_lrc_*.pt")))[:8]
        if len(lrc_paths) < 8 and self.strict:
            raise FileNotFoundError(f"{sample_id}: attesi 8 graph_lrc_*.pt, trovati {len(lrc_paths)}")

        graphs: List[Data] = []
        for p in lrc_paths:
            g = torch.load(p, map_location="cpu")   # 👍 meglio su CPU nel dataset
            g = _freeze_graph(g)                    # 👈 QUI
            if self.set_target_on_each_graph:
                g.y = age_val
                g = _freeze_graph(g)                # ricongela y appena impostata
            graphs.append(g)

        # 2) Carica DFE: deep_features per-nodo + graph_rw
        deep_features_path = (
            os.path.join(sample_dir, "deep_features_from_rw.pt")
            if (self.prefer_from_rw and os.path.exists(os.path.join(sample_dir, "deep_features_from_rw.pt")))
            else os.path.join(sample_dir, "deep_features.pt")
        )
        if not os.path.exists(deep_features_path):
            raise FileNotFoundError(f"{sample_id}: deep_features mancante ({deep_features_path})")

        graph_rw_path = os.path.join(sample_dir, "graph_rw.pt")
        if not os.path.exists(graph_rw_path):
            raise FileNotFoundError(f"{sample_id}: graph_rw.pt mancante")

        # carica su CPU e congela
        x = torch.load(deep_features_path, map_location="cpu")  # atteso Tensor [N_rw, F]
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError(f"{sample_id}: deep_features non Tensor 2D (trovato {type(x)}, ndim={getattr(x,'ndim',None)})")
        x = x.detach()             # niente grafi autograd
        x.requires_grad_(False)    # congelato
        x = x.to(torch.float32)    # tipo consistente

        g_rw = torch.load(graph_rw_path, map_location="cpu")
        ei = getattr(g_rw, "edge_index", None)
        if torch.is_tensor(ei):
            ei = ei.long()

        # (opzionale ma utile) sanity check N vs edge_index
        if torch.is_tensor(ei) and ei.numel() > 0:
            N = x.shape[0]
            if int(ei.max()) >= N:
                raise ValueError(f"{sample_id}: edge_index max={int(ei.max())} >= N={N} (mismatch tra RW e deep_features)")

        dfe_graph = Data(
            x=x,
            edge_index=ei,
            edge_attr=getattr(g_rw, "edge_attr", None),
            y=age_val,
        )
        dfe_graph = _freeze_graph(dfe_graph)  # ← disattiva grad anche su y/edge_attr
        graphs.append(dfe_graph)

        return graphs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root_dir={self.root_dir}, dataset={self.dataset_name}, split={self.split}, n={len(self)})"