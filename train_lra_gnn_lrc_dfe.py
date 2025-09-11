# -*- coding: utf-8 -*- train_lra_gnn_lrc_dfe.py

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Optional
#from torch_geometric.loader import DataLoader
from tqdm import tqdm  # ✅ NEW
from torch.utils.data import DataLoader

# Modello + valutazione (versioni LRC+DFE)
from models.lra_gnn_lrc_dfe import LRA_GNN_LRC_DFE
from training.train_model_lrc_dfe import evaluate_model_lrc_dfe as evaluate_model

# Dataset LRC+DFE (lista di grafi)
from dataset.age_estimation_dataset_lrc_dfe import AgeEstimationDatasetLrcDfe

def _select_device(prefer: str = None):
    import os, torch
    pref = (prefer or os.environ.get("LRA_DEVICE", "")).lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _to_device_graph(graph, device):
    """Sposta su device un Data o una list[Data]."""
    def _to_dev(g):
        for attr in ("x", "y", "edge_index", "edge_attr", "batch"):
            if hasattr(g, attr) and getattr(g, attr) is not None and torch.is_tensor(getattr(g, attr)):
                setattr(g, attr, getattr(g, attr).to(device))
        return g
    if isinstance(graph, list):
        return [_to_dev(g) for g in graph]
    return _to_dev(graph)

def train_lra_gnn_lrc_dfe(
    dataset_name: str,
    embeddings_dir: str,
    checkpoints_dir: str,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    save_embeddings: bool = True,  # compatibilità firma
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    device_str: Optional[str] = None,   # ✅ NEW: permetti override device
):
    """
    Training ablation LRC+DFE (no PRLAE). Il modello si aspetta list[Data] (8 LRC + 1 DFE).
    Salva il best model in: checkpoints_ablation/<dataset>/lrc_dfe/best_lra_gnn_<dataset>.pth
    """
    os.makedirs(checkpoints_dir, exist_ok=True)
    device = _select_device(device_str)  # ✅ usa l'override se passato

    # DataLoader di default (se non passati)
    if train_loader is None or val_loader is None:
        root = embeddings_dir  # es: embeddings_ablation_morph_lrc_dfe
        train_ds = AgeEstimationDatasetLrcDfe(
            os.path.join(root, "train"),
            dataset_name=dataset_name,
            split="train",
            strict=False,
            prefer_from_rw=True,
            set_target_on_each_graph=True,
            clap2016_csv=os.path.join(root, "train", "metadata.csv")
        )
        val_ds = AgeEstimationDatasetLrcDfe(
            os.path.join(root, "val"),
            dataset_name=dataset_name,
            split="val",
            strict=False,
            prefer_from_rw=True,
            set_target_on_each_graph=True,
            clap2016_csv=os.path.join(root, "val", "metadata.csv")

        )
        # ✅ aggiungi worker; pin_memory se CUDA
        #pin = (device.type == "cuda")
        #train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
        #                          num_workers=2, pin_memory=pin, collate_fn=lambda b: b[0])
        #val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
        #                          num_workers=2, pin_memory=pin, collate_fn=lambda b: b[0])
        pin = (device.type == "cuda")
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=pin,
            collate_fn=lambda b: b[0]   # <-- return the single item (list[Data]) untouched
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=pin,
            collate_fn=lambda b: b[0]
        )
    # Modello
    model = LRA_GNN_LRC_DFE(
        num_layers=12,
        num_heads=8,
        in_channels=512,
        hidden_channels=512,
        out_channels=1,
        enable_lrc=True,
        enable_dfe=True,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_mae = float("inf")
    best_path = os.path.join(checkpoints_dir, f"best_lra_gnn_{dataset_name.lower()}.pth")

    print(f"▶️  Training LRC+DFE su {dataset_name} | device={device} | epochs={num_epochs}", flush=True)
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0

        t0 = time.time()
        # ✅ progress bar per batch
        for graph in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
            graph = _to_device_graph(graph, device)

            # target: prendo il primo grafo della lista
            target = graph[0].y if isinstance(graph, list) else graph.y

            optimizer.zero_grad()
            out = model(graph)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out.view_as(target), target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

        train_loss = epoch_loss / max(n, 1)

        # Validazione
        val_mae, val_cs5, val_eps = evaluate_model(model, val_loader, device, criterion)

        dt = time.time() - t0
        print(f"[{epoch:03d}] train_loss={train_loss:.4f} | val_MAE={val_mae:.3f} | CS@5={val_cs5:.2f}% | eps={val_eps:.3f} | {dt:.1f}s",
              flush=True)

        # Salva best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_path)
            print(f"  💾 Best migliorato → {best_path} (MAE={best_val_mae:.3f})", flush=True)

    print(f"✅ Fine training. Best MAE={best_val_mae:.3f} | ckpt={best_path}", flush=True)