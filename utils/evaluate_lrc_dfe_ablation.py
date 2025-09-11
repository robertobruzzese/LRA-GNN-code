#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 🔍 Script: evaluate_lrc_dfe_ablation.py
import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # <<-- usa insert(0), non append
# 🔍 Script: evaluate_lrc_dfe_ablation.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from typing import Optional

# ✅ Nuovi cloni LRC+DFE
from models.lra_gnn_lrc_dfe import LRA_GNN_LRC_DFE
from training.train_model_lrc_dfe import evaluate_model_lrc_dfe as evaluate_model
from dataset.age_estimation_dataset_lrc_dfe import AgeEstimationDatasetLrcDfe

def _select_device(prefer: Optional[str] = None):
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

def main(dataset_name: str, device_arg: Optional[str] = None):
    device = _select_device(device_arg)
    print(f"✅ Device eval: {device}")
    dataset = dataset_name.upper()
    exp_name = "lrc_dfe"

    checkpoint_path = f"checkpoints_ablation/{dataset.lower()}/{exp_name}/best_lra_gnn_{dataset.lower()}.pth"
    embeddings_dir = f"embeddings_ablation_{dataset.lower()}_{exp_name}"
    val_dir = os.path.join(embeddings_dir, "val")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint non trovato: {checkpoint_path}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"❌ Directory embedding mancante: {val_dir}")

    # ✅ Dataset LRC+DFE → lista di grafi (8 LRC + 1 DFE)
    val_dataset = AgeEstimationDatasetLrcDfe(
        root_dir=val_dir,
        dataset_name=dataset,
        split="val",
        strict=False,
        prefer_from_rw=True,
        set_target_on_each_graph=True,
        clap2016_csv=os.path.join(os.path.dirname(val_dir), "val", "metadata.csv")
    )
    # ✅ collate_fn per evitare liste annidate
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            collate_fn=lambda b: b[0], num_workers=0)

    print("✅ USO hidden_channels=512 nel modello")

    # 🧠 Modello list-aware
    model = LRA_GNN_LRC_DFE(
        num_layers=12,
        num_heads=8,
        in_channels=512,
        hidden_channels=512,
        out_channels=1,
        enable_lrc=True,
        enable_dfe=True
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 📊 Valutazione
    print(f"\n📊 Valutazione LRC + DFE su {dataset} (device={device})...\n")
    criterion = nn.MSELoss()
    mae, cs5, eps = evaluate_model(model, val_loader, device, criterion)

    print(f"📌 Risultati su {dataset}:")
    print(f"✅ MAE     = {mae:.2f}")
    print(f"✅ CS@5    = {cs5:.2f}%")
    print(f"✅ ε-error = {eps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["MORPH", "FGNET", "UTKFACE", "CLAP2016"])
    parser.add_argument("--device", type=str, choices=["cpu","cuda","mps"], default=None)
    args = parser.parse_args()

    main(args.dataset, device_arg=args.device)