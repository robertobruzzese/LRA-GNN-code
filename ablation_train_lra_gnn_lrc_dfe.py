#!/usr/bin/env python3
# ablation_train_lra_gnn_lrc_dfe.py

import os
import argparse
from train_lra_gnn_lrc_dfe import train_lra_gnn_lrc_dfe

def main():
    parser = argparse.ArgumentParser(description="Training ablation LRC+DFE (no PRLAE)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["MORPH", "FGNET", "CLAP2016", "UTKFACE"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    args = parser.parse_args()

    dataset = args.dataset.upper()
    embeddings_dir  = f"embeddings_ablation_{dataset.lower()}_lrc_dfe"
    checkpoints_dir = f"checkpoints_ablation/{dataset.lower()}/lrc_dfe"

    os.makedirs(os.path.join(embeddings_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(embeddings_dir, "val"),   exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Avvia il training dedicato LRC+DFE (usa DataLoader interni già corretti)
    train_lra_gnn_lrc_dfe(
        dataset_name=dataset,
        embeddings_dir=embeddings_dir,
        checkpoints_dir=checkpoints_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        # Se preferisci creare tu i DataLoader e passarli, puoi usare:
        # train_loader=..., val_loader=...
    )

if __name__ == "__main__":
    main()