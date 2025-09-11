# utils/generate_val_embeddings.py

import torch
import os
import argparse
from models.lra_gnn import LRA_GNN
from save_embeddings_morph import save_embeddings  # cambia se non è MORPH

from dataset.age_estimation_dataset_morph import AgeEstimationDatasetMorph
from torch_geometric.loader import DataLoader

def main(dataset_name, checkpoint_path, embeddings_dir, batch_size=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔧 In base al dataset
    dataset = AgeEstimationDatasetMorph(
        root_dir=os.path.join(embeddings_dir, "val"),
        enable_lrc=True,       # o False a seconda dell’ablation
        enable_dfe=False       # o True a seconda dell’ablation
    )
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ⚙️ Modello identico a quello usato in training
    model = LRA_GNN(
        num_layers=12,
        num_heads=8,
        in_channels=512,
        hidden_channels=512,
        out_channels=1,
        enable_lrc=True,
        enable_dfe=False
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 🧠 Salvataggio embedding su val/
    save_embeddings(
        model, val_loader, device,
        save_dir=os.path.join(embeddings_dir, "val")
    )

    print("✅ Embedding val/ generati senza leakage.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MORPH")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--embeddings_dir", type=str, required=True)

    args = parser.parse_args()

    main(args.dataset, args.checkpoint, args.embeddings_dir)