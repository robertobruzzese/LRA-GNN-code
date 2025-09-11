import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from models.lra_gnn import LRA_GNN
from training.train_model import evaluate_model

# 📦 Importa i dataset specifici
from dataset.age_estimation_dataset_morph import AgeEstimationDatasetMorph
from dataset.age_estimation_dataset_fgnet import AgeEstimationDatasetFGNET
from dataset.age_estimation_dataset_utkface import AgeEstimationDatasetUTKFace
from dataset.age_estimation_dataset_clap2016 import AgeEstimationDatasetClap2016

def main(dataset_name):
    dataset = dataset_name.upper()
    exp_name = "no_lrc_no_dfe"

    checkpoint_path = f"checkpoints_ablation/{dataset.lower()}/{exp_name}/best_lra_gnn_{dataset.lower()}.pth"
    embeddings_dir = f"embeddings_ablation_{dataset.lower()}_{exp_name}"
    train_dir = os.path.join(embeddings_dir, "train")
    val_dir = os.path.join(embeddings_dir, "val")


    if dataset_name == "UTKFACE":
        val_dataset = AgeEstimationDatasetUTKFace(val_dir, dataset_name, "val", enable_lrc=False, enable_dfe=False)
    elif dataset_name == "CLAP2016":
            val_dataset = AgeEstimationDatasetClap2016(val_dir, dataset_name, "val", enable_lrc=False, enable_dfe=False)
    elif dataset_name == "FGNET":
            val_dataset = AgeEstimationDatasetFGNET(val_dir, dataset_name, "val", enable_lrc=False, enable_dfe=False)
    elif dataset_name == "MORPH":
            val_dataset = AgeEstimationDatasetMorph(val_dir, dataset_name, "val", enable_lrc=False, enable_dfe=False)
    else:
            raise ValueError(f"❌ Dataset {dataset_name} non supportato")

    print(f"🔍 Val samples: {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔧 Inizializza il modello LRA-GNN (GCN only)
    model = LRA_GNN(
        num_layers=12,
        num_heads=8,
        in_channels=128,
        hidden_channels=128,
        out_channels=1,
        enable_lrc=False,
        enable_dfe=False
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print(f"\n📊 Valutazione GCN only su {dataset}...\n")

    criterion = nn.MSELoss()
    mae, cs5, eps = evaluate_model(model, val_loader, device, criterion)

    print(f"📌 Risultati su {dataset}:")
    print(f"✅ MAE     = {mae:.2f}")
    print(f"✅ CS@5    = {cs5:.2f}%")
    print(f"✅ ε-error = {eps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Nome del dataset: MORPH, FGNET, UTKFACE, CLAP2016")
    args = parser.parse_args()

    main(args.dataset)