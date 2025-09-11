import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from models.lra_gnn import LRA_GNN
from dataset.age_estimation_dataset_utkface import AgeEstimationDatasetUTKFace
from dataset.age_estimation_dataset_fgnet import AgeEstimationDatasetFGNET
from dataset.age_estimation_dataset_clap2016 import AgeEstimationDatasetClap2016
from dataset.age_estimation_dataset_morph import AgeEstimationDatasetMorph
from training.train_model import evaluate_model  # ✅ usa la funzione completa

def main(dataset_name):
    dataset_name = dataset_name.upper()
    exp_name = "lrc_no_dfe"
    embedding_dir = f"embeddings_ablation_{dataset_name.lower()}_{exp_name}"
    checkpoint_dir = os.path.join("checkpoints_ablation", dataset_name.lower(), exp_name)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_lra_gnn_{dataset_name.lower()}.pth")

    print(f"📦 Caricamento embedding da: {embedding_dir}")
    val_dir = os.path.join(embedding_dir, "val")

    if dataset_name == "UTKFACE":
        val_dataset = AgeEstimationDatasetUTKFace(val_dir, dataset_name, "val", enable_lrc=True, enable_dfe=False)
    elif dataset_name == "CLAP2016":
        val_dataset = AgeEstimationDatasetClap2016(val_dir, dataset_name, "val", enable_lrc=True, enable_dfe=False)
    elif dataset_name == "FGNET":
        val_dataset = AgeEstimationDatasetFGNET(val_dir, dataset_name, "val", enable_lrc=True, enable_dfe=False)
    elif dataset_name == "MORPH":
        val_dataset = AgeEstimationDatasetMorph(val_dir, dataset_name, "val", enable_lrc=True, enable_dfe=False)
    else:
        raise ValueError(f"❌ Dataset {dataset_name} non supportato")

    print(f"🔍 Val samples: {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LRA_GNN(
        num_layers=12,
        num_heads=8,
        in_channels=512,
        hidden_channels=512,
        out_channels=1,
        enable_lrc=True,
        enable_dfe=False
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ Modello caricato da: {checkpoint_path}")
    model.eval()
    from torch_geometric.data import Data
    import torch.nn as nn

    class _CompatWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x, edge_index, edge_attr=None):
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            out = self.base(data)
            return out[0] if isinstance(out, tuple) else out

    model = _CompatWrapper(model).to(device)
    # 📊 Valutazione
    print(f"\n📊 Valutazione LRC only (senza DFE né PRLAE) su {dataset_name}...\n")
    criterion = nn.MSELoss()
    mae, cs5, eps = evaluate_model(model, val_loader, device, criterion)

    print(f"📌 Risultati su {dataset_name}:")
    print(f"✅ MAE     = {mae:.2f}")
    print(f"✅ CS@5    = {cs5:.2f}%")
    print(f"✅ ε-error = {eps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Nome del dataset: MORPH, FGNET, UTKFACE, CLAP2016")
    args = parser.parse_args()

    main(args.dataset)