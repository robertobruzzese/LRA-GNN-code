# 🔍 Script: evaluate_dfe_ablation.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
# 📦 Importa i dataset specifici
from dataset.age_estimation_dataset_morph import AgeEstimationDatasetMorph
from dataset.age_estimation_dataset_fgnet import AgeEstimationDatasetFGNET
from dataset.age_estimation_dataset_utkface import AgeEstimationDatasetUTKFace
from dataset.age_estimation_dataset_clap2016 import AgeEstimationDatasetClap2016
from models.lra_gnn import LRA_GNN
from training.train_model import evaluate_model
# --- PARACADUTE: self-loops se edge_index è vuoto/malformato ---
import torch
from torch_geometric.utils import add_self_loops
from models.lra_gnn import LRA_GNN

_original_forward = LRA_GNN.forward

def _forward_with_safe_edges(self, data, *args, **kwargs):
    ei = getattr(data, 'edge_index', None)
    # condizioni: None, tensore vuoto, dimensioni non (2, E)
    need_fix = (
        ei is None
        or not torch.is_tensor(ei)
        or ei.numel() == 0
        or ei.dim() != 2
        or ei.size(0) != 2
    )
    if need_fix:
        num_nodes = data.x.size(0)
        device = data.x.device
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
        ei, _ = add_self_loops(ei, num_nodes=num_nodes)  # crea solo self-loops
        data.edge_index = ei
    return _original_forward(self, data, *args, **kwargs)

LRA_GNN.forward = _forward_with_safe_edges
# --- fine paracadute ---
def main(dataset_name):
    dataset = dataset_name.upper()
    exp_name = "no_lrc_dfe"

    checkpoint_path = f"checkpoints_ablation/{dataset.lower()}/{exp_name}/best_lra_gnn_{dataset.lower()}.pth"
    embeddings_dir = f"embeddings_ablation_{dataset.lower()}_{exp_name}"
    train_dir = os.path.join(embeddings_dir, "train")
    val_dir   = os.path.join(embeddings_dir, "val")

    if dataset_name == "UTKFACE":
        val_dataset = AgeEstimationDatasetUTKFace(val_dir, dataset_name, "val", enable_lrc=False, enable_dfe=True)
    elif dataset_name == "CLAP2016":
            val_dataset = AgeEstimationDatasetClap2016(val_dir, dataset_name, "val", enable_lrc=False, enable_dfe=True)
    elif dataset_name == "FGNET":
            val_dataset = AgeEstimationDatasetFGNET(val_dir, dataset_name, "val", enable_lrc=False, enable_dfe=True)
    elif dataset_name == "MORPH":
            val_dataset = AgeEstimationDatasetMorph(val_dir, dataset_name, "val", enable_lrc=False, enable_dfe=True)
    else:
            raise ValueError(f"❌ Dataset {dataset_name} non supportato")

    print(f"🔍 Val samples: {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    print("✅ USO hidden_channels=512 nel modello")
    # 🧠 Modello
    model = LRA_GNN(
        num_layers=12,
        num_heads=8,
        in_channels=512,
        hidden_channels=512,
        out_channels=1,
        enable_lrc=False,    # ✅ esplicitato
        enable_dfe=True      # ✅ esplicitato
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
    print(f"\n📊 Valutazione DFE only (senza LRC né PRLAE) su {dataset}...\n")
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