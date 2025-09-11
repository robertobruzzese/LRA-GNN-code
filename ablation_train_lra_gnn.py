# 🔧 Script: ablation_train_lra_gnn.py

# 🔧 Script: ablation_train_lra_gnn.py

import argparse
import os
from train_lra_gnn import train_lra_gnn
from models.lra_gnn import LRA_GNN
# --- PATCH di sicurezza: assicura almeno i self-loops se edge_index è vuoto ---
# --- PATCH di sicurezza: aggiungi self-loops se edge_index è vuoto ---
import torch
from torch_geometric.utils import add_self_loops
from models.lra_gnn import LRA_GNN  # assicurati che sia importato

_original_forward = LRA_GNN.forward

def _forward_with_safe_edges(self, data, *args, **kwargs):
    # edge_index potrebbe essere None o vuoto (E=0)
    ei = getattr(data, 'edge_index', None)
    if (ei is None) or (ei.numel() == 0):
        num_nodes = data.x.size(0)
        device = data.x.device
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
        ei, _ = add_self_loops(ei, num_nodes=num_nodes)
        data.edge_index = ei

    # passa qualunque argomento/keyword all'originale (es. return_features=True)
    return _original_forward(self, data, *args, **kwargs)

LRA_GNN.forward = _forward_with_safe_edges
# --- fine patch ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["MORPH", "FGNET", "CLAP2016", "UTKFACE"])
    parser.add_argument("--enable_lrc", action="store_true", default=False, help="Abilita LRC (Latent Relation Capturing)")
    parser.add_argument("--enable_dfe", action="store_true", default=False, help="Abilita DFE (Deep Feature Extraction)")
    parser.add_argument("--no_save_embeddings", action="store_true", help="Disabilita il salvataggio degli embeddings")
    args = parser.parse_args()

    dataset = args.dataset.upper()

    # 🔤 Crea suffisso descrittivo in base alle opzioni
    lrc_str = "lrc" if args.enable_lrc else "no_lrc"
    dfe_str = "dfe" if args.enable_dfe else "no_dfe"
    suffix = f"{lrc_str}_{dfe_str}"

    # 📂 Crea directory distinte per embeddings e checkpoints
    checkpoints_dir = f"checkpoints_ablation/{dataset.lower()}/{suffix}/"
    embeddings_dir = f"embeddings_ablation_{dataset.lower()}_{suffix}/"

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(embeddings_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(embeddings_dir, "val"), exist_ok=True)

    # 🚀 Avvia il training
    train_lra_gnn(
        dataset_name=dataset,
        enable_lrc=args.enable_lrc,
        enable_dfe=args.enable_dfe,
        use_prlae=False,  # Ablation => niente RL
        embeddings_dir=embeddings_dir,
        checkpoints_dir=checkpoints_dir,
        ablation=True,
        save_embeddings=not args.no_save_embeddings  # <-- aggiunto
    )