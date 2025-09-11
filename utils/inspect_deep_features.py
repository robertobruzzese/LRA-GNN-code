import torch
import os
from torch_geometric.data import Data

# 🔧 MODIFICA QUI con il percorso di un file specifico
path = "embeddings_ablation_morph_no_lrc_dfe/train/03712_03M31/deep_features.pt"

# Carica il file
obj = torch.load(path)

print(f"✅ Tipo del file: {type(obj)}")

if isinstance(obj, Data):
    print("🔍 È un oggetto Data.")
    print(f"x shape: {obj.x.shape}")
    print(f"edge_index shape: {obj.edge_index.shape if obj.edge_index is not None else 'None'}")
    print(f"Contiene attributi: {obj.__dict__}")
elif isinstance(obj, torch.Tensor):
    print("🔍 È un Tensore puro.")
    print(f"Shape: {obj.shape}")
    print(f"Contenuto: {obj}")
else:
    print("⚠️ Tipo non gestito:", type(obj))