import os
import torch
from torch_geometric.data import Data

embeddings_dir = "embeddings_ablation_morph_no_lrc_dfe/train"  # o val

for fname in os.listdir(embeddings_dir):
    g_path = os.path.join(embeddings_dir, fname, "graph_rw.pt")
    if os.path.exists(g_path):
        try:
            g = torch.load(g_path)
            if isinstance(g, Data) and g.edge_index.size(1) == 0:
                print(f"⚠️ Grafo vuoto: {fname}")
        except Exception as e:
            print(f"❌ Errore su {fname}: {e}")