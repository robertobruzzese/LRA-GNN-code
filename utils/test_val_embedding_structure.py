import os
import torch
from torch_geometric.data import Data

val_dir = "embeddings_ablation_morph_lrc_no_dfe/val"
samples = os.listdir(val_dir)

valid_count = 0
for sample in samples:
    sample_path = os.path.join(val_dir, sample)
    if not os.path.isdir(sample_path):
        print(f"❌ Non è una cartella: {sample}")
        continue

    graph_files = sorted([
        os.path.join(sample_path, f) for f in os.listdir(sample_path)
        if f.startswith("graph_lrc_") and f.endswith(".pt")
    ])

    if len(graph_files) != 8:
        print(f"⚠️ {sample}: attesi 8 file, trovati {len(graph_files)}")
        continue

    graphs = [torch.load(f) for f in graph_files]
    if all(isinstance(g, Data) for g in graphs):
        valid_count += 1
    else:
        print(f"⚠️ {sample}: almeno uno dei file non è un grafo valido.")

print(f"✅ Campioni validi con 8 grafi PyG: {valid_count}")