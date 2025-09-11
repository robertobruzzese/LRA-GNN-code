import os
import torch
from collections import defaultdict

embedding_root = "embeddings_ablation_morph_no_lrc_no_dfe/train"
dimension_counts = defaultdict(int)

for subdir, _, files in os.walk(embedding_root):
    for file in files:
        if file.endswith(".pt"):
            path = os.path.join(subdir, file)
            try:
                data = torch.load(path)
                if isinstance(data, dict):
                    # per file salvati come dict con 'embedding'
                    if 'embedding' in data:
                        tensor = data['embedding']
                    else:
                        continue
                else:
                    tensor = data
                dim = tensor.shape[-1]
                dimension_counts[dim] += 1
            except Exception as e:
                print(f"Errore caricando {path}: {e}")

# Stampa le dimensioni trovate
print("\n📊 Dimensioni trovate tra gli embedding:")
for dim, count in dimension_counts.items():
    print(f"- {count} embedding con dimensione {dim}")