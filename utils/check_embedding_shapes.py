import torch
import os

path = "embeddings_ablation_morph_no_lrc_no_dfe/train/"
expected_dim = 512

for f in sorted(os.listdir(path)):
    if f.endswith(".pt"):
        file_path = os.path.join(path, f)
        try:
            data = torch.load(file_path)
            emb = data['embedding']
            print(f"{f}: {emb.shape}")
            if emb.shape[0] != expected_dim:
                print(f"❌ Dimensione non valida in {f}: {emb.shape}")
        except Exception as e:
            print(f"⚠️  Errore nel file {f}: {e}")