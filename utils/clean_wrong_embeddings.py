import os
import torch

base_dir = "embeddings_ablation_morph_no_lrc_no_dfe"
subdirs = ["train", "val"]
expected_dim = 128

for sub in subdirs:
    embedding_dir = os.path.join(base_dir, sub)
    if not os.path.isdir(embedding_dir):
        print(f"⛔ Cartella non trovata: {embedding_dir}")
        continue

    print(f"\n🔍 Controllo embedding in: {embedding_dir}")
    
    for f in sorted(os.listdir(embedding_dir)):
        if f.endswith(".pt"):
            path = os.path.join(embedding_dir, f)
            try:
                data = torch.load(path)
                emb_dim = data["embedding"].shape[0]
                if emb_dim != expected_dim:
                    print(f"🗑️  Rimuovo {sub}/{f} (dim={emb_dim})")
                    os.remove(path)
            except Exception as e:
                print(f"⚠️ Errore con {f}: {e}")