import os
import torch

base_dir = "embeddings_ablation_morph_lrc_no_dfe/train"
samples = sorted(os.listdir(base_dir))

missing_age = []

for sample in samples:
    sample_dir = os.path.join(base_dir, sample)
    embedding_path = os.path.join(sample_dir, "embedding.pt")
    if os.path.exists(embedding_path):
        try:
            data = torch.load(embedding_path)
            if 'age' not in data:
                missing_age.append(sample)
        except Exception as e:
            print(f"⚠️ Errore nel file {embedding_path}: {e}")

if missing_age:
    print("❌ I seguenti sample NON hanno il campo 'age' in embedding.pt:")
    for s in missing_age:
        print(f"  - {s}")
else:
    print("✅ Tutti i file embedding.pt contengono il campo 'age'")