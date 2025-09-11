import os
import torch

folder = "embeddings_ablation_morph_no_lrc_no_dfe/train"
count_512 = 0
files_512 = []

for root, _, files in os.walk(folder):
    for f in files:
        if f.endswith(".pt"):
            path = os.path.join(root, f)
            try:
                obj = torch.load(path)
                if isinstance(obj, dict) and 'embedding' in obj:
                    emb = obj['embedding']
                    if isinstance(emb, torch.Tensor) and list(emb.shape) == [1, 512]:
                        count_512 += 1
                        files_512.append(path)
            except Exception as e:
                print(f"⚠️ Errore caricando {path}: {e}")

print(f"\n✅ Totale embedding con shape [1, 512]: {count_512}")
if files_512:
    print("📄 File corrispondenti:")
    for f in files_512:
        print(f"- {f}")