import os
import torch

VAL_DIR = "embeddings_ablation_morph_no_lrc_dfe/val"
invalid_files = []

for fname in os.listdir(VAL_DIR):
    if not fname.endswith(".pt"):
        continue
    fpath = os.path.join(VAL_DIR, fname)
    try:
        obj = torch.load(fpath, map_location="cpu")
        age = obj.get("age", None)

        if not isinstance(age, (int, float)):
            print(f"❌ Errore: {fname} ha age={age} (tipo: {type(age)})")
            invalid_files.append(fname)

    except Exception as e:
        print(f"❌ Errore nel caricamento di {fname}: {e}")
        invalid_files.append(fname)

print(f"\n📊 File corrotti trovati: {len(invalid_files)}")