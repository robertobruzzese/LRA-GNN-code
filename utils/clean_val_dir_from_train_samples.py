# utils/clean_val_dir_from_train_samples.py

import os
import shutil

val_dir = "embeddings_ablation_morph_lrc_no_dfe/val"
existing_embeddings = {
    f.replace(".pt", "") for f in os.listdir(val_dir) if f.endswith(".pt")
}

all_dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]

removed = 0
for d in all_dirs:
    if d not in existing_embeddings:
        shutil.rmtree(os.path.join(val_dir, d))
        removed += 1

print(f"✅ Rimossi {removed} sample che non appartengono alla validazione.")
print(f"✅ Rimaste {len(existing_embeddings)} directory corrette.")