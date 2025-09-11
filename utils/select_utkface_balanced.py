import os
import shutil
import random
from collections import defaultdict

# 📁 Cartella con immagini UTKFace
utkface_dir = "/Users/robertobruzzese/Documents/progetto/DATASET-FACE-AGE-ESTIMATION/UTKFace/utkface_aligned_cropped/UTKFace"
train_dir = "datasets/data/UTKFace/images/Train/"
val_dir = "datasets/data/UTKFace/images/Validation/"

# 🔢 Parametri
decades = [(i, i + 9) for i in range(0, 100, 10)]  # [(0,9), ..., (90,99)]
images_per_decade = 200
val_ratio = 0.2
val_per_decade = int(images_per_decade * val_ratio)
train_per_decade = images_per_decade - val_per_decade

# 🧺 Raccoglitore per decade
decade_buckets = defaultdict(list)

# 📜 Scorri i file
for filename in sorted(os.listdir(utkface_dir)):
    if not filename.lower().endswith(".jpg"):
        continue
    try:
        age = int(filename.split("_")[0])
    except ValueError:
        continue
    if 0 <= age <= 100:
        for decade_start, decade_end in decades:
            if decade_start <= age <= decade_end:
                decade_buckets[(decade_start, decade_end)].append(filename)
                break

# 📤 Crea cartelle output
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 🔁 Seleziona immagini per decade
train_selected = []
val_selected = []

for (start, end), files in decade_buckets.items():
    random.shuffle(files)
    files = files[:images_per_decade]  # max 200
    n = len(files)
    val_count = int(round(n * val_ratio))
    train_count = n - val_count

    train_files = files[:train_count]
    val_files = files[train_count:]

    print(f"📦 Decade {start}-{end}: Train={len(train_files)} | Validation={len(val_files)}")

    for fname in train_files:
        src = os.path.join(utkface_dir, fname)
        dst = os.path.join(train_dir, fname)
        shutil.copy(src, dst)
        train_selected.append(fname)

    for fname in val_files:
        src = os.path.join(utkface_dir, fname)
        dst = os.path.join(val_dir, fname)
        shutil.copy(src, dst)
        val_selected.append(fname)

# ✅ Report finale
print(f"\n✅ Totale immagini selezionate: {len(train_selected)} in Train, {len(val_selected)} in Validation")
print(f"📁 Train: {train_dir}")
print(f"📁 Validation: {val_dir}")