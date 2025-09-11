import os
import random
import shutil

# 📂 Directory sorgente con tutte le immagini
source_dir = "datasets/data/FGNET/images/"
# 📂 Destinazioni
train_dir = "datasets/data/FGNET/images/Train/"
val_dir = "datasets/data/FGNET/images/Validation/"

# 📦 Crea cartelle di destinazione se non esistono
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 📋 Elenco di tutti i file immagine
all_images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
random.shuffle(all_images)  # 🔀 Shuffle per random split

# 🔢 Split
total = len(all_images)
split_idx = int(0.8 * total)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# 📤 Copia file
for img in train_images:
    shutil.copy(os.path.join(source_dir, img), os.path.join(train_dir, img))

for img in val_images:
    shutil.copy(os.path.join(source_dir, img), os.path.join(val_dir, img))

print(f"✅ Split completato: {len(train_images)} in train, {len(val_images)} in val.")