import os
import pandas as pd
import shutil

# 📂 Percorso base dove si trovano tutte le cartelle delle immagini
clap_base_dir = "/Users/robertobruzzese/Documents/progetto/DATASET-FACE-AGE-ESTIMATION/CLAP-2016 stratificato"
subfolders = ["train_1", "train_2", "test_1", "test_2", "valid"]

# 📥 Percorsi CSV input
csv_train_path = os.path.join(clap_base_dir, "CLAP_balanced_train_decade0_8.csv")
csv_val_path = os.path.join(clap_base_dir, "CLAP_balanced_val_decade0_8.csv")

# 📤 Directory di destinazione nel progetto
output_train_dir = "datasets/data/CLAP2016/images/Train"
output_val_dir = "datasets/data/CLAP2016/images/Validation"

# 🧼 Crea directory se non esistono
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)

# 📄 Leggi CSV
train_df = pd.read_csv(csv_train_path)
val_df = pd.read_csv(csv_val_path)

# 🔄 Funzione per trovare il percorso completo dell'immagine
def find_image_path(image_name):
    for subfolder in subfolders:
        path = os.path.join(clap_base_dir, subfolder, image_name)
        if os.path.exists(path):
            return path
    return None

# 🟦 Copia immagini del training set
print("📦 Copia immagini per il training set...")
missing_train = []
for img_name in train_df["image"]:
    src_path = find_image_path(img_name)
    if src_path:
        shutil.copy(src_path, os.path.join(output_train_dir, img_name))
    else:
        missing_train.append(img_name)

# 🟨 Copia immagini del validation set
print("📦 Copia immagini per il validation set...")
missing_val = []
for img_name in val_df["image"]:
    src_path = find_image_path(img_name)
    if src_path:
        shutil.copy(src_path, os.path.join(output_val_dir, img_name))
    else:
        missing_val.append(img_name)

# ✅ Report finale
print(f"\n✅ Completato!")
print(f"Train immagini copiate: {len(train_df) - len(missing_train)} / {len(train_df)}")
print(f"Validation immagini copiate: {len(val_df) - len(missing_val)} / {len(val_df)}")

if missing_train or missing_val:
    print("\n❌ Immagini mancanti:")
    if missing_train:
        print("Training:", missing_train)
    if missing_val:
        print("Validation:", missing_val)