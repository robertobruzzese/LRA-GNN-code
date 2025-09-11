import os
import pandas as pd
import shutil

# 📂 Percorso base del dataset CLAP-2016
clap_base_dir = "/Users/robertobruzzese/Documents/progetto/DATASET-FACE-AGE-ESTIMATION/CLAP-2016 stratificato"
subfolders = ["train_1", "train_2", "test_1", "test_2", "valid"]

# 📄 Percorsi ai file GT
train_gt_path = os.path.join(clap_base_dir, "train_gt.csv")
val_gt_path   = os.path.join(clap_base_dir, "valid_gt.csv")

# 📁 Output directory per i nuovi CSV
output_dir = "datasets/data/CLAP2016"
os.makedirs(output_dir, exist_ok=True)

# 🔄 Funzione per caricare e salvare i CSV
def process_gt_file(gt_path, output_csv_name):
    if not os.path.exists(gt_path):
        print(f"❌ File non trovato: {gt_path}")
        return None

    df = pd.read_csv(gt_path)
    df["image"] = df["image"].astype(str)
    df = df[["image", "mean", "stdv"]]  # usa stdv come colonna della deviazione standard
    df["decade"] = (df["mean"] // 10).astype(int)
    df = df.rename(columns={"stdv": "std"})  # standardizza il nome

    output_path = os.path.join(output_dir, output_csv_name)
    df.to_csv(output_path, index=False)
    print(f"✅ Salvato: {output_path} ({len(df)} righe)")
    return df

# ▶️ Genera i due file
train_df = process_gt_file(train_gt_path, "CLAP_complete_train.csv")
val_df   = process_gt_file(val_gt_path, "CLAP_complete_val.csv")

# === Parte 2: Copia le immagini ===

# 📁 Directory di output immagini
output_train_dir = os.path.join(output_dir, "images", "Train")
output_val_dir   = os.path.join(output_dir, "images", "Validation")
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)

# 🔄 Funzione per trovare il percorso completo dell'immagine
def find_image_path(image_name):
    for subfolder in subfolders:
        path = os.path.join(clap_base_dir, subfolder, image_name)
        if os.path.exists(path):
            return path
    return None

# 🔽 Copia le immagini
def copy_images(df, dest_dir, split_name):
    missing = []
    for img_name in df["image"]:
        src_path = find_image_path(img_name)
        if src_path:
            shutil.copy(src_path, os.path.join(dest_dir, img_name))
        else:
            missing.append(img_name)
    print(f"📦 Copiate immagini per {split_name}: {len(df) - len(missing)} / {len(df)}")
    if missing:
        print(f"❌ Immagini mancanti in {split_name}: {missing}")

# ▶️ Esegui copia
if train_df is not None:
    copy_images(train_df, output_train_dir, "Train")

if val_df is not None:
    copy_images(val_df, output_val_dir, "Validation")