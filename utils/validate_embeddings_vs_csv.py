import os
import pandas as pd

# === PERCORSI ===
csv_path = "datasets/data/CLAP2016/CLAP_balanced_val_decade0_8.csv"
embeddings_dir = "embeddings_clap2016/val"

# === LEGGE CSV ===
df = pd.read_csv(csv_path)
df["image"] = df["image"].astype(str).str.replace(".jpg", "", regex=False)

csv_image_ids = set(df["image"].tolist())

# === CARTELLE PRESENTI NELLA DIR ===
folder_names = set(
    fname for fname in os.listdir(embeddings_dir)
    if os.path.isdir(os.path.join(embeddings_dir, fname))
)

# === CONFRONTO ===
extra_folders = sorted(folder_names - csv_image_ids)
missing_folders = sorted(csv_image_ids - folder_names)

# === RISULTATI ===
print(f"✅ Totale immagini in CSV: {len(csv_image_ids)}")
print(f"✅ Totale cartelle embedding: {len(folder_names)}\n")

print(f"⚠️ Cartelle senza corrispondenza nel CSV ({len(extra_folders)}):")
for f in extra_folders:
    print(f"  - {f}")

print(f"\n⚠️ Immagini nel CSV senza cartella embedding ({len(missing_folders)}):")
for f in missing_folders:
    print(f"  - {f}")