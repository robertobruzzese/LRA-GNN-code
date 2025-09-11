import os

# === Percorsi delle cartelle ===
train_dir = "datasets/data/MORPH/images/Train"
preprocessed_dir = os.path.join(train_dir, "images_preprocessed")
points_reduced_dir = "datasets/data/MORPH/points_reduced"

# === Estensioni valide per le immagini ===
valid_image_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# === Prendi i nomi base (senza estensione) delle immagini originali ===
all_images = [os.path.splitext(f)[0] for f in os.listdir(train_dir) if f.lower().endswith(valid_image_exts)]
all_images_set = set(all_images)

# === Prendi i nomi base delle immagini preprocessate ===
preprocessed_images = [os.path.splitext(f)[0] for f in os.listdir(preprocessed_dir) if f.lower().endswith(valid_image_exts)]
preprocessed_images_set = set(preprocessed_images)

# === Prendi i nomi base dei file .pts nei punti ridotti ===
pts_files = [os.path.splitext(f)[0] for f in os.listdir(points_reduced_dir) if f.endswith(".pts")]
pts_files_set = set(pts_files)

# === Trova mancanti ===
missing_in_preprocessed = all_images_set - preprocessed_images_set
missing_pts = all_images_set - pts_files_set
fully_ok = all_images_set & preprocessed_images_set & pts_files_set

# === Stampa riepilogo ===
print(f"\n🔍 RIEPILOGO CHECK:")
print(f"📁 Immagini originali trovate: {len(all_images_set)}")
print(f"📁 Immagini preprocessate: {len(preprocessed_images_set)}")
print(f"📁 File .pts ridotti: {len(pts_files_set)}")
print(f"✅ Completamente processate (immagine + preproc + pts): {len(fully_ok)}")
print(f"❌ Mancano in preprocessate: {len(missing_in_preprocessed)}")
print(f"❌ Mancano .pts in points_reduced: {len(missing_pts)}")

# === Elenco dettagliato se vuoi ispezionare
if missing_in_preprocessed:
    print("\n⚠️ Immagini mancanti in 'images_preprocessed':")
    for name in sorted(missing_in_preprocessed):
        print(" -", name)

if missing_pts:
    print("\n⚠️ File .pts mancanti in 'points_reduced':")
    for name in sorted(missing_pts):
        print(" -", name)

if fully_ok:
    print("\n✅ Immagini completamente processate:")
    for name in sorted(fully_ok):
        print(" -", name)
