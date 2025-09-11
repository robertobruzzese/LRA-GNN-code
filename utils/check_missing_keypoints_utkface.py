import os

# === Config ===
splits = ["Train", "Validation"]
base_path = "datasets/data/UTKFace/images"
valid_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
log_path = "missing_keypoints.log"

missing_all = []

with open(log_path, "w") as log_file:
    for split in splits:
        print(f"\n🔍 Controllo split: {split}")
        log_file.write(f"\n🔍 Controllo split: {split}\n")

        preprocessed_dir = os.path.join(base_path, split, "images_preprocessed")
        points_dir = os.path.join(base_path, split, "points_reduced")

        missing = []
        files = [f for f in os.listdir(preprocessed_dir) if f.lower().endswith(valid_exts)]

        for f in files:
            name_wo_ext = os.path.splitext(f)[0]
            pts_path = os.path.join(points_dir, f"{name_wo_ext}.pts")
            if not os.path.exists(pts_path):
                missing.append(f)
                log_file.write(f"❌ Mancano i keypoints per: {f}\n")

        print(f"📦 Totali immagini: {len(files)}")
        print(f"❌ Mancanti .pts: {len(missing)}")

        log_file.write(f"📦 Totali immagini: {len(files)}\n")
        log_file.write(f"❌ Mancanti .pts: {len(missing)}\n")

        missing_all.extend(missing)

print(f"\n📄 Log salvato in: {log_path}")