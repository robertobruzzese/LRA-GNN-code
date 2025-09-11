import os

splits = ["Train", "Validation"]
base_path = "datasets/data/UTKFace/images"
valid_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# 📁 Crea cartella logs se non esiste
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, "missing_keypoints_list.log")

with open(log_path, "w") as log_file:
    for split in splits:
        preprocessed_dir = os.path.join(base_path, split, "images_preprocessed")
        points_dir = os.path.join(base_path, split, "points_reduced")

        image_files = [f for f in os.listdir(preprocessed_dir) if f.endswith(".jpg") or f.endswith(".chip.jpg")]
        missing_pts = []

        for image_name in image_files:
            base_name = os.path.splitext(image_name)[0]
            pts_path = os.path.join(points_dir, f"{base_name}.pts")
            if not os.path.exists(pts_path):
                missing_pts.append(image_name)

        header = f"\n🔍 Controllo split: {split}\n❌ Mancanti .pts ({len(missing_pts)}):"
        print(header)
        log_file.write(header + "\n")

        for fname in missing_pts:
            print(f"   - {fname}")
            log_file.write(f"   - {fname}\n")

print(f"\n📄 Log salvato in: {log_path}")