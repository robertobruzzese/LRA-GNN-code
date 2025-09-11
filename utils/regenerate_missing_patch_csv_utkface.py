import os
import subprocess
from datetime import datetime

splits = ["Train", "Validation"]
base_path = "datasets/data/UTKFace/images"
log_path = "logs/missing_patch_csv.log"

os.makedirs("logs", exist_ok=True)

with open(log_path, "w") as log_file:
    log_file.write(f"📅 Log generazione patch CSV - {datetime.now()}\n\n")

    for split in splits:
        preprocessed_dir = os.path.join(base_path, split, "images_preprocessed")
        patches_dir = os.path.join(base_path, split, "patches")
        image_files = [f for f in os.listdir(preprocessed_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        missing = []
        for image_name in image_files:
            base_name = os.path.splitext(image_name)[0]
            csv_path = os.path.join(patches_dir, f"{base_name}_patches.csv")
            if not os.path.exists(csv_path):
                missing.append(image_name)

        log_file.write(f"\n🔍 Split: {split} - Immagini senza CSV: {len(missing)}\n")
        for img in missing:
            log_file.write(f"  ⛔ {img}\n")

        if missing:
            # Esegui lo script originale segment_patches_utkface.py per questo split
            print(f"\n🚀 Eseguo segmentazione per split {split}...")
            env = os.environ.copy()
            env["SPLIT_OVERRIDE"] = split  # per passare lo split se lo script lo supporta
            subprocess.run(["python", "utils/segment_patches_utkface.py"], env=env)

    log_file.write("\n✅ Fine rigenerazione.\n")