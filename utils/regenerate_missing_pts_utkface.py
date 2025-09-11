import os
import subprocess

splits = ["Train", "Validation"]
base_path = "datasets/data/UTKFace/images"
valid_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# Percorso script di landmark detection
extract_script = "utils/extract_landmarks_single.py"  # Adattalo se usi un nome diverso

# Crea cartella log se serve
os.makedirs("logs", exist_ok=True)
log_path = "logs/regenerate_missing_pts.log"

with open(log_path, "w") as log_file:
    for split in splits:
        preprocessed_dir = os.path.join(base_path, split, "images_preprocessed")
        points_dir = os.path.join(base_path, split, "points_reduced")
        os.makedirs(points_dir, exist_ok=True)

        image_files = [f for f in os.listdir(preprocessed_dir) if f.endswith(valid_exts)]

        print(f"\n🔍 Split: {split}", file=log_file)
        print(f"\n🔍 Split: {split}")

        for image_file in image_files:
            img_path = os.path.join(preprocessed_dir, image_file)
            pts_path = os.path.join(points_dir, f"{os.path.splitext(image_file)[0]}.pts")

            if not os.path.exists(pts_path):
                print(f"♻️  Rigenero .pts per {image_file}")
                print(f"♻️  Rigenero .pts per {image_file}", file=log_file)

                # Esegui lo script passando l'immagine come argomento
                result = subprocess.run(["python", extract_script, "--image", img_path, "--output", pts_path],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True)

                # Log esito
                log_file.write(result.stdout)
                log_file.write(result.stderr)

print(f"\n✅ Log completo in: {log_path}")