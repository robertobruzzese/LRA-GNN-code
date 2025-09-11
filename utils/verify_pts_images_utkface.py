import os

split = "Train"  # o "Validation"
base_path = "datasets/data/UTKFace/images"
images_dir = os.path.join(base_path, split, "images_preprocessed")
points_dir = os.path.join(base_path, split, "points_reduced")

missing = []

for img in os.listdir(images_dir):
    if img.lower().endswith((".jpg", ".jpeg", ".png")):
        base_name = os.path.splitext(img)[0]
        pts_path = os.path.join(points_dir, f"{base_name}.pts.chip.pts")
        if not os.path.exists(pts_path):
            missing.append(img)

print(f"📂 Split: {split}")
print(f"❌ Mancano i keypoints per {len(missing)} immagini:")
for m in missing:
    print(" -", m)