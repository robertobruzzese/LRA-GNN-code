import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv

# === Percorsi FGNET ===
splits = ["Train", "Validation"]
base_path = "datasets/data/FGNET/images"
valid_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

for split in splits:
    preprocessed_dir = os.path.join(base_path, split, "images_preprocessed")
    points_dir = os.path.join(base_path, split, "points_reduced")
    patches_dir = os.path.join(base_path, split, "patches")
    segmented_dir = os.path.join(base_path, split, "images_segmented")

    os.makedirs(patches_dir, exist_ok=True)
    os.makedirs(segmented_dir, exist_ok=True)

    image_files = [f for f in os.listdir(preprocessed_dir) if f.lower().endswith(valid_exts)]

    for image_name in image_files:
        image_path = os.path.join(preprocessed_dir, image_name)
        image_base = os.path.splitext(image_name)[0]
        keypoints_path = os.path.join(points_dir, f"{image_base}.pts")
        csv_path = os.path.join(patches_dir, f"{image_base}_patches.csv")

        if not os.path.exists(keypoints_path):
            print(f"⚠️  Keypoints non trovati per {image_name}, salto.")
            continue

        print(f"\n📂 Elaborazione immagine: {image_name}")

        # === STEP 1: Caricamento immagine ===
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        print(f"📏 Dimensioni immagine: {width}x{height}")

        # === STEP 2: Caricamento keypoints ===
        keypoints = []
        with open(keypoints_path, "r") as f:
            lines = f.readlines()
            reading = False
            for line in lines:
                line = line.strip()
                if line == "{":
                    reading = True
                    continue
                elif line == "}":
                    break
                if reading:
                    try:
                        x, y = map(float, line.split())
                        keypoints.append([x, y])
                    except ValueError:
                        continue
        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        # === STEP 3: Suddivisione in Patch (6x6) ===
        num_patches_x = 6
        num_patches_y = 6
        patch_w = width // num_patches_x
        patch_h = height // num_patches_y
        patches = []
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                x_start = i * patch_w
                y_start = j * patch_h
                x_end = x_start + patch_w
                y_end = y_start + patch_h
                patches.append(((x_start, y_start), (x_end, y_end)))

        # === STEP 4: Mappatura keypoints → patch ===
        keypoints_np = keypoints.numpy()
        keypoint_patch_map = {}
        for kp in keypoints_np:
            x_kp, y_kp = kp
            for idx, ((x_start, y_start), (x_end, y_end)) in enumerate(patches):
                if x_start <= x_kp < x_end and y_start <= y_kp < y_end:
                    keypoint_patch_map[idx] = keypoint_patch_map.get(idx, []) + [kp]

        # === STEP 5: Scrittura CSV ===
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["patch_id", "keypoints"])
            for patch_idx, keypoints_list in keypoint_patch_map.items():
                keypoints_str = ";".join([f"{kp[0]:.2f},{kp[1]:.2f}" for kp in keypoints_list])
                writer.writerow([patch_idx, keypoints_str])
        print(f"💾 Salvato CSV in {csv_path}")

        # === STEP 6: Visualizzazione ===
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image)
        for idx, ((x_start, y_start), (x_end, y_end)) in enumerate(patches):
            ax.plot([x_start, x_end], [y_start, y_start], color="red", linewidth=1)
            ax.plot([x_start, x_end], [y_end, y_end], color="red", linewidth=1)
            ax.plot([x_start, x_start], [y_start, y_end], color="red", linewidth=1)
            ax.plot([x_end, x_end], [y_start, y_end], color="red", linewidth=1)
            cx = (x_start + x_end) // 2
            cy = (y_start + y_end) // 2
            ax.text(cx, cy, str(idx), color="white", fontsize=10, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.4, boxstyle='circle'))

        for (x_kp, y_kp) in keypoints_np:
            ax.scatter(x_kp, y_kp, c="blue", s=50)

        for patch_idx in keypoint_patch_map.keys():
            (x_start, y_start), (x_end, y_end) = patches[patch_idx]
            rect = plt.Rectangle((x_start, y_start), patch_w, patch_h, edgecolor='yellow', facecolor='none', linewidth=2)
            ax.add_patch(rect)

        vis_save_path = os.path.join(segmented_dir, f"{image_base}_patches_vis.jpg")
        plt.savefig(vis_save_path)
        print(f"🖼️ Visualizzazione salvata in: {vis_save_path}")
        plt.close()