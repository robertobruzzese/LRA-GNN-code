import os
from utils.preprocessing import preprocess_image

# === Percorsi delle cartelle ===
base_dir = "datasets/data/FGNET/images"
splits = ["Train", "Validation"]

for split in splits:
    input_dir = os.path.join(base_dir, split)
    output_dir = os.path.join(input_dir, "images_preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]

    print(f"\n🔍 [{split}] Trovate {len(image_files)} immagini da preprocessare...")

    for idx, image_name in enumerate(image_files, 1):
        image_path = os.path.join(input_dir, image_name)
        preprocessed_image_path = os.path.join(output_dir, image_name)

        try:
            print(f"🖼️ [{split}] [{idx}/{len(image_files)}] Preprocessing: {image_name}")
            preprocess_image(image_path, preprocessed_image_path)
        except Exception as e:
            print(f"⚠️ Errore durante il preprocessing di {image_name}: {e}")

print("\n✅ Preprocessing batch completato per Train e Validation.")
