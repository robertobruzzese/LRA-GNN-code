import os
from PIL import Image
import matplotlib.pyplot as plt

# Elenco immagini fallite
failed_ids = [
    "001513", "000291", "004017", "004012", "006106",
    "002213", "005595", "006725", "003836", "005948", "003093"
]

images_dir = "datasets/data/CLAP2016/images/Validation/images_preprocessed"
output_dir = "debug_failed_images"
os.makedirs(output_dir, exist_ok=True)

for img_id in failed_ids:
    img_path = os.path.join(images_dir, f"{img_id}.jpg")
    if os.path.exists(img_path):
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"Image {img_id}")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"{img_id}.png"))
        plt.close()
    else:
        print(f"❌ Immagine non trovata: {img_path}")