# utils/check_missing_embeddings_morph_ablation.py

import os

VAL_DIR = "embeddings_ablation_morph_lrc_no_dfe/val"

def check_embeddings(val_dir):
    missing_embeddings = []

    for subdir in os.listdir(val_dir):
        subdir_path = os.path.join(val_dir, subdir)

        # Salta se non è una directory
        if not os.path.isdir(subdir_path):
            continue

        # Path atteso dell'embedding salvato
        expected_embedding_path = os.path.join(val_dir, f"{subdir}.pt")

        if not os.path.exists(expected_embedding_path):
            missing_embeddings.append(subdir)

    print(f"🔍 Totale directory in val/: {len(os.listdir(val_dir))}")
    print(f"❌ Embedding mancanti: {len(missing_embeddings)}")
    for name in missing_embeddings:
        print(f" - {name}")

if __name__ == "__main__":
    check_embeddings(VAL_DIR)