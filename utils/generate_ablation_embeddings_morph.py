import os
import torch
import shutil
import re

# Config
SOURCE_ROOT = "embeddings_morph"
TARGET_ROOT = "embeddings_ablation_morph"
LRC_HEADS = 8
SPLITS = ["train", "val"]


# Estrae età da nome file (es. 061002_7F46 → 46)
def extract_age(folder_name):
    match = re.search(r'(\d{2})$', folder_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Formato nome non valido per età: {folder_name}")

# Crea le versioni richieste per una singola immagine
def process_one_image(split, folder_name):
    source_dir = os.path.join(SOURCE_ROOT, split, folder_name)
    deep_path = os.path.join(source_dir, "deep_features.pt")
    rw_path = os.path.join(source_dir, "graph_rw.pt")

    # Estrai età
    try:
        age = extract_age(folder_name)
    except ValueError as e:
        print(f"❓ {e}")
        return

    # 📁 Caso no_lrc_no_dfe → usa solo graph_rw
    if os.path.exists(rw_path):
        try:
            rw_data = torch.load(rw_path)
            rw_mean = rw_data.x.mean(dim=0)
            out_dir = os.path.join(f"{TARGET_ROOT}_no_lrc_no_dfe", split)
            os.makedirs(out_dir, exist_ok=True)
            torch.save({"embedding": rw_mean, "age": age}, os.path.join(out_dir, f"{folder_name}.pt"))
        except Exception as e:
            print(f"❌ Errore RW {folder_name}: {e}")

    # 📁 Caso no_lrc_dfe → usa solo deep_features
    if os.path.exists(deep_path):
        try:
            deep_feat = torch.load(deep_path)
            deep_mean = deep_feat.mean(dim=0)
            out_dir = os.path.join(f"{TARGET_ROOT}_no_lrc_dfe", split)
            os.makedirs(out_dir, exist_ok=True)
            torch.save({"embedding": deep_mean, "age": age}, os.path.join(out_dir, f"{folder_name}.pt"))
        except Exception as e:
            print(f"❌ Errore DFE {folder_name}: {e}")

    # 📁 caso lrc_no_dfe LRC (media tra le 8 heads)
    lrc_features = []
    for i in range(LRC_HEADS):
        lrc_path = os.path.join(source_dir, f"graph_lrc_{i}.pt")
        if not os.path.exists(lrc_path):
            print(f"⚠️ Mancante {lrc_path} → LRC saltato per {folder_name}")
            lrc_features = []
            break
        try:
            data = torch.load(lrc_path)
            lrc_features.append(data.x)
        except Exception as e:
            print(f"❌ Errore LRC {i} {folder_name}: {e}")
            lrc_features = []
            break

    if lrc_features:
        try:
            lrc_stack = torch.stack(lrc_features)  # (8, N, 512)
            lrc_mean = lrc_stack.mean(dim=(0, 1))  # (512,)

            # 📁 lrc_no_dfe
            out_dir = os.path.join(f"{TARGET_ROOT}_lrc_no_dfe", split)
            os.makedirs(out_dir, exist_ok=True)
            torch.save({"embedding": lrc_mean, "age": age}, os.path.join(out_dir, f"{folder_name}.pt"))

            # 📁 lrc_dfe → concat(deep, lrc)
            if os.path.exists(deep_path):
                deep_feat = deep_feat.to(lrc_mean.device)  # allinea il device
                deep_mean = deep_feat.mean(dim=0)
                concat_mean = torch.cat([deep_mean, lrc_mean], dim=0)  # (1024,)
                out_dir = os.path.join(f"{TARGET_ROOT}_lrc_dfe", split)
                os.makedirs(out_dir, exist_ok=True)
                torch.save({"embedding": concat_mean, "age": age}, os.path.join(out_dir, f"{folder_name}.pt"))
        except Exception as e:
            print(f"❌ Errore combinazione LRC+DFE {folder_name}: {e}")

# 🔁 Esegui su tutto il dataset
def main():
    
    for split in SPLITS:
        split_dir = os.path.join(SOURCE_ROOT, split)
        for folder in os.listdir(split_dir):
            if os.path.isdir(os.path.join(split_dir, folder)):
                process_one_image(split, folder)

    print("✅ Embedding ablation completata.")

if __name__ == "__main__":
    main()