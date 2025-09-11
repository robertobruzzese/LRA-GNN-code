import os
import torch

# 📁 Modifica questo path in base alla tua cartella di validation
VAL_DIR = "embeddings_ablation_morph_no_lrc_dfe/val"

malformed_files = []

for subfolder in os.listdir(VAL_DIR):
    folder_path = os.path.join(VAL_DIR, subfolder)
    deep_feature_path = os.path.join(folder_path, "deep_features.pt")

    if not os.path.isdir(folder_path):
        continue
    if not os.path.isfile(deep_feature_path):
        print(f"⚠️ Manca: {deep_feature_path}")
        continue

    try:
        obj = torch.load(deep_feature_path, map_location="cpu")

        # 🔍 Caso 1: è un dizionario ma senza "features"
        if isinstance(obj, dict):
            if "features" not in obj:
                print(f"❌ Errore: {deep_feature_path} è un dizionario ma manca 'features'")
                malformed_files.append(deep_feature_path)
            elif not isinstance(obj["features"], torch.Tensor):
                print(f"❌ Errore: 'features' in {deep_feature_path} non è un tensore")
                malformed_files.append(deep_feature_path)

        # 🔍 Caso 2: è qualcosa di strano
        elif not isinstance(obj, torch.Tensor):
            print(f"❌ Formato inatteso in {deep_feature_path}: {type(obj)}")
            malformed_files.append(deep_feature_path)

    except Exception as e:
        print(f"❌ Errore nel caricamento di {deep_feature_path}: {e}")
        malformed_files.append(deep_feature_path)

print(f"\n🔍 File sospetti trovati: {len(malformed_files)}")