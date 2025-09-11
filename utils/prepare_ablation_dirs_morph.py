import os

# 📁 Esperimenti
experiments = ["no_lrc_dfe", "lrc_no_dfe", "lrc_dfe", "no_lrc_no_dfe"]
base_dst = "embeddings_ablation_morph"

for exp in experiments:
    for split in ["train", "val"]:
        dir_path = os.path.join(f"{base_dst}_{exp}", split)
        
        if os.path.exists(dir_path):
            print(f"🔍 Pulizia file in: {dir_path}")
            for f in os.listdir(dir_path):
                file_path = os.path.join(dir_path, f)
                if os.path.isfile(file_path) and f.endswith(".pt"):
                    os.remove(file_path)
                    print(f"🗑️ Rimosso: {file_path}")
        else:
            print(f"❗ Directory mancante: {dir_path} – la creo ora.")
            os.makedirs(dir_path)

print("✅ Pulizia completata. Le sottocartelle non sono state toccate.")