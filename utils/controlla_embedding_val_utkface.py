import torch
import os

# Path alla cartella embeddings VAL per LRC-only
val_dir = "embeddings_ablation_utkface_lrc_no_dfe/val"

bad_files = 0
ages = []

for file in os.listdir(val_dir):
    if not file.endswith(".pt"):
        continue

    full_path = os.path.join(val_dir, file)
    try:
        data = torch.load(full_path)

        if not isinstance(data, dict):
            print(f"❌ {file}: non è un dizionario.")
            bad_files += 1
            continue

        emb = data.get("embedding", None)
        age = data.get("age", None)

        if emb is None or age is None:
            print(f"❌ {file}: manca 'embedding' o 'age'.")
            bad_files += 1
            continue

        if not isinstance(emb, torch.Tensor) or emb.ndim != 1 or emb.shape[0] != 512:
            print(f"❌ {file}: embedding non valido: shape {emb.shape}")
            bad_files += 1
            continue

        if not isinstance(age, (float, int)) or age < 0 or age > 120:
            print(f"❌ {file}: età non valida: {age}")
            bad_files += 1
            continue

        ages.append(age)

    except Exception as e:
        print(f"❌ {file}: errore durante il load -> {e}")
        bad_files += 1

print(f"\n✅ Controllati {len(ages) + bad_files} file.")
print(f"❌ File non validi: {bad_files}")
print(f"📊 Età - Min: {min(ages) if ages else 'n/a'}, Max: {max(ages) if ages else 'n/a'}, Media: {sum(ages)/len(ages) if ages else 'n/a'}")