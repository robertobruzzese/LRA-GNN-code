import os

directory = "embeddings_morph/val"

for filename in os.listdir(directory):
    new_name = filename.replace("'", "").replace("[", "").replace("]", "")
    if filename != new_name:
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        print(f"🔁 Rinominando: {filename} → {new_name}")
        os.rename(old_path, new_path)

print("✅ Pulizia completata.")