import os

base_path = "embeddings"

for root, dirs, _ in os.walk(base_path):
    for d in dirs:
        full_path = os.path.join(root, d)
        if os.path.isdir(full_path) and not os.listdir(full_path):
            print(f"🧹 Rimuovo cartella vuota: {full_path}")
            os.rmdir(full_path)
