import os
import torch
import hashlib

def get_hash(filepath):
    data = torch.load(filepath)
    m = hashlib.sha256()
    m.update(data["embedding"].numpy().tobytes())
    m.update(str(data["age"]).encode())
    return m.hexdigest()

train_dir = "embeddings_clap2016/train"
val_dir = "embeddings_clap2016/val"

train_hashes = set()
val_hashes = set()

for filename in os.listdir(train_dir):
    if filename.endswith(".pt"):
        train_hashes.add(get_hash(os.path.join(train_dir, filename)))

for filename in os.listdir(val_dir):
    if filename.endswith(".pt"):
        val_hashes.add(get_hash(os.path.join(val_dir, filename)))

common = train_hashes & val_hashes

print(f"🔁 File identici in train e val: {len(common)}")