import os
import torch
import matplotlib.pyplot as plt
from collections import Counter

# 🔍 Directory del validation set
embedding_dir = "embeddings_ablation_utkface_lrc_no_dfe/val"

# 📦 Lista file .pt
pt_files = [f for f in os.listdir(embedding_dir) if f.endswith(".pt")]

if len(pt_files) == 0:
    raise FileNotFoundError(f"Nessun file .pt trovato in {embedding_dir}")

# 📊 Conta classi (decadi)
decade_classes = []

for pt_file in pt_files:
    full_path = os.path.join(embedding_dir, pt_file)
    data = torch.load(full_path)
    age = data["age"]
    age_class = int(age) // 10
    decade_classes.append(age_class)

# 📈 Frequenze
class_counts = Counter(decade_classes)
classes = sorted(class_counts.keys())
frequencies = [class_counts[c] for c in classes]

# 🎨 Plot
plt.figure(figsize=(6, 5))
plt.bar(classes, frequencies)
plt.xlabel("Classe (decade)")
plt.ylabel("Numero di esempi")
plt.title("Distribuzione classi nel validation set (UTKFACE, LRC)")
plt.grid(True)
plt.tight_layout()
plt.show()