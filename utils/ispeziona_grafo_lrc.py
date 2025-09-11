import os
import torch
from tqdm import tqdm
from torch_geometric.data import Data

# 🔁 Cartella contenente i grafi LRC
base_dir = "embeddings_ablation_clap2016_lrc_no_dfe/val"

# 🔢 Numero di teste LRC
num_heads = 8

# 📋 Statistiche
missing_x = []
none_x = []
wrong_shape = []

# 🔍 Per ogni sottocartella (immagine)
for img_name in tqdm(os.listdir(base_dir), desc="🔍 Scansione grafi"):
    img_path = os.path.join(base_dir, img_name)
    if not os.path.isdir(img_path):
        continue

    for i in range(num_heads):
        graph_path = os.path.join(img_path, f"graph_lrc_{i}.pt")
        if not os.path.exists(graph_path):
            continue

        try:
            data = torch.load(graph_path)
        except Exception as e:
            print(f"❌ Errore nel caricamento di {graph_path}: {e}")
            continue

        if not hasattr(data, 'x'):
            missing_x.append(graph_path)
        elif data.x is None:
            none_x.append(graph_path)
        elif data.x.shape[0] == 0 or len(data.x.shape) != 2:
            wrong_shape.append((graph_path, data.x.shape))

# ✅ Report finale
print("\n🔍 Report finale:")
print(f"Grafi senza attributo x      : {len(missing_x)}")
print(f"Grafi con x = None           : {len(none_x)}")
print(f"Grafi con shape sospetto     : {len(wrong_shape)}")

if missing_x:
    print("\n❌ Grafi senza attributo x:")
    for path in missing_x[:5]:
        print(" -", path)
    if len(missing_x) > 5:
        print("...")

if none_x:
    print("\n⚠️ Grafi con x = None:")
    for path in none_x[:5]:
        print(" -", path)
    if len(none_x) > 5:
        print("...")

if wrong_shape:
    print("\n📏 Grafi con shape sospetto:")
    for path, shape in wrong_shape[:5]:
        print(f" - {path} | shape: {shape}")
    if len(wrong_shape) > 5:
        print("...")