import torch
import os

# 🔁 Sostituisci con un sample reale nella tua cartella
sample_dir = "embeddings_ablation_morph_no_lrc_dfe/train/01_0M32"
file_path = os.path.join(sample_dir, "deep_features.pt")

# 📦 Carica
embedding_data = torch.load(file_path)

# 🧠 Verifica formato
if isinstance(embedding_data, dict):
    embedding = embedding_data['embedding']
elif isinstance(embedding_data, torch.Tensor):
    embedding = embedding_data
else:
    raise ValueError("Formato non riconosciuto")

print(f"✅ Embedding shape: {embedding.shape}")