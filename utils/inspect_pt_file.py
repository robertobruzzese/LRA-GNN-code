import torch

file_path = "embeddings_FGNET/train/sample_139.pt"  # Modifica se necessario

data = torch.load(file_path, map_location="cpu")

print("📦 Tipo di oggetto:", type(data))

if isinstance(data, dict):
    print("🔑 Chiavi presenti:", data.keys())
    if 'age' in data:
        print("🧠 Età:", data['age'])
    if 'y' in data:
        print("🎯 Label:", data['y'])
else:
    # PyG Data object?
    print("📋 Attributi disponibili:", [k for k in dir(data) if not k.startswith("_")])
    if hasattr(data, "age"):
        print("🧠 Età:", data.age)
    if hasattr(data, "y"):
        print("🎯 Label:", data.y)