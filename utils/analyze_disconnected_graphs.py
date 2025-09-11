import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Percorsi da modificare secondo la tua struttura
train_dir = "embeddings_clap2016/train"
val_dir = "embeddings_clap2016/val"
img_dir = "datasets/data/CLAP2016/images/Train/images_preprocessed"

disconnected_graphs = [
    "001847.jpg", "000024.jpg", "001513.jpg", "004012.jpg", "002507.jpg", "002213.jpg",
    "004017.jpg", "005595.jpg", "006725.jpg", "003836.jpg", "005948.jpg", "003093.jpg"
]

def find_graph_dir(img_name):
    found = None
    for base_dir in [train_dir, val_dir]:
        candidate = os.path.join(base_dir, img_name.replace(".jpg", ""))
        print(f"🔍 Cerco: {candidate}")
        if os.path.isdir(candidate):
            found = candidate
            break
    if not found:
        print(f"❌ Non trovata cartella per {img_name}")
    return found

def analyze_graph(img_name):
    graph_dir = find_graph_dir(img_name)
    if not graph_dir:
        print(f"❌ Cartella non trovata per {img_name}")
        return None

    node_feat_path = os.path.join(graph_dir, "node_features.pt")
    if not os.path.exists(node_feat_path):
        print(f"❌ node_features.pt non trovato per {img_name}")
        return None

    try:
        node_features = torch.load(node_feat_path)
    except Exception as e:
        print(f"❌ Errore nel caricamento di {node_feat_path}: {e}")
        return None

    if torch.isnan(node_features).any():
        print(f"⚠️ NaN presenti negli embedding di {img_name}")
        return {"image": img_name, "error": "NaN in embeddings"}

    node_np = node_features.numpy()
    sim_matrix = np.inner(node_np, node_np)
    triu_indices = np.triu_indices(len(node_np), k=1)
    similarities = sim_matrix[triu_indices]
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    max_sim = np.max(similarities)
    min_sim = np.min(similarities)

    # Mostra immagine se esiste
    original_img_path = os.path.join(img_dir, img_name)
    if os.path.exists(original_img_path):
        img = Image.open(original_img_path)
        img.show(title=f"Image: {img_name}")
    else:
        print(f"⚠️ Immagine non trovata: {original_img_path}")

    # Istogramma delle similarità
    plt.figure(figsize=(6, 4))
    plt.hist(similarities, bins=20, color='skyblue')
    plt.title(f"Similarities for {img_name}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    return {
        "image": img_name,
        "n_nodes": len(node_np),
        "mean_similarity": round(mean_sim, 4),
        "std_similarity": round(std_sim, 4),
        "max_similarity": round(max_sim, 4),
        "min_similarity": round(min_sim, 4)
    }

if __name__ == "__main__":
    results = []
    for img_name in disconnected_graphs:
        print(f"📊 Analisi: {img_name}")
        result = analyze_graph(img_name)
        if result:
            results.append(result)

    df = pd.DataFrame(results)
    print("\n=== RISULTATI STATISTICI ===\n")
    print(df.to_string(index=False))
