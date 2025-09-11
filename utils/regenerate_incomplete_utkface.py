import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from graph_constructor_utkface import construct_initial_graph
from lrc import LatentRelationCapturer
from deep_feature_extraction import extract_deep_features
# === Config ===
splits = [
    {
        "patches_dir": "datasets/data/UTKFace/images/Train/patches",
        "images_dir": "datasets/data/UTKFace/images/Train/images_preprocessed",
        "embeddings_root": "embeddings_utkface/train"
    },
    {
        "patches_dir": "datasets/data/UTKFace/images/Validation/patches",
        "images_dir": "datasets/data/UTKFace/images/Validation/images_preprocessed",
        "embeddings_root": "embeddings_utkface/val"
    }
]

image_size = (224, 224)
num_patches_x, num_patches_y = 6, 6
threshold_distance = 0.88
tau_random_walk = 0.87
num_heads = 8
num_layers = 4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# === Lista file attesi in ogni cartella ===
required_files = [
    "graph_initial.pt",
    "graph_rw.pt",
    "patches_tensor.pt",
    "patch_to_node.pt",
    "deep_features.pt"
]

# === Funzione principale ===
def regenerate_incomplete_embeddings(patches_dir, images_dir, embeddings_root):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_filename in image_files:
        image_name = os.path.splitext(image_filename)[0]
        image_path = os.path.join(images_dir, image_filename)
        csv_path = os.path.join(patches_dir, f"{image_name}_patches.csv")
        embedding_dir = os.path.join(embeddings_root, image_name)

        # Controllo se la cartella è incompleta o assente
        regenerate = False
        if not os.path.exists(embedding_dir):
            regenerate = True
        else:
            for rf in required_files:
                fpath = os.path.join(embedding_dir, rf)
                if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                    regenerate = True
                    break

        if not regenerate:
            continue

        print(f"\n♻️ Rigenero {image_name}...")

        os.makedirs(embedding_dir, exist_ok=True)
        if not os.path.exists(csv_path):
            print(f"❌ CSV mancante: {csv_path}")
            continue

        result = construct_initial_graph(
            csv_path=csv_path,
            image_path=image_path,
            image_size=image_size,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            threshold_distance=threshold_distance,
            tau_random_walk=tau_random_walk,
            embedding_dir=embedding_dir
        )

        if result is None:
            print(f"⚠️ Skipping {image_name}: errore nella costruzione iniziale")
            continue

        graph_initial, graph_augmented, patches_tensor, patch_to_node = result

        # === Salvataggio grafi
        torch.save(graph_initial, os.path.join(embedding_dir, "graph_initial.pt"))
        torch.save(graph_augmented, os.path.join(embedding_dir, "graph_rw.pt"))
        torch.save(patches_tensor, os.path.join(embedding_dir, "patches_tensor.pt"))
        torch.save(patch_to_node, os.path.join(embedding_dir, "patch_to_node.pt"))

        # === LRC
        X = graph_augmented.x
        lrc = LatentRelationCapturer(in_dim=X.shape[1], num_heads=num_heads)
        A_m_list = lrc(X)

        for idx, A_m in enumerate(A_m_list):
            edge_index_m, edge_weight_m = dense_to_sparse(A_m)
            graph_m = Data(x=X, edge_index=edge_index_m, edge_attr=edge_weight_m)
            torch.save(graph_m, os.path.join(embedding_dir, f"graph_lrc_{idx}.pt"))

        # === Deep feature extraction
        graph_paths = sorted(glob.glob(os.path.join(embedding_dir, "graph_lrc_*.pt")))
        deep_feats = extract_deep_features(graph_paths, feature_dim=X.shape[1], num_layers=num_layers, device=device)
        torch.save(deep_feats, os.path.join(embedding_dir, "deep_features.pt"))

        print(f"✅ {image_name} rigenerato con successo.")

    print(f"\n🎉 Rigenerazione completata per {embeddings_root}.")

# === Avvia rigenerazione per ogni split ===
if __name__ == "__main__":
    for split in splits:
        print(f"\n🔍 Controllo e rigenerazione: {split['embeddings_root']}")
        regenerate_incomplete_embeddings(**split)