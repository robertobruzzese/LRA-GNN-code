import os
import glob
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from utils.graph_constructor_clap2016 import construct_initial_graph
from utils.lrc import LatentRelationCapturer
from utils.deep_feature_extraction import extract_deep_features

# === Configurazione generale ===
patches_dir_train = "datasets/data/CLAP2016/images/Train/patches"
patches_dir_val = "datasets/data/CLAP2016/images/Validation/patches"
images_dir_train = "datasets/data/CLAP2016/images/Train/images_preprocessed"
images_dir_val = "datasets/data/CLAP2016/images/Validation/images_preprocessed"
embeddings_root_train = "embeddings_clap2016/train"
embeddings_root_val = "embeddings_clap2016/val"

image_size = (224, 224)
num_patches_x, num_patches_y = 6, 6
threshold_distance = 0.88
tau_random_walk = 0.87
num_heads = 8
num_layers = 4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'


# === Funzione principale per ogni split ===
def process_split_from_csv(split, patches_dir, images_dir, embeddings_root):
    csv_path = os.path.join("datasets", "data", "CLAP2016", f"CLAP_complete_{split}.csv")
    if not os.path.exists(csv_path):
        print(f"❌ CSV non trovato: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df["image"] = df["image"].astype(str)
    image_ids = df["image"].str.replace(".jpg", "").tolist()

    for image_id in image_ids:
        image_filename = f"{image_id}.jpg"
        csv_patch_path = os.path.join(patches_dir, f"{image_id}_patches.csv")
        image_path = os.path.join(images_dir, image_filename)
        embedding_dir = os.path.join(embeddings_root, image_id)
        os.makedirs(embedding_dir, exist_ok=True)

        if not os.path.exists(csv_patch_path):
            print(f"⚠️  Salto {image_id}: CSV patch mancante ({csv_patch_path})")
            continue
        if not os.path.exists(image_path):
            print(f"⚠️  Salto {image_id}: immagine mancante ({image_path})")
            continue

        print(f"\n📂 Processing {image_id}...")

        # Step 1: Costruzione grafo iniziale e RW
        result = construct_initial_graph(
            csv_path=csv_patch_path,
            image_path=image_path,
            image_size=image_size,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            threshold_distance=threshold_distance,
            tau_random_walk=tau_random_walk,
            embedding_dir=embedding_dir
        )
        if result is None:
            print(f"❌ Skippata {image_id} per errore nelle patch.")
            continue

        graph_initial, graph_augmented, patches_tensor, patch_to_node = result
        torch.save(graph_initial, os.path.join(embedding_dir, "graph_initial.pt"))
        torch.save(graph_augmented, os.path.join(embedding_dir, "graph_rw.pt"))
        torch.save(patches_tensor, os.path.join(embedding_dir, "patches_tensor.pt"))
        torch.save(patch_to_node, os.path.join(embedding_dir, "patch_to_node.pt"))
        print("💾 Grafi iniziale e RW salvati.")

        # Step 2: LRC (multi-head attention)
        X = graph_augmented.x
        lrc = LatentRelationCapturer(in_dim=X.shape[1], num_heads=num_heads)
        A_m_list = lrc(X)
        for idx, A_m in enumerate(A_m_list):
            edge_index_m, edge_weight_m = dense_to_sparse(A_m)
            graph_m = Data(x=X, edge_index=edge_index_m, edge_attr=edge_weight_m)
            torch.save(graph_m, os.path.join(embedding_dir, f"graph_lrc_{idx}.pt"))
        print(f"💾 Grafi LRC ({num_heads} teste) salvati.")

        # Step 3: Deep Feature Extraction
        deep_feat_path = os.path.join(embedding_dir, "deep_features.pt")


        graph_paths = sorted(glob.glob(os.path.join(embedding_dir, "graph_lrc_*.pt")))
        if not graph_paths:  # Nessun file LRC trovato → fallback a DFE-only
            graph_rw_path = os.path.join(embedding_dir, "graph_rw.pt")
            if not os.path.exists(graph_rw_path):
                print(f"⚠️  Nessun grafo disponibile per {image_id}")
                continue
            graph_paths = [graph_rw_path]

        deep_feats = extract_deep_features(graph_paths, feature_dim=X.shape[1], num_layers=num_layers, device=device)
        torch.save(deep_feats, os.path.join(embedding_dir, "deep_features.pt"))
        print("💾 Deep features salvate.")

    print(f"\n✅ Split completato per {split.upper()}.")


# === Avvio ===
print("🔁 Processing TRAINING set...")
process_split_from_csv("train", patches_dir_train, images_dir_train, embeddings_root_train)

print("\n🔁 Processing VALIDATION set...")
process_split_from_csv("val", patches_dir_val, images_dir_val, embeddings_root_val)

print("\n✅ Tutte le immagini CLAP2016 sono state elaborate.")