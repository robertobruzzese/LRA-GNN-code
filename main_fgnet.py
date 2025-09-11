import os
import glob
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from utils.graph_constructor_fgnet import construct_initial_graph
from utils.lrc import LatentRelationCapturer
from utils.deep_feature_extraction import extract_deep_features

# === Config generale (niente flag da toccare a mano) ===
splits = [("Train", "train"), ("Validation", "val")]  # (nome cartella reale, nome out)
base_dir = "datasets/data/FGNET/images"
embeddings_root = "embeddings_fgnet"

image_size = (224, 224)
num_patches_x, num_patches_y = 6, 6
threshold_distance = 0.88
tau_random_walk = 0.87

# se =0 non crea LRC, e sotto faremo fallback su RW
num_heads = 8

# ResGCN
num_layers = 4
device = "mps" if torch.backends.mps.is_available() else "cpu"

for split_name, split_out in splits:
    print(f"\n🔁 Elaborazione split: {split_name}")

    patches_dir = os.path.join(base_dir, split_name, "patches")
    images_dir = os.path.join(base_dir, split_name, "images_preprocessed")

    if not os.path.isdir(images_dir):
        print(f"⚠️ Cartella immagini mancante: {images_dir}")
        continue

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for image_filename in image_files:
        image_id = os.path.splitext(image_filename)[0]
        csv_path = os.path.join(patches_dir, f"{image_id}_patches.csv")
        image_path = os.path.join(images_dir, image_filename)
        embedding_dir = os.path.join(embeddings_root, split_out, image_id)
        os.makedirs(embedding_dir, exist_ok=True)

        if not os.path.exists(csv_path):
            print(f"⚠️  Salto {image_id}: CSV patch mancante ({csv_path})")
            continue
        if not os.path.exists(image_path):
            print(f"⚠️  Salto {image_id}: immagine mancante ({image_path})")
            continue

        print(f"\n📂 Processing {split_out}/{image_id}...")

        # === Step 1: costruzione grafi iniziale e RW
        try:
            result = construct_initial_graph(
                csv_path=csv_path,
                image_path=image_path,
                image_size=image_size,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
                threshold_distance=threshold_distance,
                tau_random_walk=tau_random_walk,
                embedding_dir=embedding_dir,
            )
        except Exception as e:
            print(f"❌ Errore in construct_initial_graph per {image_id}: {e}")
            continue

        if result is None:
            print(f"❌ Skippata {image_id} per errore nelle patch.")
            continue

        graph_initial, graph_augmented, patches_tensor, patch_to_node = result

        torch.save(graph_initial, os.path.join(embedding_dir, "graph_initial.pt"))
        torch.save(graph_augmented, os.path.join(embedding_dir, "graph_rw.pt"))
        print("💾 Grafi iniziale e RW salvati.")

        # === Step 2: LRC (se num_heads > 0)
        X = graph_augmented.x
        if num_heads and num_heads > 0:
            try:
                lrc = LatentRelationCapturer(in_dim=X.shape[1], num_heads=num_heads)
                A_m_list = lrc(X)
                for idx, A_m in enumerate(A_m_list):
                    edge_index_m, edge_weight_m = dense_to_sparse(A_m)
                    graph_m = Data(x=X, edge_index=edge_index_m, edge_attr=edge_weight_m)
                    torch.save(graph_m, os.path.join(embedding_dir, f"graph_lrc_{idx}.pt"))
                print(f"💾 Grafi LRC ({num_heads} teste) salvati.")
            except Exception as e:
                print(f"⚠️ Errore durante LRC su {image_id}: {e}. Procedo senza LRC.")

        # === Step 3: Deep Feature Extraction (auto: usa LRC se c’è, altrimenti RW)
        try:
            graph_paths = sorted(glob.glob(os.path.join(embedding_dir, "graph_lrc_*.pt")))
            if not graph_paths:
                rw_path = os.path.join(embedding_dir, "graph_rw.pt")
                if not os.path.exists(rw_path):
                    print(f"⚠️ Nessun grafo disponibile per {image_id}")
                    continue
                graph_paths = [rw_path]

            deep_feats = extract_deep_features(
                graph_paths,
                feature_dim=X.shape[1],
                num_layers=num_layers,
                device=device,
            )
            torch.save(deep_feats, os.path.join(embedding_dir, "deep_features.pt"))
            print("💾 Deep features salvate.")
        except Exception as e:
            print(f"❌ Errore estrazione deep features per {image_id}: {e}")
            continue

print("\n✅ Tutte le immagini di Train e Validation sono state elaborate.")