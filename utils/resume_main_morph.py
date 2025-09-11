import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from graph_constructor_morph import construct_initial_graph
from lrc import LatentRelationCapturer
from deep_feature_extraction import extract_deep_features
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import glob
#Attenzione questo script si lancia quando main_morph va in crash
#sostituendo val con train e Train con Validation questo script si adatta 
#al caso di resume del crash per entrambi le cartelle
# === CONFIG ===
base_dir = "datasets/data/MORPH/images/Validation"
images_dir = os.path.join(base_dir, "images_preprocessed")
patches_dir = os.path.join(base_dir, "patches")
embedding_root = "embeddings_morph/val"
image_size = (224, 224)
num_patches_x, num_patches_y = 6, 6
threshold_distance = 0.88
tau_random_walk = 0.87
num_heads = 12
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
num_layers = 4
REQUIRED_FILES = [
    "graph_initial.pt",
    "graph_rw.pt",
    "patches_tensor.pt",
    "patch_to_node.pt",
    "deep_features.pt"
]

def is_embedding_complete(embedding_dir):
    return all(os.path.exists(os.path.join(embedding_dir, f)) for f in REQUIRED_FILES)

def process_image(image_filename):
    image_name = os.path.splitext(image_filename)[0]
    image_path = os.path.join(images_dir, image_filename)
    csv_path = os.path.join(patches_dir, f"{image_name}_patches.csv")
    embedding_dir = os.path.join(embedding_root, image_name)
    os.makedirs(embedding_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"⚠️  Skipping {image_name}: missing CSV ({csv_path})")
        return

    print(f"\n🚀 Processing: {image_name}")

    # Step 1: graph initial + RW
    graph_initial, graph_augmented, patches_tensor, patch_to_node = construct_initial_graph(
        csv_path=csv_path,
        image_path=image_path,
        image_size=image_size,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        threshold_distance=threshold_distance,
        tau_random_walk=tau_random_walk,
        embedding_dir=embedding_dir
    )

    torch.save(graph_initial, os.path.join(embedding_dir, "graph_initial.pt"))
    torch.save(graph_augmented, os.path.join(embedding_dir, "graph_rw.pt"))
    torch.save(patches_tensor, os.path.join(embedding_dir, "patches_tensor.pt"))
    torch.save(patch_to_node, os.path.join(embedding_dir, "patch_to_node.pt"))

    # Step 2: LRC
    X = graph_augmented.x
    lrc = LatentRelationCapturer(in_dim=X.shape[1], num_heads=num_heads)
    A_m_list = lrc(X)

    for idx, A_m in enumerate(A_m_list):
        edge_index_m, edge_weight_m = dense_to_sparse(A_m)
        graph_m = Data(x=X, edge_index=edge_index_m, edge_attr=edge_weight_m)
        torch.save(graph_m, os.path.join(embedding_dir, f"graph_lrc_{idx}.pt"))

    # Step 3: ResGCN (Deep Features)
    graph_paths = sorted(glob.glob(os.path.join(embedding_dir, "graph_lrc_*.pt")))
    deep_feats = extract_deep_features(graph_paths, feature_dim=X.shape[1], num_layers=num_layers, device=device)
    torch.save(deep_feats, os.path.join(embedding_dir, "deep_features.pt"))

    print(f"✅ {image_name} done.\n")

def main():
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_images = len(image_files)
    completed = 0
    for image_filename in image_files:
        image_name = os.path.splitext(image_filename)[0]
        embedding_dir = os.path.join(embedding_root, image_name)

        if is_embedding_complete(embedding_dir):
            completed += 1
            continue

        process_image(image_filename)

    print(f"\n📊 Completati: {completed} su {total_images}")

if __name__ == "__main__":
    main()