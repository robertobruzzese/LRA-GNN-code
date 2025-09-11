import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.graph_constructor_morph import visualize_graph, visualize_graph_topology
import torch

# === Config ===
embedding_dir = "embeddings/01_0M32"  # cambialo al nome immagine che vuoi
graph_path = os.path.join(embedding_dir, "graph_initial.pt")
patches_path = os.path.join(embedding_dir, "patches_tensor.pt")
patch_to_node_path = os.path.join(embedding_dir, "patch_to_node.pt")

# === Carica grafo e metadati
graph = torch.load(graph_path)
patches_tensor = torch.load(patches_path)
patch_to_node = torch.load(patch_to_node_path)

# === Visualizza
visualize_graph(graph, patches_tensor, patch_to_node,save_path='output/graph_output')
visualize_graph_topology(graph, patch_to_node)