import torch
import os
from utils.graph_constructor_morph import visualize_graph_diff

embedding_dir = "embeddings/01_0M32"
graph_initial = torch.load(os.path.join(embedding_dir, "graph_initial.pt"))
graph_augmented = torch.load(os.path.join(embedding_dir, "graph_rw.pt"))
patches_tensor = torch.load(os.path.join(embedding_dir, "patches_tensor.pt"))
patch_to_node = torch.load(os.path.join(embedding_dir, "patch_to_node.pt"))

visualize_graph_diff(
    graph_initial,
    graph_augmented,
    patch_to_node,
    patches_tensor,
    save_path="output/graph_diff_output.png"
)