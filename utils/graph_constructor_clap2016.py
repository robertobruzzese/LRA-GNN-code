import os
import csv
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_networkx
from torchvision.models import resnet18, ResNet18_Weights

from utils.random_walk_update_adjacency import random_walk_update_adjacency

# ======================================================================================
# 🔹 RESNET18 ENCODER (senza classificatore)
# ======================================================================================
weights = ResNet18_Weights.DEFAULT
resnet_encoder = resnet18(weights=weights)
resnet_encoder.fc = nn.Identity()
resnet_encoder.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ======================================================================================
# 🔹 LOGGING UTILITY
# ======================================================================================
def log_patch_failure(image_path: str, reason: str, log_path: str = "logs/patch_failures.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"[{image_path}] ❌ {reason}\n")

def log_empty_graph(image_name: str, log_path: str = "logs/empty_graphs.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{image_name}\n")
def log_initial_graph(image_path: str, msg: str, log_path: str = "logs/initial_graph.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"[{image_path}] {msg}\n")
# ======================================================================================
# 🔹 GLOBALE: Immagini disconnesse da ignorare
# ======================================================================================
BAD_IMAGES = {
    "001847", "000024", "001513", "004012", "002507", "002213",
    "004017", "005595", "006725", "003836", "005948", "003093"
}
# ======================================================================================
# 🔹 UTILITY
# ======================================================================================
@torch.no_grad()
def extract_patch_features(image: np.ndarray, patch_coords: Dict[int, Tuple[int, int, int, int]], image_path: str) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    for pid, (x0, y0, x1, y1) in patch_coords.items():
        patch = image[y0:y1, x0:x1]
        try:
            if patch.shape[0] == 0 or patch.shape[1] == 0:
                raise ValueError(f"Patch {pid} vuota (shape={patch.shape})")

            input_tensor = transform(patch).unsqueeze(0)
            feat = resnet_encoder(input_tensor).squeeze(0)
            feats.append(feat)

        except Exception as e:
            log_patch_failure(image_path=image_path, reason=f"Errore patch {pid}: {str(e)}")
            raise e
    return torch.stack(feats)

def load_patch_keypoints_from_csv(csv_path: str) -> Dict[int, Any]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV non trovato: {csv_path}")
    patch_kp: Dict[int, Any] = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            pid = int(row[0])
            kps = [tuple(map(float, kp.split(","))) for kp in row[1].split(";")]
            patch_kp[pid] = kps
    return patch_kp

# ======================================================================================
# 🔹 GRAF COSTRUCTION
# ======================================================================================
def construct_initial_graph(
    csv_path: str,
    image_path: str,
    image_size: Tuple[int, int],
    num_patches_x: int = 6,
    num_patches_y: int = 6,
    threshold_distance: float = 0.9,
    tau_random_walk: Optional[float] = None,
    embedding_dir: str = "embeddings_utkface",
):
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(embedding_dir, exist_ok=True)
    # 🛑 Salta se i file di embedding sono già presenti (grafo già creato)
    graph_dir = embedding_dir
    required_files = ["graph_initial.pt", "graph_rw.pt", "node_features.pt"]
    if all(os.path.exists(os.path.join(graph_dir, f)) for f in required_files):
        print(f"✅ {img_name}: già processata, salto.")
        return None

    
    if img_name in BAD_IMAGES:
        print(f"🚫 Ignoro immagine disconnessa: {img_name}")
        return None

    emb_nodes_path = os.path.join(embedding_dir, "node_features.pt")
    emb_edges_path = os.path.join(embedding_dir, "edge_features.pt")

    patch_kps = load_patch_keypoints_from_csv(csv_path)
    patch_ids = sorted(patch_kps.keys())
    patch_to_node = {pid: i for i, pid in enumerate(patch_ids)}

    pw, ph = image_size[0] // num_patches_x, image_size[1] // num_patches_y
    patch_centers = [((pid % num_patches_x) * pw + pw // 2,
                      (pid // num_patches_x) * ph + ph // 2) for pid in patch_ids]
    patches_tensor = torch.tensor(patch_centers, dtype=torch.float32)

    patch_coords = {
        pid: (
            (pid % num_patches_x) * pw,
            (pid // num_patches_x) * ph,
            (pid % num_patches_x) * pw + pw,
            (pid // num_patches_x) * ph + ph,
        )
        for pid in patch_ids
    }

    if os.path.exists(emb_nodes_path):
        print(f"📥 Carico node embedding da {emb_nodes_path}")
        node_features = torch.load(emb_nodes_path)
    else:
        print(f"📤 Estraggo embedding e salvo in {emb_nodes_path}")
        img = np.array(Image.open(image_path).convert("RGB"))
        try:
            node_features = extract_patch_features(img, patch_coords, image_path)
            torch.save(node_features, emb_nodes_path)
        except Exception as e:
            log_patch_failure(image_path, f"Impossibile estrarre patch: {str(e)}")
            return None

    e_idx, e_w, e_emb = [], [], []
    for i in range(len(patch_ids)):
        for j in range(i + 1, len(patch_ids)):
            fi, fj = node_features[i], node_features[j]
            sim = F.cosine_similarity(fi, fj, dim=0).item()
            if sim > threshold_distance:
                e_idx += [[i, j], [j, i]]
                e_w += [sim, sim]
                e_emb += [(fi + fj) / 2, (fj + fi) / 2]

    edge_index = torch.tensor(e_idx, dtype=torch.long).t() if e_idx else torch.empty((2, 0), dtype=torch.long)
    edge_attr  = torch.tensor(e_w, dtype=torch.float32).unsqueeze(1) if e_w else torch.empty((0, 1))
    edge_emb   = torch.stack(e_emb) if e_emb else torch.empty((0, 512))

    edge_index, edge_attr = coalesce(edge_index, edge_attr)

    graph_initial = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    msg = f"Inizializzato grafo: nodi={node_features.size(0)}, archi_init={edge_index.size(1)}"
    print(f"⚠️ {msg}")
    log_initial_graph(image_path=image_path, msg=msg)
    print(f"⚠️ Inizializzato grafo per {img_name}: "
      f"nodi={node_features.size(0)}, archi_init={edge_index.size(1)}")
    print(f"📊 {img_name} – Nodi: {node_features.size(0)} – Archi iniziali: {edge_index.size(1)}")
    tau = tau_random_walk or threshold_distance

    torch.save(edge_emb, emb_edges_path)

    adj_upd = random_walk_update_adjacency(
        edge_index=edge_index,
        node_features=node_features,
        p=4, q=0.25,
        num_walks=10, walk_length=20,
        tau=tau,
    )
    new_eidx = adj_upd.nonzero(as_tuple=False).t()
    graph_rw = Data(x=node_features, edge_index=new_eidx, edge_attr=None)
    graph_rw.edge_index = connect_isolated_nodes(graph_rw, node_features, tau)
    if graph_rw.edge_index.dim() < 2 or graph_rw.edge_index.size(1) == 0:
        print(f"❌ {img_name}: grafo RW vuoto.")
        log_empty_graph(img_name)  # funzione già suggerita
        return None
    else:
        print(f"✅ {img_name}: grafo RW con {graph_rw.edge_index.size(1)} archi")

    torch.save(graph_initial, os.path.join(embedding_dir, "graph_initial.pt"))
    torch.save(graph_rw,       os.path.join(embedding_dir, "graph_rw.pt"))

    n_new = len(set(map(tuple, graph_rw.edge_index.t().tolist())) - set(map(tuple, edge_index.t().tolist())))
    print(f"\n📌 {img_name}: nodi={node_features.size(0)}, archi_iniziali={edge_index.size(1)}, nuovi={n_new}\n")

    return graph_initial, graph_rw, patches_tensor, patch_to_node

# ======================================================================================
# 🔹 CONNECT ISOLATED NODES
# ======================================================================================
def connect_isolated_nodes(graph: Data, node_features: torch.Tensor, tau: float):
    N = node_features.size(0)
    connected = set(graph.edge_index.flatten().tolist())
    eidx = graph.edge_index.t().tolist()

    for i in range(N):
        if i in connected:
            continue
        best_j, best_sim = -1, -1.0
        for j in range(N):
            if i == j:
                continue
            s = F.cosine_similarity(node_features[i], node_features[j], dim=0).item()
            if s >= tau and s > best_sim:
                best_j, best_sim = j, s
        if best_j != -1:
            eidx += [[i, best_j], [best_j, i]]
            print(f"🔗 Nodo {i} connesso a {best_j} (sim={best_sim:.3f})")
    return torch.tensor(eidx, dtype=torch.long).t()

# ======================================================================================
# 🔹 VISUALIZATION (facoltativa)
# ======================================================================================
try:
    import matplotlib.pyplot as plt
    import networkx as nx
except ImportError:
    plt, nx = None, None


def _require_vis():
    if plt is None or nx is None:
        raise ImportError("Installare matplotlib e networkx per la visualizzazione.")


def visualize_graph(graph: Data, patches_tensor: torch.Tensor, patch_to_node: Dict[int, int], highlight_edges=None, save_path: Optional[str] = None):
    _require_vis()
    G = to_networkx(graph, to_undirected=False)
    h = 224
    pos = {patch_to_node[pid]: (patches_tensor[i, 0].item(), h - patches_tensor[i, 1].item()) for i, pid in enumerate(patch_to_node.keys())}

    plt.figure(figsize=(6, 6))
    hl_set = set(map(tuple, map(sorted, highlight_edges))) if highlight_edges else set()
    new_e, norm_e, seen = [], [], set()
    for u, v in G.edges():
        k = tuple(sorted((u, v)))
        if k in seen: continue
        seen.add(k)
        (new_e if k in hl_set else norm_e).append((u, v))

    nx.draw_networkx_edges(G, pos, edgelist=norm_e, edge_color="blue", width=1.0, alpha=0.6, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edges(G, pos, edgelist=new_e,  edge_color="red",  width=1.5, alpha=0.9, style="dashed", connectionstyle='arc3,rad=0.25')
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="red")
    nx.draw_networkx_labels(G, pos, labels={patch_to_node[pid]: pid for pid in patch_to_node}, font_size=9, font_weight="bold")
    plt.title("📌 Grafo Patch (rossi = nuovi)")
    plt.axis('off')
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✅ Salvato {save_path}")
    plt.show()


def visualize_graph_topology(graph: Data, patch_to_node: Optional[Dict[int, int]] = None, highlight_edges=None, spread_factor: float = 1.5):
    _require_vis()
    G = to_networkx(graph, to_undirected=True)
    pos = nx.spring_layout(G, seed=42, k=spread_factor / (G.number_of_nodes() ** 0.5), iterations=100)

    plt.figure(figsize=(8, 8))
    new_e = set(map(tuple, map(sorted, highlight_edges))) if highlight_edges else set()
    norm_e = [e for e in G.edges if tuple(sorted(e)) not in new_e]

    nx.draw_networkx_edges(G, pos, edgelist=norm_e, edge_color="lightgray", width=1.2)
    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=list(new_e), edge_color="red", style="dashed", width=2.0)
    nx.draw_networkx_nodes(G, pos, node_color="orange", node_size=600, edgecolors="black", linewidths=0.8)

    if patch_to_node:
        node_to_patch = {v: k for k, v in patch_to_node.items()}
        labels = {n: str(node_to_patch.get(n, n)) for n in G.nodes}
    else:
        labels = {n: str(n) for n in G.nodes}

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="white", font_weight="bold")
    plt.title("📌 Topologia Grafo")
    plt.axis('off')
    plt.show()


def visualize_graph_diff(graph_initial: Data, graph_augmented: Data, patch_to_node: Dict[int, int], patches_tensor: torch.Tensor, save_path: Optional[str] = None):
    _require_vis()
    G_i = to_networkx(graph_initial, to_undirected=False)
    G_a = to_networkx(graph_augmented, to_undirected=False)
    new_e = set(map(tuple, map(sorted, G_a.edges()))) - set(map(tuple, map(sorted, G_i.edges())))

    h = 224
    pos = {patch_to_node[pid]: (patches_tensor[i, 0].item(), h - patches_tensor[i, 1].item()) for i, pid in enumerate(patch_to_node.keys())}

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G_a, pos, node_color="red", node_size=300)
    nx.draw_networkx_edges(G_a, pos, edgelist=list(map(tuple, map(sorted, G_i.edges()))), edge_color="blue", width=1.0, alpha=0.6, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(G_a, pos, edgelist=list(new_e), edge_color="red", style="dashed", width=1.5, alpha=0.9, connectionstyle="arc3,rad=0.25")
    nx.draw_networkx_labels(G_a, pos, labels={patch_to_node[pid]: pid for pid in patch_to_node}, font_size=10, font_weight="bold", font_color="white")
    plt.title("📌 Differenze Grafo")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Salvato {save_path}")
    else:
        plt.show()