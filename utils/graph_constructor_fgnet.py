import torch
import os
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, coalesce
from utils.random_walk_update_adjacency import random_walk_update_adjacency 
import torchvision.transforms as T
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx


# 🔹 Encoder CNN (es. ResNet18 senza classificatore)
resnet_encoder = resnet18(pretrained=True)
resnet_encoder.fc = nn.Identity()
resnet_encoder.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_patch_features(image, patch_coords):
    features = []
    for _, (x_start, y_start, x_end, y_end) in patch_coords.items():
        patch = image[y_start:y_end, x_start:x_end]  # estrai
        input_tensor = transform(patch).unsqueeze(0)  # (1, 3, 224, 224)
        feat = resnet_encoder(input_tensor).squeeze(0)  # (512,)
        features.append(feat)
    return torch.stack(features)

def load_patch_keypoints_from_csv(csv_path):
    patch_keypoints = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ ERRORE: File CSV non trovato {csv_path}")

    with open(csv_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            patch_id = int(row[0])
            keypoints = [tuple(map(float, kp.split(","))) for kp in row[1].split(";")]
            patch_keypoints[patch_id] = keypoints
    return patch_keypoints

def construct_initial_graph(csv_path, image_path, image_size, num_patches_x=6, num_patches_y=6, threshold_distance=0.9, tau_random_walk=None, embedding_dir="embeddings_FGNET"):

    embedding_subdir = embedding_dir  # Usa direttamente embedding_dir passato dal main
    os.makedirs(embedding_subdir, exist_ok=True)


    #os.makedirs("embeddings_FGNET", exist_ok=True)

    #image_name = os.path.splitext(os.path.basename(image_path))[0]
    #embedding_subdir = os.path.join("embeddings_FGNET", image_name)
    #os.makedirs(embedding_subdir, exist_ok=True)
    #emb_path = os.path.join(embedding_subdir, "node_features.pt")
    #edge_path = os.path.join(embedding_subdir, "edge_features.pt")
    # Usa direttamente la directory passata dal main
    embedding_subdir = embedding_dir
    os.makedirs(embedding_subdir, exist_ok=True)

    emb_path = os.path.join(embedding_subdir, "node_features.pt")
    edge_path = os.path.join(embedding_subdir, "edge_features.pt")


    patch_keypoints = load_patch_keypoints_from_csv(csv_path)

    patch_ids = sorted(patch_keypoints.keys())
    patch_to_node = {patch_id: idx for idx, patch_id in enumerate(patch_ids)}

    patch_w, patch_h = image_size[0] // num_patches_x, image_size[1] // num_patches_y
    patches = {
        patch_id: (
            (patch_id % num_patches_x) * patch_w + patch_w // 2,
            (patch_id // num_patches_x) * patch_h + patch_h // 2
        )
        for patch_id in patch_ids
    }
    patches_tensor = torch.tensor([patches[patch_id] for patch_id in patch_ids], dtype=torch.float32)

    patch_coords = {
        patch_id: (
            (patch_id % num_patches_x) * patch_w,
            (patch_id // num_patches_x) * patch_h,
            (patch_id % num_patches_x) * patch_w + patch_w,
            (patch_id // num_patches_x) * patch_h + patch_h
        ) for patch_id in patch_ids
    }

    if os.path.exists(emb_path):
        print(f"📥 Caricamento degli embedding dei nodi da {emb_path}")
        node_features = torch.load(emb_path)
    else:
        print(f"📤 Estrazione degli embedding dei nodi e salvataggio in {emb_path}")
        image = np.array(Image.open(image_path).convert("RGB"))
        node_features = extract_patch_features(image, patch_coords)
        torch.save(node_features, emb_path)

    edge_index = []
    edge_weights = []
    edge_embeddings = []

    for i in range(len(patch_ids)):
        for j in range(i + 1, len(patch_ids)):
            patch_i, patch_j = patch_ids[i], patch_ids[j]
            feat_i = node_features[patch_to_node[patch_i]]
            feat_j = node_features[patch_to_node[patch_j]]

            sim = torch.nn.functional.cosine_similarity(feat_i, feat_j, dim=0).item()

            if sim > threshold_distance:
                edge_index.append([patch_to_node[patch_i], patch_to_node[patch_j]])
                edge_index.append([patch_to_node[patch_j], patch_to_node[patch_i]])
                edge_weights.append(sim)
                edge_weights.append(sim)
                edge_embeddings.append((feat_i + feat_j) / 2)
                edge_embeddings.append((feat_j + feat_i) / 2)

    edge_index = torch.tensor(edge_index, dtype=torch.long).T if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1) if edge_weights else torch.empty((0, 1))
    edge_embeddings = torch.stack(edge_embeddings) if edge_embeddings else torch.empty((0, 512))

    edge_index, edge_weights = coalesce(edge_index, edge_weights)

    # 🔹 1. Crea grafo iniziale
    graph_initial = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)
    tau_rw = tau_random_walk if tau_random_walk is not None else threshold_distance

    torch.save(node_features, emb_path)
    torch.save(edge_embeddings, edge_path)
    
    threshold_distance = 0.5

    # 🔁 2. Applica il random walk per arricchire il grafo
    adj_updated = random_walk_update_adjacency(
        edge_index=graph_initial.edge_index,
        node_features=graph_initial.x,
        p=4, q=0.25,
        num_walks=10,
        walk_length=20,
        tau=tau_rw  # 👈 soglia specifica per RW
    )


    print(f"\n📌 Grafo creato con {graph_initial.x.shape[0]} nodi e {graph_initial.edge_index.shape[1]} archi")
    print(f"📊 Feature dei nodi (prime 5 righe):\n{graph_initial.x[:5]}")
    print(f"🔗 Edge Index (prime 5 connessioni):\n{graph_initial.edge_index[:, :5]}")
    print(f"📏 Edge Attr (similarità, prime 5):\n{graph_initial.edge_attr[:5]}")

    new_edge_index = adj_updated.nonzero(as_tuple=False).t()

    # 🔹 3. Crea grafo arricchito
    #graph_augmented = Data(x=node_features, edge_index=new_edge_index, edge_attr=None)

    # 🔹 3. Crea grafo arricchito
    graph_augmented = Data(x=node_features, edge_index=new_edge_index, edge_attr=None)

    # 🔹 4. Collega i nodi isolati dopo il random walk
    graph_augmented.edge_index = connect_isolated_nodes(graph_augmented, node_features, tau_rw)




    torch.save(graph_initial, os.path.join(embedding_subdir, "graph_initial.pt"))
    torch.save(graph_augmented, os.path.join(embedding_subdir, "graph_rw.pt"))


    # 🔍 Confronto archi prima e dopo il random walk
    initial_edges_set = set([tuple(edge) for edge in graph_initial.edge_index.t().tolist()])
    augmented_edges_set = set([tuple(edge) for edge in graph_augmented.edge_index.t().tolist()])

    new_edges = augmented_edges_set - initial_edges_set

    print(f"\n🔎 Nuovi archi trovati dal random walk: {len(new_edges)}")
    if len(new_edges) > 0:
        print("➡️ Esempi di nuovi archi aggiunti:")
        for i, edge in enumerate(list(new_edges)[:5]):
            print(f"  {edge}")
    else:
        print("⚠️ Nessun nuovo arco è stato aggiunto. Verifica tau o parametri del walk.")


     # 🔙 4. Restituisci entrambi
    return graph_initial, graph_augmented, patches_tensor, patch_to_node


def connect_isolated_nodes(graph, node_features, tau):
    N = node_features.size(0)
    connected = set(graph.edge_index.flatten().tolist())
    edge_index_list = graph.edge_index.t().tolist()

    for i in range(N):
        if i not in connected:
            best_sim = -1
            best_j = -1
            for j in range(N):
                if i == j:
                    continue
                sim = F.cosine_similarity(node_features[i], node_features[j], dim=0)
                if sim >= tau and sim > best_sim:
                    best_sim = sim
                    best_j = j
            if best_j != -1:
                edge_index_list.append([i, best_j])
                edge_index_list.append([best_j, i])
                print(f"🔗 Nodo isolato {i} connesso a {best_j} (sim={best_sim:.3f})")

    return torch.tensor(edge_index_list).T


def visualize_graph(graph, patches_tensor, patch_to_node, highlight_edges=None,save_path=None):
    import matplotlib.pyplot as plt
    import networkx as nx

    # ✅ Usa grafo direzionale per abilitare curvatura
    G = to_networkx(graph, to_undirected=False)

    # ↔️ Posizione nodi in base alle patch
    image_height = 224  # o qualunque sia la dimensione verticale reale dell’immagine
    pos = {
        patch_to_node[patch_id]: (
            patches_tensor[idx, 0].item(),
            image_height - patches_tensor[idx, 1].item()  # inverti l’asse Y
        )
        for idx, patch_id in enumerate(patch_to_node.keys())
    }


    plt.figure(figsize=(6, 6))

    # 🔍 Set ordinato per evitare duplicati
    highlight_set = set(tuple(sorted(e)) for e in highlight_edges) if highlight_edges else set()

    # 🔎 Filtra archi evitando doppioni
    seen_edges = set()
    real_new_edges = []
    real_normal_edges = []
    for u, v in G.edges():
        key = tuple(sorted((u, v)))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        if key in highlight_set:
            real_new_edges.append((u, v))
        else:
            real_normal_edges.append((u, v))

    # 🔵 Archi originali
    nx.draw_networkx_edges(G, pos,
        edgelist=real_normal_edges,
        edge_color="blue",
        width=1.0,
        alpha=0.6,
        connectionstyle='arc3,rad=0.1'
    )

    # 🔴 Archi nuovi (tratteggiati)
    nx.draw_networkx_edges(G, pos,
        edgelist=real_new_edges,
        edge_color="red",
        style="dashed",
        width=1.5,
        alpha=0.9,
        connectionstyle='arc3,rad=0.25'
    )

    # 🔘 Nodi
    nx.draw_networkx_nodes(G, pos,
        node_size=150,
        node_color="red"
    )

    # 🏷️ Etichette nodi (patch ID)
    nx.draw_networkx_labels(G, pos,
        labels={patch_to_node[patch_id]: patch_id for patch_id in patch_to_node.keys()},
        font_size=9,
        font_weight='bold'
    )

    # 🔄 Mappa inversa: nodo → patch
    node_to_patch = {v: k for k, v in patch_to_node.items()}

    # 🏷️ Etichette archi
    ax = plt.gca()
    for i, (u, v) in enumerate(seen_edges):
        patch_u = node_to_patch.get(u, u)
        patch_v = node_to_patch.get(v, v)
        label = f"e({patch_u}–{patch_v})"

        # Punto medio
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2

        # Offset personalizzati
        if (patch_u, patch_v) == (28, 14) or (patch_v, patch_u) == (28, 14):
            dx, dy = 10.0, 0.0
        elif (patch_u, patch_v) == (22, 20) or (patch_v, patch_u) == (22, 20):
            dx, dy = -10.0, 0.0
        else:
            dx = 2.0 if i % 2 == 0 else -2.0
            dy = 2.0 if i % 2 == 0 else -2.0

        ax.text(xm + dx, ym + dy, label,
                fontsize=8,
                color='gray',
                ha='center',
                va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))

    plt.title("📌 Grafo con ID originali delle Patch (archi nuovi tratteggiati)")
    plt.axis('off')
    plt.tight_layout()


    if save_path:
        plt.savefig(f"{save_path}.jpg", format='jpg', dpi=300)
        print(f"✅ Figura salvata in {save_path}.jpg")
    plt.show()
    plt.close()



def visualize_graph_topology(graph, patch_to_node=None, highlight_edges=None, spread_factor=1.5):

    G = to_networkx(graph, to_undirected=True)

    # Layout "forza fisica" con attrazione più forte
    pos = nx.spring_layout(G, seed=42, k=spread_factor / (G.number_of_nodes() ** 0.5), iterations=100)

    plt.figure(figsize=(8, 8))

    edge_list = list(G.edges)
    new_edges = set()
    normal_edges = []

    if highlight_edges:
        new_edges = set([tuple(sorted(e)) for e in highlight_edges])
        for e in edge_list:
            e_sorted = tuple(sorted(e))
            if e_sorted in new_edges:
                continue
            normal_edges.append(e)
    else:
        normal_edges = edge_list

    nx.draw_networkx_edges(G, pos,
        edgelist=normal_edges,
        edge_color="lightgray",
        width=1.2
    )

    if highlight_edges:
        nx.draw_networkx_edges(G, pos,
            edgelist=list(new_edges),
            edge_color="red",
            style="dashed",
            width=2.0
        )

    nx.draw_networkx_nodes(G, pos,
        node_color="orange",
        node_size=600,
        edgecolors="black",
        linewidths=0.8
    )

    # Etichette corrette con patch ID
    if patch_to_node:
        node_to_patch = {v: k for k, v in patch_to_node.items()}
        labels = {node: str(node_to_patch.get(node, node)) for node in G.nodes}
    else:
        labels = {node: str(node) for node in G.nodes}

    nx.draw_networkx_labels(G, pos,
        labels=labels,
        font_size=10,
        font_color="white",
        font_weight="bold"
    )

    plt.title("📌 Topologia del grafo (archi nuovi tratteggiati)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

