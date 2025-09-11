import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from collections import defaultdict
import random

def random_walk_update_adjacency(edge_index, node_features, p=1, q=1, num_walks=10, walk_length=20, tau=0.9):
    """
    edge_index: [2, num_edges] tensor
    node_features: [N, M] tensor (embedding dei nodi)
    p, q: parametri del random walk
    tau: soglia per cos(x_i, x_j)
    """
    # 1. Costruisci il grafo con NetworkX
    G = nx.Graph()
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    # Aggiungi tutti i nodi esplicitamente, anche quelli isolati
    num_nodes = node_features.size(0)
    G.add_nodes_from(range(num_nodes))

    
    # 2. Prepara la nuova matrice di adiacenza
    N = node_features.size(0)
    updated_adj = torch.zeros(N, N)

    def get_transition_probs(prev, curr):
        probs = []
        neighbors = list(G.neighbors(curr))
        for neighbor in neighbors:
            if neighbor == prev:
                prob = 1 / p
            elif G.has_edge(prev, neighbor):
                prob = 1
            else:
                prob = 1 / q
            probs.append(prob)
        return neighbors, probs

    # 3. Random walks
    for start_node in range(N):
        for _ in range(num_walks):
            walk = [start_node]
            if len(list(G.neighbors(start_node))) == 0:
                continue
            curr = start_node
            prev = -1
            for _ in range(walk_length):
                neighbors, probs = get_transition_probs(prev, curr)
                if not neighbors:
                    break
                probs = np.array(probs)
                probs /= probs.sum()
                next_node = np.random.choice(neighbors, p=probs)
                walk.append(next_node)
                prev, curr = curr, next_node

 

            for i in range(len(walk)):
                for j in range(i + 1, len(walk)):
                    a, b = walk[i], walk[j]
                    if a == b:
                        continue
                    sim = F.cosine_similarity(node_features[a], node_features[b], dim=0)

                    # 🔹 Log della similarità tra nodi nel cammino
                    print(f"🔍 Walk {start_node}: sim({a}, {b}) = {sim.item():.4f}", end='')

                    if sim >= tau:
                        updated_adj[a, b] = 1
                        updated_adj[b, a] = 1
                        print(" ✅ [AGGIUNTO]")
                    else:
                        print(" ❌")


    # 5. Somma alla A originale
    orig_adj = torch.zeros(N, N)
    for a, b in edges:
        orig_adj[a, b] = 1
        orig_adj[b, a] = 1

    final_adj = orig_adj + updated_adj
    return final_adj
