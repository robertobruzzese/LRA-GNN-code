import torch
from torch_geometric.utils import to_dense_adj
import os

def check_graph_connectivity(graph_path):
    # === Carica il grafo PyG
    graph = torch.load(graph_path)
    num_nodes = graph.num_nodes
    num_edges = graph.edge_index.size(1)

    print(f"\n📊 Checking graph: {graph_path}")
    print(f"🔢 Nodes: {num_nodes}")
    print(f"🔗 Edges: {num_edges}")

    # === Calcola la matrice di adiacenza densa
    adj = to_dense_adj(graph.edge_index)[0]  # [N, N]

    # === Conta gli archi massimi teorici (senza self-loop)
    max_edges = num_nodes * (num_nodes - 1)
    print(f"📈 Max possible edges (no self-loop): {max_edges}")

    # === Verifica se tutti i nodi sono connessi tra loro (tranne se stessi)
    fully_connected = ((adj - torch.eye(num_nodes)) > 0).all().item()
    print(f"✅ Fully connected (excluding self-loops)? {fully_connected}")

    # === Elenco degli archi mancanti
    missing = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and adj[i, j] == 0:
                missing.append((i, j))

    print(f"❌ Missing edges: {len(missing)}")
    if missing:
        print("🔍 First few missing:", missing[:10])


if __name__ == "__main__":
    # 🔧 Sostituisci con il tuo path al grafo LRC
    graph_file = "embeddings/01_0M32/graph_lrc_0.pt"
    if os.path.exists(graph_file):
        check_graph_connectivity(graph_file)
    else:
        print(f"❌ File not found: {graph_file}")