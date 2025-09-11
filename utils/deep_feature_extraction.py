import torch
from models.resgcn import DeepResGCN

def normalize_adjacency(A):
    I = torch.eye(A.size(0), device=A.device)
    A_hat = A + I
    D = torch.diag(A_hat.sum(dim=1))
    D_inv_sqrt = torch.linalg.inv(torch.sqrt(D))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

def extract_deep_features(graph_paths, feature_dim, num_layers, device='cpu'):
    model = DeepResGCN(dim=feature_dim, num_layers=num_layers).to(device)
    deep_features = []

    # Supporta anche caso DFE-only con singolo grafo
    if isinstance(graph_paths, str):
        graph_paths = [graph_paths]

    for path in graph_paths:
        graph = torch.load(path, map_location=device)
        X = graph.x.to(device)

        A_dense = torch.zeros((X.size(0), X.size(0)), device=device)
        A_dense[graph.edge_index[0], graph.edge_index[1]] = 1.0
        A_hat = normalize_adjacency(A_dense)

        H = model(A_hat, X)
        pooled = H.mean(dim=0)  # media dei nodi
        deep_features.append(pooled)

    return torch.stack(deep_features)  # shape: [N, dim]
