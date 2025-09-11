import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentRelationCapturerGAT(nn.Module):
    """
    LRC Module con GATConv (trainabile) per multi-head attention.
    """
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(LatentRelationCapturerGAT, self).__init__()
        self.attentions = nn.ModuleList([
            GATConv(in_channels, out_channels, heads=1, concat=False) 
            for _ in range(num_heads)
        ])
    
    def forward(self, x, edge_index):
        """
        Args:
            x          : Tensor [N, in_channels] - embedding nodi
            edge_index : Tensor [2, E] - edge list (può essere fully connected)
        
        Returns:
            outputs : lista di [N, out_channels] tensor da ogni testa
        """
        edge_index = edge_index.long()
        return [att(x, edge_index) for att in self.attentions]

        
    
class LatentRelationCapturer(nn.Module):
    """
    Versione non trainabile: calcolo esplicito di M fully connected adjacency matrices (Eq.3).
    """
    def __init__(self, in_dim, num_heads=8, d_k=64):
        super().__init__()
        self.num_heads = num_heads
        self.W_q = nn.ModuleList([nn.Linear(in_dim, d_k) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(in_dim, d_k) for _ in range(num_heads)])
    
    def forward(self, X):
        A_list = []
        for i in range(self.num_heads):
            Q = self.W_q[i](X)  # [N, d_k]
            K = self.W_k[i](X)  # [N, d_k]
            scores = torch.matmul(Q, K.T) / (Q.shape[1] ** 0.5)  # [N, N]
            A_m = torch.softmax(scores, dim=1)  # [N, N]
            A_list.append(A_m)
        return A_list


class DeepResidualGCN(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, A_hat, X):
        H = X
        for i in range(self.num_layers):
            H_res = H
            H = A_hat @ H  # GCN-like propagation
            H = self.layers[i](H)
            H = F.relu(H + H_res)  # Residual connection + non-linearity
        return H

    @staticmethod
    def normalize_adjacency(A):
        I = torch.eye(A.size(0), device=A.device)
        A_hat = A + I
        D = torch.diag(A_hat.sum(dim=1))
        D_inv_sqrt = torch.linalg.inv(torch.sqrt(D))
        return D_inv_sqrt @ A_hat @ D_inv_sqrt

    @staticmethod
    def extract_from_graph_paths(graph_paths, feature_dim, num_layers, device='cpu'):
        model = DeepResidualGCN(dim=feature_dim, num_layers=num_layers).to(device)
        deep_features = []

        for path in graph_paths:
            graph = torch.load(path, map_location=device)
            X = graph.x.to(device)
            A_dense = torch.zeros((X.size(0), X.size(0)), device=device)
            A_dense[graph.edge_index[0], graph.edge_index[1]] = 1.0
            A_hat = DeepResidualGCN.normalize_adjacency(A_dense)

            H = model(A_hat, X)
            pooled = H.mean(dim=0)
            deep_features.append(pooled)

        return torch.stack(deep_features)

