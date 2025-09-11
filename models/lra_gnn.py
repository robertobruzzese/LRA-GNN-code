import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from models.lrc_gat import LatentRelationCapturerGAT, DeepResidualGCN

class LRA_GNN(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=512, num_heads=8, num_layers=12,
                 out_channels=1, enable_lrc=True, enable_dfe=True):
        super().__init__()

        self.enable_lrc = enable_lrc
        self.enable_dfe = enable_dfe

        # 🔸 Adatta dimensione input
        if self.enable_dfe and not self.enable_lrc:
            self.fc_pre = nn.Linear(in_channels, hidden_channels)  # DFE only
        else:
            self.fc_pre = nn.Identity()

        if self.enable_lrc:
            self.attention_out_dim = 128
            self.attention = LatentRelationCapturerGAT(
                in_channels=in_channels,
                out_channels=self.attention_out_dim,
                num_heads=num_heads
            )
        else:
            self.attention_out_dim = in_channels

        if self.enable_lrc:
            self.att_proj = nn.Linear(self.attention_out_dim, hidden_channels)
        else:
            self.att_proj = nn.Identity()

        if self.enable_dfe or self.enable_lrc:
            self.res_gcn = DeepResidualGCN(dim=hidden_channels, num_layers=num_layers)

        self.fc = nn.Linear(
            hidden_channels if (self.enable_dfe or self.enable_lrc) else in_channels,
            out_channels
        )

    def forward(self, graph, return_features=False):
        if not hasattr(graph, "batch") or graph.batch is None:
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)
        if isinstance(graph, list):  # Caso LRC + DFE con lista di grafi
            pooled = []
            for g in graph:
                x, edge_index, batch = g.x, g.edge_index.long(), g.batch
                head_outputs = self.attention(x, edge_index)

                head_features = []
                for h in head_outputs:
                    h = self.att_proj(h)
                    A_dense = torch.zeros((h.size(0), h.size(0)), device=h.device)
                    A_dense[edge_index[0], edge_index[1]] = 1.0
                    A_hat = DeepResidualGCN.normalize_adjacency(A_dense)
                    H = self.res_gcn(A_hat, h)
                    pooled_h = global_mean_pool(H, batch)
                    head_features.append(pooled_h)

                pooled.append(torch.stack(head_features, dim=0).mean(dim=0))

            x = torch.stack(pooled, dim=0).mean(dim=0)

        elif hasattr(graph, 'edge_index') and graph.edge_index is not None and \
             hasattr(graph, 'batch') and graph.batch is not None:
            x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch

            if self.enable_lrc:
                head_outputs = self.attention(x, edge_index)
            else:
                head_outputs = [x]

            deep_features = []
            for head_x in head_outputs:
                head_x = self.att_proj(head_x)
                if self.enable_dfe or self.enable_lrc:
                    A_dense = torch.zeros((head_x.size(0), head_x.size(0)), device=head_x.device)
                    A_dense[edge_index[0], edge_index[1]] = 1.0
                    A_hat = DeepResidualGCN.normalize_adjacency(A_dense)
                    H = self.res_gcn(A_hat, head_x)
                else:
                    H = F.relu(self.fc_pre(head_x))

                pooled = global_mean_pool(H, batch)
                deep_features.append(pooled)

            x = torch.stack(deep_features, dim=0).mean(dim=0)

        else:  # Caso GCN-only o DFE-only
            x = graph.x
            if x.ndim == 1:
                x = x.unsqueeze(0)
            x = F.relu(self.fc_pre(x))
            batch = graph.batch if hasattr(graph, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = global_mean_pool(x, batch)

        return x if return_features else self.fc(x).squeeze(-1)