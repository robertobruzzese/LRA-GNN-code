# models/lra_gnn_lrc_dfe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from models.lrc_gat import LatentRelationCapturerGAT, DeepResidualGCN  # riusa le tue classi

class LRA_GNN_LRC_DFE(nn.Module):
    """
    Variante del modello per l'ablation LRC+DFE:
    - accetta sia un singolo Data che una list[Data] (es. 8 LRC + 1 DFE);
    - mette in sicurezza batch/edge_index per ogni grafo;
    - usa lo stesso backbone (LatentRelationCapturerGAT + DeepResidualGCN).
    """
    def __init__(self, in_channels=512, hidden_channels=512, num_heads=8, num_layers=12,
                 out_channels=1, enable_lrc=True, enable_dfe=True):
        super().__init__()

        self.enable_lrc = enable_lrc
        self.enable_dfe = enable_dfe

        # Pre-proiezione per il caso DFE-only (resta compatibile)
        self.fc_pre = nn.Linear(in_channels, hidden_channels) if (self.enable_dfe and not self.enable_lrc) else nn.Identity()

        # LRC (attenzione)
        if self.enable_lrc:
            self.attention_out_dim = 128
            self.attention = LatentRelationCapturerGAT(
                in_channels=in_channels,
                out_channels=self.attention_out_dim,
                num_heads=num_heads
            )
            self.att_proj = nn.Linear(self.attention_out_dim, hidden_channels)
        else:
            self.attention_out_dim = in_channels
            self.attention = None
            self.att_proj = nn.Identity()

        # GNN profonda se LRC o DFE è attivo
        if self.enable_dfe or self.enable_lrc:
            self.res_gcn = DeepResidualGCN(dim=hidden_channels, num_layers=num_layers)
        else:
            self.res_gcn = None

        self.fc = nn.Linear(
            hidden_channels if (self.enable_dfe or self.enable_lrc) else in_channels,
            out_channels
        )

    def _prep_graph(self, g):
        """Rende sicuro un singolo Data: batch presente, edge_index long (o vuoto)."""
        # batch
        if not hasattr(g, "batch") or g.batch is None:
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=g.x.device)

        # edge_index
        ei = getattr(g, "edge_index", None)
        if (ei is None) or (not torch.is_tensor(ei)) or (ei.numel() == 0) or (ei.dim() != 2) or (ei.size(0) != 2):
            ei = torch.empty((2, 0), dtype=torch.long, device=g.x.device)  # grafico senza archi
        else:
            ei = ei.long()

        return g.x, ei, g.batch

    def _run_single_graph(self, x_in, edge_index, batch):
        """Esegue il ramo LRC/DFE su un singolo grafo e restituisce il pooled feature vector."""
        # Multi-head LRC (o identity)
        head_outputs = self.attention(x_in, edge_index) if self.enable_lrc else [x_in]

        head_feats = []
        for h in head_outputs:
            h = self.att_proj(h)
            if self.enable_dfe or self.enable_lrc:
                # costruisci A densa e normalizza
                A_dense = torch.zeros((h.size(0), h.size(0)), device=h.device)
                if edge_index.numel() > 0:
                    A_dense[edge_index[0], edge_index[1]] = 1.0
                A_hat = DeepResidualGCN.normalize_adjacency(A_dense)
                H = self.res_gcn(A_hat, h)
            else:
                H = F.relu(self.fc_pre(h))

            pooled_h = global_mean_pool(H, batch)
            head_feats.append(pooled_h)

        # media sulle head
        return torch.stack(head_feats, dim=0).mean(dim=0)

    def forward(self, graph, return_features=False):
        # Caso: lista di grafi (es. 8 LRC + 1 DFE)
        if isinstance(graph, list):
            pooled_graphs = []
            for g in graph:
                x_in, edge_index, batch = self._prep_graph(g)
                pooled_graphs.append(self._run_single_graph(x_in, edge_index, batch))
            x = torch.stack(pooled_graphs, dim=0).mean(dim=0)  # media tra grafi
            return x if return_features else self.fc(x).squeeze(-1)

        # Caso: singolo grafo
        x_in, edge_index, batch = self._prep_graph(graph)
        x = self._run_single_graph(x_in, edge_index, batch)
        return x if return_features else self.fc(x).squeeze(-1)