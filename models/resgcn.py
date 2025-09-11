import torch
import torch.nn as nn
import torch.nn.functional as F

class ResGCNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, A_hat, H_l, H_l_prev, H_0):
        H_init_res = (1 - self.beta) * (A_hat @ H_l) + self.beta * H_0
        H_dyn_res = (1 - self.alpha) * self.W(A_hat @ H_l) + self.alpha * H_l_prev
        return F.relu(H_init_res + H_dyn_res)

class DeepResGCN(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([ResGCNLayer(dim) for _ in range(num_layers)])

    def forward(self, A_hat, H_0):
        H_l = H_0
        H_l_prev = torch.zeros_like(H_0)
        for layer in self.layers:
            H_next = layer(A_hat, H_l, H_l_prev, H_0)
            H_l_prev = H_l
            H_l = H_next
        return H_l
