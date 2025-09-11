import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentRelationCapturer(nn.Module):
    def __init__(self, in_dim, d_k=64, num_heads=12):
        super(LatentRelationCapturer, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.sqrt_dk = d_k ** 0.5
        self.W_q = nn.ModuleList([nn.Linear(in_dim, d_k) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(in_dim, d_k) for _ in range(num_heads)])

    def forward(self, X, A_0=None):  # A_0 non più necessario
        A_m_list = []
        for i in range(self.num_heads):
            Q = self.W_q[i](X)
            K = self.W_k[i](X)
            scores = torch.matmul(Q, K.T) / self.sqrt_dk
            A_m = F.softmax(scores, dim=-1)
            A_m_list.append(A_m)
        return A_m_list

