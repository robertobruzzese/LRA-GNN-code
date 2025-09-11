# models/prl_helpers.py
import torch

def build_state(xi, env, device):
    """
    xi:  tensor [1, D]  (embedding)
    env: RLEnvironment con env.r, env.c (int)
    """
    pos = torch.tensor([[env.r/10.0, env.c/10.0]], device=device)  # [1, 2]
    s = torch.cat([xi, pos], dim=1)  # [1, D+2]
    return s