import torch
d = torch.load("embeddings_ablation_morph_no_lrc_dfe/val/0007_00M16.pt")
print(d.keys())
print(type(d['embedding']), d['embedding'].shape)
print(type(d['age']), d['age'])