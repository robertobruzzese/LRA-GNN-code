import torch
x = torch.load("embeddings_ablation_morph_no_lrc_dfe/train/061002_7F46.pt")
print(x.keys())  # dovresti vedere: dict_keys(['embedding', 'age'])