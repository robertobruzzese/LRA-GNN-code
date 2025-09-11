import torch
d = torch.load("embeddings_ablation_morph_lrc_no_dfe/val/163547_00M36.pt")
print(d.keys())  # controlla se c'è "age"
print(d["age"])  # età dell'immagine