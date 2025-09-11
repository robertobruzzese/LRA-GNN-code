import torch

file = "embeddings_ablation_morph_lrc_no_dfe/val/0007_00M16.pt"  # 👈 Cambia nome file se serve
data = torch.load(file)

print(data.keys())                     # Devono esserci solo 'embedding', 'age'
print(data['embedding'].shape)        # Es: torch.Size([512]) o torch.Size([134])
print(data['age'])                    # Età float o int