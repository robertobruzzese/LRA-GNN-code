import torch

classifier_path = "checkpoints_ablation/utkface/prlae_no_lrc_dfe/classifier.pth"
checkpoint = torch.load(classifier_path, map_location="cpu")

print("\n🔍 Chiavi presenti nel file classifier.pth:\n")
for k in checkpoint.keys():
    print(f" - {k}")

print("\n📐 Dimensioni dei tensori:\n")
for k, v in checkpoint.items():
    print(f"{k}: {v.shape}")