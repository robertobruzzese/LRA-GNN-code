import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 📂 Carica le deep features
deep_feats = torch.load("embeddings/01_0M32_deep_features.pt")  # cambia nome se serve
deep_feats_np = deep_feats.detach().cpu().numpy()  # [12, 512]
print("👉 Differenze tra vettori:")
for i in range(len(deep_feats)):
    for j in range(i+1, len(deep_feats)):
        dist = torch.norm(deep_feats[i] - deep_feats[j]).item()
        print(f"Dist({i},{j}) = {dist:.6f}")

# 📉 PCA (2D)
pca = PCA(n_components=2)
deep_feats_pca = pca.fit_transform(deep_feats_np)

# 📉 t-SNE (2D)
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
deep_feats_tsne = tsne.fit_transform(deep_feats_np)

# 🎨 Plot PCA
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(deep_feats_pca[:, 0], deep_feats_pca[:, 1], c='blue')
plt.title("PCA delle Deep Features")
plt.xlabel("PC1")
plt.ylabel("PC2")

# 🎨 Plot t-SNE
plt.subplot(1, 2, 2)
plt.scatter(deep_feats_tsne[:, 0], deep_feats_tsne[:, 1], c='green')
plt.title("t-SNE delle Deep Features")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

plt.tight_layout()
plt.show()
