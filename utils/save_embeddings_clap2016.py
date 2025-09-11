from torch_geometric.nn import global_mean_pool
import os
import torch
from tqdm import tqdm

def save_embeddings(model, dataloader, device, save_dir="embeddings/"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔄 Generazione embedding"):
            if isinstance(batch, list):  # 🔹 LRC: lista di 8 grafi
                batch = [g.to(device) for g in batch]
                features = model(batch, return_features=True)  # [8, hidden_dim]
                if isinstance(features, list):
                    features = torch.stack(features, dim=0).mean(dim=0)  # media sulle 8 heads
                features = features.unsqueeze(0)  # [1, hidden_dim]
                y = batch[0].y
                image_names = [batch[0].image_name] if hasattr(batch[0], "image_name") else batch[0].image_names
            else:  # 🔹 GCN-only o DFE-only
                batch = batch.to(device)
                features = model(batch, return_features=True)  # [num_nodes, hidden_dim]

                # 🔒 Usa global_mean_pool solo se batch è coerente
                if hasattr(batch, "batch") and features.size(0) > 1 and batch.batch.max().item() < features.size(0):
                    features = global_mean_pool(features, batch.batch)  # [1, hidden_dim]
                else:
                    features = features.mean(dim=0, keepdim=True)  # fallback sicuro

                y = batch.y
                image_names = [batch.image_name] if hasattr(batch, "image_name") else batch.image_names

            # 🧪 Compatibilità y
            if y.ndim == 0:
                y = y.unsqueeze(0)

            # 🔁 Salva embedding per ogni campione
            for i in range(features.size(0)):
                emb = features[i].cpu()
                age = y[i].item()
                image_name = image_names[i]
                if isinstance(image_name, list):
                    image_name = image_name[0]
                elif isinstance(image_name, torch.Tensor):
                    image_name = image_name.item() if image_name.numel() == 1 else str(image_name.tolist())

                torch.save(
                    {"embedding": emb, "age": age},
                    os.path.join(save_dir, f"{image_name}.pt")
                )

    print(f"✅ Embedding salvati in {save_dir}")