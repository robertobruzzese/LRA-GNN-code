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
                features_list = model(batch, return_features=True)  # list of [N, hidden_dim]
                
                # 🔸 Prende la media degli 8 feature matrix
                if isinstance(features_list, list):
                    stacked = torch.stack([f.mean(dim=0) for f in features_list], dim=0)  # [8, hidden_dim]
                    features = stacked.mean(dim=0, keepdim=True)  # [1, hidden_dim]
                else:
                    features = features_list.mean(dim=0, keepdim=True)

                y = batch[0].y
                image_name = batch[0].image_name

            else:  # 🔹 GCN-only o DFE-only
                batch = batch.to(device)
                features = model(batch, return_features=True)  # [num_nodes, hidden_dim]
                
                # 🔒 Usa global_mean_pool se c'è batch info
                if hasattr(batch, "batch") and features.size(0) > 1 and batch.batch.max().item() < features.size(0):
                    features = global_mean_pool(features, batch.batch)  # [1, hidden_dim]
                else:
                    features = features.mean(dim=0, keepdim=True)  # fallback

                y = batch.y
                image_name = batch.image_name

            # 🧪 Salva embedding
            emb = features[0].cpu()
            age = y.item()
            torch.save(
                {"embedding": emb, "age": age},
                os.path.join(save_dir, f"{image_name}.pt")
            )

    print(f"✅ Embedding salvati in {save_dir}")