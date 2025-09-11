import os
import torch
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool

def save_embeddings(model, dataloader, device, save_dir="embeddings/"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔄 Generazione embedding"):
            if isinstance(batch, list):  # LRC attivo
                batch = [g.to(device) for g in batch]
                features = model(batch, return_features=True)
                if isinstance(features, list):
                    features = torch.stack(features, dim=0).mean(dim=0)
                features = features.unsqueeze(0)
                y = batch[0].y
                image_names = [batch[0].image_name] if hasattr(batch[0], "image_name") else batch[0].image_names
            else:  # GCN / DFE
                batch = batch.to(device)
                features = model(batch, return_features=True)
                if hasattr(batch, "batch") and features.size(0) > 1 and batch.batch.max().item() < features.size(0):
                    features = global_mean_pool(features, batch.batch)
                else:
                    features = features.mean(dim=0, keepdim=True)

                y = batch.y
                image_names = [batch.image_name] if hasattr(batch, "image_name") else batch.image_names

            if y.ndim == 0:
                y = y.unsqueeze(0)

            for i in range(features.size(0)):
                emb = features[i].cpu()
                age = y[i].item()
                image_name = image_names[i]
                file_name = f"{image_name}_utk.pt"

                torch.save(
                    {"embedding": emb, "age": age},
                    os.path.join(save_dir, file_name)
                )

    print(f"✅ Embedding UTKFace salvati in {save_dir}")