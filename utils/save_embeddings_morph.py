from torch_geometric.nn import global_mean_pool
import os
import torch
from tqdm import tqdm

def save_embeddings(model, dataloader, device, save_dir="embeddings/"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔄 Generazione embedding"):
            if isinstance(batch, list):  
                # 🔹 LRC: lista di 8 grafi
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

            else: 
                 # 🔹 GCN-only o DFE-only
 #              🔹 GCN-only o DFE-only
                batch = batch.to(device)
                features = model(batch, return_features=True)  # [num_nodes, hidden_dim]

                if hasattr(batch, "batch") and features.size(0) > 1 and batch.batch.max().item() < features.size(0):
                    features = global_mean_pool(features, batch.batch)  # [batch_size, hidden_dim]
                else:
                    features = features.mean(dim=0, keepdim=True)  # [1, hidden_dim]

                y = batch.y
                image_name = batch.image_name if hasattr(batch, "image_name") else batch.image_names


            if y.ndim == 0:
                y = y.unsqueeze(0)
             #🧠 Assicura compatibilità image_names
            if isinstance(image_name, str):
                image_names = [image_name]
            elif isinstance(image_name, list):
                image_names = image_name
            elif isinstance(image_name, torch.Tensor):
                image_names = [str(image_name.item())]
            else:
                raise TypeError(f"Tipo non supportato per image_name: {type(image_name)}")

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