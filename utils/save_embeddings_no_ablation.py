# utils/save_embeddings.py
import os
import torch
from tqdm import tqdm

def save_embeddings(model, dataloader, device, save_dir="embeddings/"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔄 Generazione embedding"):
            batch = batch.to(device)
            embeddings = model(batch, return_features=True)

            for i in range(embeddings.size(0)):
                emb = embeddings[i].cpu()
                age = batch.y[i].item()
                torch.save({"embedding": emb, "age": age}, os.path.join(save_dir, f"sample_{idx}.pt"))
                idx += 1

    print(f"✅ Embedding salvati in {save_dir} ({idx} file totali)")