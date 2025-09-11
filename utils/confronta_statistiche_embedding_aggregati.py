import os
import torch
import numpy as np
from tqdm import tqdm

def carica_embeddings_da_cartella(flat_dir):
    embeddings = []
    for fname in tqdm(os.listdir(flat_dir), desc=f"Caricamento da {flat_dir}"):
        if not fname.endswith(".pt"):
            continue
        fpath = os.path.join(flat_dir, fname)
        # Escludi sottocartelle
        if os.path.isdir(fpath):
            continue
        try:
            data = torch.load(fpath)
            if isinstance(data, dict) and "embedding" in data:
                embeddings.append(data["embedding"].float())
        except Exception as e:
            print(f"Errore caricando {fname}: {e}")
    if not embeddings:
        print(f"❌ Nessun embedding trovato per {flat_dir.split('/')[-1]}")
        return None
    return torch.stack(embeddings)

def stampa_statistiche(nome, embeddings):
    if embeddings is None or embeddings.numel() == 0:
        print(f"📊 {nome.upper()} - Nessun embedding valido.")
        return
    mean = embeddings.mean().item()
    std = embeddings.std().item()
    print(f"📊 {nome.upper()} - Mean: {mean:.4f}, Std: {std:.4f}")

if __name__ == "__main__":
    train_dir = "embeddings_ablation_utkface_lrc_no_dfe/train"
    val_dir = "embeddings_ablation_utkface_lrc_no_dfe/val"

    train_embeddings = carica_embeddings_da_cartella(train_dir)
    val_embeddings = carica_embeddings_da_cartella(val_dir)

    stampa_statistiche("train", train_embeddings)
    stampa_statistiche("val", val_embeddings)