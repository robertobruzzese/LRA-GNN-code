import torch

# Sostituisci con il percorso corretto del file che ti ha dato errore
path = "embeddings_CLAP2016/train/002672/graph_rw.pt"

# Carica il file .pt
data = torch.load(path)

# Stampa il contenuto
print(data)

# Se vuoi vedere se esiste 'y'
if hasattr(data, 'y'):
    print(f"\n✅ Età (y): {data.y.item()}")
else:
    print("\n❌ Attributo 'y' non presente nel grafo.")