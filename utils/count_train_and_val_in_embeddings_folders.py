import os
import torch

def conta_grafi_validi(cartella):
    total = 0
    validi = 0
    vuoti = 0
    mancanti = 0

    for nome in os.listdir(cartella):
        path_grafo = os.path.join(cartella, nome, "graph_rw.pt")
        total += 1
        if not os.path.exists(path_grafo):
            mancanti += 1
            continue
        try:
            data = torch.load(path_grafo)
            if data.edge_index.size(1) > 0:
                validi += 1
            else:
                vuoti += 1
        except Exception as e:
            print(f"Errore su {nome}: {e}")
            mancanti += 1

    print(f"\n📁 Cartella: {cartella}")
    print(f"Totale cartelle trovate:        {total}")
    print(f"Grafi VALIDI (edge_index > 0):  {validi}")
    print(f"Grafi VUOTI (0 archi):          {vuoti}")
    print(f"Grafi MANCANTI o corrotti:      {mancanti}")

# Esegui sui due split
conta_grafi_validi("embeddings_morph/train")
conta_grafi_validi("embeddings_morph/val")