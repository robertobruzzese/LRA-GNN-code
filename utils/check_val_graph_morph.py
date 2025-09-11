import os
import torch

val_dir = "embeddings_morph/val"
total = 0
missing_graphs = 0
empty_graphs = 0
valid_graphs = 0

for fname in os.listdir(val_dir):
    folder_path = os.path.join(val_dir, fname)
    if not os.path.isdir(folder_path):
        continue

    total += 1
    graph_path = os.path.join(folder_path, "graph_rw.pt")

    if not os.path.exists(graph_path):
        print(f"❌ Manca: {fname}")
        missing_graphs += 1
        continue

    try:
        graph = torch.load(graph_path)
        if graph.edge_index.size(0) == 2 and graph.edge_index.size(1) > 0:
            valid_graphs += 1
        else:
            print(f"⚠️ Vuoto: {fname}")
            empty_graphs += 1
    except Exception as e:
        print(f"❌ Errore caricamento {fname}: {e}")
        missing_graphs += 1

print("\n📊 RISULTATO FINALE:")
print(f"Totale cartelle in val/:        {total}")
print(f"Grafi VALIDi (non vuoti):       {valid_graphs}")
print(f"Grafi VUOTI (0 archi):          {empty_graphs}")
print(f"Grafi MANCANTI o corrotti:      {missing_graphs}")