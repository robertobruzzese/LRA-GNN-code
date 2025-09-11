import os
import torch
from collections import defaultdict

# 📁 Directory base
embedding_base_dir = "embeddings_ablation_morph_lrc_no_dfe"
splits = ["train", "val"]

# 📊 Inizializza risultati
results = {
    split: {
        "total_embeddings": 0,
        "dims": defaultdict(int),
        "invalid_embeddings": [],
        "wrong_dim_embeddings": [],
        "dim_128_embeddings": [],
        "total_graphs": 0,
        "wrong_graphs": [],
        "invalid_graphs": [],
    }
    for split in splits
}

# 🔍 Loop su split
for split in splits:
    embedding_root = os.path.join(embedding_base_dir, split)
    print(f"\n📂 Analisi split: {split}")

    for subdir, _, files in os.walk(embedding_root):
        for file in files:
            path = os.path.join(subdir, file)

            # 🔸 Controlla EMBEDDING files
            if file.endswith(".pt") and not file.startswith("graph_lrc_"):
                results[split]["total_embeddings"] += 1
                try:
                    data = torch.load(path, map_location='cpu')
                    if isinstance(data, dict) and 'embedding' in data:
                        tensor = data['embedding']
                    elif isinstance(data, torch.Tensor):
                        tensor = data
                    else:
                        results[split]["invalid_embeddings"].append(path)
                        continue

                    dim = tensor.shape[-1]
                    results[split]["dims"][dim] += 1
                    if dim != 512:
                        results[split]["wrong_dim_embeddings"].append(path)
                        if dim == 128:
                            results[split]["dim_128_embeddings"].append(path)
                except Exception as e:
                    results[split]["invalid_embeddings"].append(path)

            # 🔸 Controlla GRAFI LRC
            elif file.startswith("graph_lrc_") and file.endswith(".pt"):
                results[split]["total_graphs"] += 1
                try:
                    graph = torch.load(path, map_location='cpu')
                    if hasattr(graph, 'x'):
                        if graph.x.shape[1] != 512:
                            results[split]["wrong_graphs"].append(path)
                    else:
                        results[split]["invalid_graphs"].append(path)
                except Exception:
                    results[split]["invalid_graphs"].append(path)

# 📊 Stampa i risultati
for split in splits:
    print(f"\n📊 RISULTATI PER SPLIT: {split.upper()}")

    # Embedding
    print(f"📦 Totale embedding analizzati: {results[split]['total_embeddings']}")
    for dim, count in sorted(results[split]["dims"].items()):
        status = "✅ CORRETTO" if dim == 512 else "❌ ERRATO"
        print(f"- {count} embedding con dimensione {dim} → {status}")
    print(f"✅ Totali corretti (dim=512): {results[split]['dims'].get(512, 0)}")
    print(f"❌ Totali errati (dim≠512): {len(results[split]['wrong_dim_embeddings'])}")
    print(f"❌ File con dimensione 128 (errati): {len(results[split]['dim_128_embeddings'])}")
    print(f"🚫 File embedding non validi (caricamento fallito): {len(results[split]['invalid_embeddings'])}")

    # Graph LRC
    print(f"\n🧩 Totale graph_lrc_*.pt analizzati: {results[split]['total_graphs']}")
    print(f"✅ Graph con x.shape[1] == 512: {results[split]['total_graphs'] - len(results[split]['wrong_graphs']) - len(results[split]['invalid_graphs'])}")
    print(f"❌ Graph con x.shape[1] ≠ 512: {len(results[split]['wrong_graphs'])}")
    print(f"🚫 Graph LRC non validi (errore o x mancante): {len(results[split]['invalid_graphs'])}")

    if results[split]["wrong_graphs"]:
        print(f"\n❌ Elenco grafi LRC con dimensione errata in {split}:")
        for path in results[split]["wrong_graphs"]:
            print(f"- {path}")