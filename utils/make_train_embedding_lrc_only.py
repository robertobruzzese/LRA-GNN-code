import os
import re
import torch
import argparse
from tqdm import tqdm
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.lra_gnn import LRA_GNN

@torch.no_grad()
def main(val_root, checkpoint, dataset="MORPH", overwrite=False, device=None, image_to_age=None):
    device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    image_to_age = image_to_age or {}
    # Modello in modalità LRC-only
    model = LRA_GNN(
        num_layers=12,
        num_heads=8,
        in_channels=512,
        hidden_channels=512,
        out_channels=1,
        enable_lrc=True,
        enable_dfe=False
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Elenco sottocartelle (una per immagine)
    if not os.path.isdir(val_root):
        raise FileNotFoundError(f"Cartella val non trovata: {val_root}")

    subdirs = [d for d in os.listdir(val_root) if os.path.isdir(os.path.join(val_root, d))]
    subdirs.sort()
    print(f"🔍 Trovate {len(subdirs)} sottocartelle in {val_root}")

    num_done, num_skipped, num_missing_heads = 0, 0, 0

    for sid in tqdm(subdirs, desc="🔄 Generazione embedding (LRC-only)"):
        folder = os.path.join(val_root, sid)
        out_pt = os.path.join(val_root, f"{sid}.pt")

        if (not overwrite) and os.path.exists(out_pt):
            num_skipped += 1
            continue

        # Carica gli 8 grafi LRC
        graphs = []
        ok = True
        for i in range(8):
            gpath = os.path.join(folder, f"graph_lrc_{i}.pt")
            if not os.path.exists(gpath):
                ok = False
                break
            graphs.append(torch.load(gpath).to(device))
        if not ok:
            num_missing_heads += 1
            continue

        # Forward: il modello deve accettare una lista di grafi e restituire una lista di feature per head
        features_list = model(graphs, return_features=True)  # ogni item: [num_nodes, hidden_dim]

        # Pool per-head (mean sui nodi), poi media tra le 8 head
        if isinstance(features_list, list):
            per_head = [f.mean(dim=0) for f in features_list]   # [hidden_dim]
            embedding = torch.stack(per_head, dim=0).mean(dim=0)  # [hidden_dim]
        else:
            # fallback: se il modello restituisce un unico tensore [num_nodes, hidden_dim]
            embedding = features_list.mean(dim=0)

        # Ricava l'età dal nome (usando il CSV solo per CLAP2016)
        if dataset.upper() == "CLAP2016":
            image_name = sid + ".jpg"
            age = float(image_to_age.get(image_name, -1.0))
        elif dataset.upper() == "MORPH":
            m = re.search(r'[FfMm](\d{1,3})', sid)
            age = float(m.group(1)) if m else -1.0
        elif dataset.upper() == "FGNET":
            m = re.search(r'[Aa](\d{1,3})', sid)
            age = float(m.group(1)) if m else -1.0
        elif dataset.upper() == "UTKFACE":
            try:
                age = float(sid.split('_')[0])
            except Exception:
                age = -1.0
        else:
            age = -1.0

        if age < 0:
            print(f"⚠️  Età non trovata per {sid}, salto.")
            continue

        torch.save({"embedding": embedding.cpu(), "age": age}, out_pt)
        num_done += 1

    print(f"✅ Fatto: creati {num_done} embedding; saltati {num_skipped} (già presenti); cartelle con head mancanti: {num_missing_heads}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True, help="Es. embeddings_ablation_morph_lrc_no_dfe/train")
    ap.add_argument("--checkpoint", required=True, help="Path al best LRA-GNN per LRC-only")
    ap.add_argument("--dataset", default="MORPH")
    ap.add_argument("--overwrite", action="store_true", help="Rigenera anche se .pt esiste")
    args = ap.parse_args()
 

    # Caricamento età dal CSV per CLAP2016
    # Caricamento età dal CSV per CLAP2016
    image_to_age = {}
    if args.dataset.upper() == "CLAP2016":
        embedding_split = os.path.basename(args.train_root.strip("/")).lower()
        csv_path = os.path.join("datasets", "data", "CLAP2016", f"CLAP_complete_{embedding_split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV età non trovato: {csv_path}")
        csv_df = pd.read_csv(csv_path)
        image_to_age = dict(zip(csv_df["image"], csv_df["mean"]))

    main(args.train_root, args.checkpoint, dataset=args.dataset, overwrite=args.overwrite, image_to_age=image_to_age)

   