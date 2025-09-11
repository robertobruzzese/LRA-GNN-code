import os
import re
import torch
import argparse
from tqdm import tqdm
import pandas as pd  # ✅ Serve per leggere il CSV

@torch.no_grad()
def main(val_root, checkpoint=None, dataset="MORPH", overwrite=False, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    if checkpoint:
        print("⚠️ Attenzione: checkpoint specificato ma non usato in DFE-only")

    # Carica età dal CSV se dataset = CLAP2016
    image_to_age = {}
    if dataset.upper() == "CLAP2016":
        embedding_split = os.path.basename(val_root.strip("/")).lower()  # es. "val"
        csv_path = os.path.join("datasets", "data", "CLAP2016", f"CLAP_complete_{embedding_split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV età non trovato: {csv_path}")
        csv_df = pd.read_csv(csv_path)
        image_to_age = dict(zip(csv_df["image"], csv_df["mean"]))

    # Elenco sottocartelle
    if not os.path.isdir(val_root):
        raise FileNotFoundError(f"Cartella val non trovata: {val_root}")

    subdirs = [d for d in os.listdir(val_root) if os.path.isdir(os.path.join(val_root, d))]
    subdirs.sort()
    print(f"🔍 Trovate {len(subdirs)} sottocartelle in {val_root}")

    num_done, num_skipped, num_missing = 0, 0, 0

    for sid in tqdm(subdirs, desc="🔄 Generazione embedding (DFE-only)"):
        folder = os.path.join(val_root, sid)
        out_pt = os.path.join(val_root, f"{sid}.pt")

        if (not overwrite) and os.path.exists(out_pt):
            num_skipped += 1
            continue

        deep_path = os.path.join(folder, "deep_features.pt")
        if not os.path.exists(deep_path):
            num_missing += 1
            continue

        deep_data = torch.load(deep_path, map_location=device)

        if isinstance(deep_data, dict) and "features" in deep_data:
            features = deep_data["features"]
        elif isinstance(deep_data, torch.Tensor):
            features = deep_data
        else:
            print(f"❌ deep_features.pt malformato in {folder}, salto.")
            continue

        embedding = features.mean(dim=0)

        # Estrai età
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

    print(f"✅ Fatto: creati {num_done} embedding; saltati {num_skipped} (già presenti); cartelle senza deep_features.pt: {num_missing}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True, help="Es. embeddings_ablation_morph_no_lrc_dfe/train")
    ap.add_argument("--checkpoint", help="Non usato in DFE-only, lasciarlo vuoto", default=None)
    ap.add_argument("--dataset", default="MORPH")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    main(args.train_root, args.checkpoint, dataset=args.dataset, overwrite=args.overwrite)