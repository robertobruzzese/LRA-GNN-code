#!/usr/bin/env python3
#check_embeddings_full.py
import os, sys, glob, argparse, torch

def expected_heads(dataset: str) -> int:
    ds = dataset.upper()
    # full pipeline: 12 head per MORPH/FGNET, 8 per UTKFACE/CLAP2016
    return 8 if ds in ("UTKFACE", "CLAP2016") else 12

def check_sample(sample_dir: str, min_lrc: int, check_shape: bool, max_lrc_to_check: int = 8):
    issues = []

    # --- presenza file principali ---
    gi_path = os.path.join(sample_dir, "graph_initial.pt")
    rw_path = os.path.join(sample_dir, "graph_rw.pt")

    has_gi = os.path.exists(gi_path)
    has_rw = os.path.exists(rw_path)

    # deep_features: accetta sia from_rw che standard
    feat_path = None
    for fn in ("deep_features_from_rw.pt", "deep_features.pt"):
        p = os.path.join(sample_dir, fn)
        if os.path.exists(p):
            feat_path = p
            break
    has_feat = feat_path is not None

    # conta LRC (non richiede esattamente N fissi, ma >= min_lrc)
    lrc_paths = sorted(glob.glob(os.path.join(sample_dir, "graph_lrc_*.pt")))
    lrc = len(lrc_paths)

    if lrc < min_lrc or not has_rw or not has_feat:
        # stile di output compatibile con i tuoi log precedenti
        issues.append(f"LRC={lrc}/{min_lrc}, DFE_graph={int(has_rw)}/1, DFE_feats={int(has_feat)}/1")
    if not has_gi:
        issues.append("missing graph_initial.pt")

    # --- controlli di forma opzionali ---
    if check_shape and not issues and feat_path is not None and has_rw:
        try:
            x = torch.load(feat_path, map_location="cpu")
            if not isinstance(x, torch.Tensor) or x.ndim != 2:
                issues.append(f"deep_features not 2D (got {type(x)} shape={getattr(x,'shape',None)})")
            else:
                N, F = x.shape
                g_rw = torch.load(rw_path, map_location="cpu")
                ei = getattr(g_rw, "edge_index", None)
                if torch.is_tensor(ei) and ei.numel() > 0:
                    ei = ei.long()
                    if int(ei.max()) >= N:
                        issues.append(f"edge_index max={int(ei.max())} >= N={N}")

                # controllo LRC (solo un sottoinsieme per velocità)
                to_check = lrc_paths[:max_lrc_to_check]
                if to_check:
                    g0 = torch.load(to_check[0], map_location="cpu")
                    ei_ref = getattr(g0, "edge_index", None)
                    x0 = getattr(g0, "x", None)
                    if isinstance(x0, torch.Tensor) and x0.ndim == 2 and x0.shape[1] != F:
                        issues.append(f"LRC x dim={tuple(x0.shape)} != (*,{F})")
                    for p in to_check[1:]:
                        gi = torch.load(p, map_location="cpu")
                        if not (torch.is_tensor(getattr(gi, "edge_index", None)) and torch.equal(gi.edge_index, ei_ref)):
                            issues.append("LRC edge_index mismatch")
                            break
        except Exception as e:
            issues.append(f"load error: {e}")

    return issues

def main():
    ap = argparse.ArgumentParser(
        description="Verifica embeddings FULL (non ablation): presenza file + controlli forma opzionali."
    )
    ap.add_argument("--root", required=True,
                    help="Cartella che contiene train/ e/o val/ (es: embeddings_morph, embeddings_fgnet, ...)")
    ap.add_argument("--dataset", required=True,
                    choices=["MORPH","FGNET","UTKFACE","CLAP2016"])
    ap.add_argument("--split", choices=["train","val","both"], default="both")
    ap.add_argument("--check-shape", action="store_true",
                    help="Carica alcuni file e controlla coerenza N vs edge_index, dims, etc.")
    ap.add_argument("--show-complete", action="store_true")
    ap.add_argument("--min-lrc", type=int, default=8,
                    help="Minimo numero di graph_lrc_*.pt richiesti (default: 8)")
    args = ap.parse_args()

    # usa sempre la soglia passata da CLI (fallback lasciato per coerenza, ma non verrà usato)
    min_lrc = args.min_lrc if args.min_lrc is not None else expected_heads(args.dataset)

    splits = []
    if args.split in ("train","both"):
        d = os.path.join(args.root, "train")
        if os.path.isdir(d): splits.append(("train", d))
    if args.split in ("val","both"):
        d = os.path.join(args.root, "val")
        if os.path.isdir(d): splits.append(("val", d))

    if not splits:
        print(f"❌ Nessuno split trovato in {args.root}")
        sys.exit(2)

    for split_name, split_dir in splits:
        total = comp = bad = 0
        print(f"\n▶️  Split: {split_name}  ({split_dir})")
        for entry in sorted(os.listdir(split_dir)):
            d = os.path.join(split_dir, entry)
            if not os.path.isdir(d): 
                continue
            total += 1
            issues = check_sample(d, min_lrc=min_lrc, check_shape=args.check_shape)
            if issues:
                print(f"{entry}: " + " | ".join(issues))
                bad += 1
            else:
                comp += 1
                if args.show_complete:
                    print(f"{entry}: OK")

        print("----")
        print(f"Totale cartelle: {total} | Incomplete: {bad} | Complete: {comp}")

if __name__ == "__main__":
    main()