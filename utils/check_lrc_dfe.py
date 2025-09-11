#!/usr/bin/env python3
import os, sys, argparse, glob

def main():
    ap = argparse.ArgumentParser(
        description="Controlla per ogni immagine: 8 graph_lrc_*.pt, 1 graph_rw.pt e 1 deep_features(_from_rw).pt"
    )
    ap.add_argument("--val-dir", required=True,
                    help="Path a .../embeddings_ablation_<dataset>_lrc_dfe/val")
    ap.add_argument("--expect-lrc", type=int, default=8)
    ap.add_argument("--show-complete", action="store_true")
    ap.add_argument("--check-shape", action="store_true",
                    help="(Opz.) verifica N feature vs edge_index del graph_rw")
    args = ap.parse_args()

    val_dir = args.val_dir
    if not os.path.isdir(val_dir):
        print(f"❌ Directory non trovata: {val_dir}", file=sys.stderr); sys.exit(2)

    total = bad = 0
    for entry in sorted(os.listdir(val_dir)):
        d = os.path.join(val_dir, entry)
        if not os.path.isdir(d): continue
        total += 1

        lrc = len(glob.glob(os.path.join(d, "graph_lrc_*.pt")))
        dfe_graph = os.path.exists(os.path.join(d, "graph_rw.pt"))
        # ✅ accetta deep_features_from_rw.pt oppure deep_features.pt
        dfe_feat_path = None
        for fn in ("deep_features_from_rw.pt", "deep_features.pt"):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                dfe_feat_path = p
                break
        dfe_feats = dfe_feat_path is not None

        ok = (lrc == args.expect_lrc) and dfe_graph and dfe_feats
        if not ok:
            print(f"{entry}: LRC={lrc}/{args.expect_lrc}, DFE_graph={int(dfe_graph)}/1, DFE_feats={int(dfe_feats)}/1")
            bad += 1
            continue

        # (Opz.) controllo coerenza N(feature) vs edge_index max
        if args.check_shape:
            try:
                import torch
                x = torch.load(dfe_feat_path, map_location="cpu")
                g = torch.load(os.path.join(d, "graph_rw.pt"), map_location="cpu")
                if hasattr(x, "shape") and len(x.shape) == 2:
                    N, F = x.shape
                    ei = getattr(g, "edge_index", None)
                    if ei is not None and getattr(ei, "numel", lambda:0)() > 0:
                        max_idx = int(ei.max().item())
                        if max_idx >= N:
                            print(f"{entry}: edge_index max={max_idx} >= N={N} (mismatch ordering?)")
                            bad += 1
                else:
                    print(f"{entry}: deep_features shape non 2D (trovato {getattr(x,'shape',None)})")
                    bad += 1
            except Exception as e:
                print(f"{entry}: errore nel check-shape: {e}")
                bad += 1

        if args.show_complete:
            print(f"{entry}: OK")

    print("----")
    print(f"Totale cartelle: {total} | Incomplete: {bad} | Complete: {total - bad}")
    sys.exit(0 if bad == 0 else 1)

if __name__ == "__main__":
    main()