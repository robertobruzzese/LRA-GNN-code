#!/usr/bin/env python3
import os, sys, argparse, glob, torch

def main():
    ap = argparse.ArgumentParser(
        description="Controlla per immagine: 8 graph_lrc_*.pt, 1 graph_rw.pt e deep_features(_from_rw).pt"
    )
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--expect-lrc", type=int, default=8)
    ap.add_argument("--show-complete", action="store_true")
    ap.add_argument("--check-shape", action="store_true")
    args = ap.parse_args()

    droot = args.val_dir
    if not os.path.isdir(droot):
        print(f"❌ Directory non trovata: {droot}", file=sys.stderr); sys.exit(2)

    total = bad = 0
    for entry in sorted(os.listdir(droot)):
        d = os.path.join(droot, entry)
        if not os.path.isdir(d): continue
        total += 1

        lrc = len(glob.glob(os.path.join(d, "graph_lrc_*.pt")))
        has_rw = os.path.exists(os.path.join(d, "graph_rw.pt"))

        feat_path = None
        for fn in ("deep_features_from_rw.pt", "deep_features.pt"):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                feat_path = p
                break
        has_feat = feat_path is not None

        ok_files = (lrc == args.expect_lrc) and has_rw and has_feat
        shape_ok = True

        if ok_files and args.check_shape:
            try:
                x = torch.load(feat_path, map_location="cpu")
                g = torch.load(os.path.join(d, "graph_rw.pt"), map_location="cpu")
                if not (isinstance(x, torch.Tensor) and x.ndim == 2):
                    print(f"{entry}: deep_features shape non 2D (trovato {getattr(x,'shape',None)})")
                    shape_ok = False
                else:
                    N = x.shape[0]
                    ei = getattr(g, "edge_index", None)
                    if ei is not None and torch.is_tensor(ei) and ei.numel() > 0:
                        max_idx = int(ei.max().item())
                        if max_idx >= N:
                            print(f"{entry}: edge_index max={max_idx} >= N={N} (mismatch)")
                            shape_ok = False
            except Exception as e:
                print(f"{entry}: errore check-shape: {e}")
                shape_ok = False

        if not ok_files:
            print(f"{entry}: LRC={lrc}/{args.expect_lrc}, DFE_graph={int(has_rw)}/1, DFE_feats={int(has_feat)}/1")
            bad += 1
        elif not shape_ok:
            bad += 1
        elif args.show_complete:
            print(f"{entry}: OK")

    print("----")
    print(f"Totale cartelle: {total} | Incomplete: {bad} | Complete: {total - bad}")
    sys.exit(0 if bad == 0 else 1)

if __name__ == "__main__":
    main()