#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
#rebuild_dfe_features_from_rw.py.

import os
import sys
# aggiungi la root del repo al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from glob import glob
import torch

# importa il tuo modello GCN profondo
from models.resgcn import DeepResGCN  # deve esistere nel tuo repo

def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    I = torch.eye(A.size(0), device=A.device)
    A_hat = A + I
    deg = A_hat.sum(dim=1)
    # evita divisioni per zero
    deg = torch.clamp(deg, min=1e-6)
    D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

def extract_dfe_node_features_from_rw(graph_rw_path: str, num_layers: int, device: torch.device) -> torch.Tensor:
    g = torch.load(graph_rw_path, map_location=device)
    if not hasattr(g, "x"):
        raise ValueError(f"{graph_rw_path}: Data senza 'x'")
    X = g.x.to(device)                # [N, F]
    N, F = X.shape

    # costruiamo A densa (coerente con il tuo codice esistente)
    A = torch.zeros((N, N), device=device)
    ei = getattr(g, "edge_index", None)
    if ei is not None and torch.is_tensor(ei) and ei.numel() > 0:
        ei = ei.long()
        # filtra eventuali indici fuori range (robustezza)
        mask = (ei[0] < N) & (ei[1] < N)
        if mask.sum().item() > 0:
            ei = ei[:, mask]
            A[ei[0], ei[1]] = 1.0

    A_hat = normalize_adjacency(A)

    model = DeepResGCN(dim=F, num_layers=num_layers).to(device)
    with torch.no_grad():
        H = model(A_hat, X)           # [N, F]
    return H.detach().cpu()

def main():
    ap = argparse.ArgumentParser(description="Rigenera deep_features.pt per-nodo da graph_rw.pt (train/ e val/).")
    ap.add_argument("--root", required=True,
                    help="Path a embeddings_ablation_<dataset>_lrc_dfe (cartella che contiene train/ e val/)")
    ap.add_argument("--num-layers", type=int, default=4,
                    help="Numero di layer di DeepResGCN (default: 4)")
    ap.add_argument("--expect-F", type=int, default=512,
                    help="Dimensionalità attesa delle feature per nodo (default: 512)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Sovrascrive deep_features.pt (fa backup se è la versione LRC per-head N=8).")
    ap.add_argument("--only-missing", action="store_true",
                    help="Rigenera solo dove manca deep_features.pt (o deep_features_from_rw.pt se non overwrite).")
    ap.add_argument("--device", choices=["auto","cuda","mps","cpu"], default="auto",
                    help="Dispositivo: auto/cuda/mps/cpu (default: auto)")
    args = ap.parse_args()

    # selezione device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    splits = []
    for s in ("train", "val"):
        d = os.path.join(args.root, s)
        if os.path.isdir(d):
            splits.append(d)
        else:
            print(f"⏭️  Skip {s}: {d} non esiste.")

    made, skipped, errors, backed_up = 0, 0, 0, 0
    for split_dir in splits:
        print(f"\n▶️  Split: {split_dir}")
        for sample_dir in sorted([p for p in glob(os.path.join(split_dir, "*")) if os.path.isdir(p)]):
            rw_path = os.path.join(sample_dir, "graph_rw.pt")
            if not os.path.exists(rw_path):
                skipped += 1
                continue

            # decide output path
            out_path = os.path.join(sample_dir, "deep_features.pt") if args.overwrite \
                       else os.path.join(sample_dir, "deep_features_from_rw.pt")

            # skip se richiesto
            if args.only_missing and os.path.exists(out_path):
                skipped += 1
                continue

            # backup se overwrite e c'è già un deep_features.pt "per-head" (N=8)
            if args.overwrite and os.path.exists(out_path):
                try:
                    old = torch.load(out_path, map_location="cpu")
                    if isinstance(old, torch.Tensor) and old.ndim == 2 and old.shape[0] == 8:
                        bkp = os.path.join(sample_dir, "deep_features_lrc_heads.pt")
                        if not os.path.exists(bkp):
                            os.replace(out_path, bkp)
                            backed_up += 1
                        else:
                            # se esiste già il backup, rimuovi l'originale per riscrivere
                            os.remove(out_path)
                    else:
                        # rimuovi il vecchio per poter riscrivere
                        os.remove(out_path)
                except Exception:
                    # se non leggibile, proveremo a sovrascrivere comunque
                    pass

            try:
                H = extract_dfe_node_features_from_rw(rw_path, num_layers=args.num_layers, device=device)
                # controllo dimensioni
                if H.ndim != 2 or H.shape[1] != args.expect_F:
                    print(f"⚠️  {os.path.basename(sample_dir)}: shape {tuple(H.shape)} (atteso [N,{args.expect_F}])")
                torch.save(H, out_path)
                made += 1
            except Exception as e:
                print(f"❌  {os.path.basename(sample_dir)}: errore {e}")
                errors += 1

    print("\n----")
    print(f"Creati/aggiornati: {made} | Skippati: {skipped} | Errori: {errors} | Backup LRC-heads: {backed_up}")

if __name__ == "__main__":
    main()