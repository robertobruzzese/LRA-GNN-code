import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import torch
from deep_feature_extraction import extract_deep_features


device = 'mps' if torch.backends.mps.is_available() else 'cpu'
num_layers = 4
feature_dim = 512  # Adatta se i tuoi nodi hanno dim. diverse

def regenerate_deep_features(root_dir):
    all_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for emb_dir in all_dirs:
        deep_feat_path = os.path.join(emb_dir, "deep_features.pt")
        if os.path.exists(deep_feat_path):
            print(f"🔁 {emb_dir}: rimuovo deep_features.pt")
            os.remove(deep_feat_path)

        # Usa i file LRC se esistono, altrimenti RW
        graph_paths = sorted(glob.glob(os.path.join(emb_dir, "graph_lrc_*.pt")))
        if not graph_paths:
            graph_rw_path = os.path.join(emb_dir, "graph_rw.pt")
            if not os.path.exists(graph_rw_path):
                print(f"⚠️  Nessun grafo LRC o RW per {emb_dir}, salto.")
                continue
            graph_paths = [graph_rw_path]

        print(f"✨ Generazione deep_features.pt per {emb_dir}")
        try:
            deep_feats = extract_deep_features(graph_paths, feature_dim=feature_dim, num_layers=num_layers, device=device)
            torch.save(deep_feats, deep_feat_path)
            print(f"✅ deep_features.pt salvato in {emb_dir}")
        except Exception as e:
            print(f"❌ Errore in {emb_dir}: {e}")


if __name__ == "__main__":
    regenerate_deep_features("embeddings_clap2016/train")
    regenerate_deep_features("embeddings_clap2016/val")