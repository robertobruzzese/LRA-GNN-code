import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

def cosine_similarity(tensor1, tensor2):
    return F.cosine_similarity(tensor1.view(1, -1), tensor2.view(1, -1)).item()

def compare_graph_features(sample_folder):
    graphs = []
    for i in range(8):
        path = os.path.join(sample_folder, f"graph_lrc_{i}.pt")
        if not os.path.exists(path):
            print(f"❌ Manca: {path}")
            return
        graphs.append(torch.load(path))

    print(f"🔍 Confronto grafi in: {sample_folder}")
    
    # Confronto sulle feature dei nodi (X)
    print("\n🧠 Similarità tra X (feature dei nodi):")
    similarities_x = []
    for i in range(8):
        for j in range(i+1, 8):
            sim = cosine_similarity(graphs[i].x.mean(dim=0), graphs[j].x.mean(dim=0))
            similarities_x.append(sim)
            print(f"   CosineSim(x_{i}, x_{j}) = {sim:.4f}")
    avg_sim_x = sum(similarities_x) / len(similarities_x)
    print(f"📊 Media similarità X: {avg_sim_x:.4f}")

    # Confronto sui pesi degli archi (edge_attr)
    print("\n🔗 Differenze tra edge_attr (pesi degli archi):")
    differences_e = []
    for i in range(8):
        for j in range(i+1, 8):
            if graphs[i].edge_attr.shape != graphs[j].edge_attr.shape:
                print(f"   ⚠️ Mismatch shape edge_attr tra {i} e {j}")
                continue
            diff = torch.norm(graphs[i].edge_attr - graphs[j].edge_attr).item()
            differences_e.append(diff)
            print(f"   NormDiff(edge_attr_{i}, edge_attr_{j}) = {diff:.4f}")
    avg_diff_e = sum(differences_e) / len(differences_e)
    print(f"📊 Media differenza edge_attr: {avg_diff_e:.4f}")

    # 🔎 Conclusione
    print("\n📌 Conclusione:")
    if avg_sim_x > 0.98:
        print("🧠 Le feature dei nodi (x) sono molto simili (o identiche) tra le head.")
    else:
        print("🧠 Le feature dei nodi (x) sono distinte tra le head.")

    if avg_diff_e < 1e-3:
        print("⚠️ I pesi degli archi (edge_attr) sono troppo simili: grafi quasi duplicati!\n")
    else:
        print("✅ I grafi LRC differiscono nei pesi degli archi: LRC funziona correttamente.\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Percorso alla sottocartella con i file graph_lrc_*.pt")
    args = parser.parse_args()

    compare_graph_features(args.folder)