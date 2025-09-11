import os
import torch

def verify_embedding_dir(directory):
    print(f"\n🔍 Verifica nella directory: {directory}")
    
    if not os.path.exists(directory):
        print("❌ Directory non trovata.")
        return
    
    pt_files = [f for f in os.listdir(directory) if f.endswith(".pt") and not f.startswith("graph")]
    
    if not pt_files:
        print("⚠️ Nessun file .pt trovato nella directory (esclusi i file di grafo).")
        return

    success = 0
    failed = 0

    for fname in pt_files:
        fpath = os.path.join(directory, fname)
        try:
            data = torch.load(fpath, map_location='cpu')
            if not isinstance(data, dict):
                print(f"❌ [{fname}] non è un dizionario.")
                failed += 1
                continue
            if "embedding" not in data or "age" not in data:
                print(f"⚠️ [{fname}] mancano 'embedding' o 'age'")
                failed += 1
                continue
            if not torch.is_tensor(data["embedding"]):
                print(f"❌ [{fname}] 'embedding' non è un tensor.")
                failed += 1
                continue
            success += 1
        except Exception as e:
            print(f"❌ Errore nel file {fname}: {e}")
            failed += 1

    print(f"\n✅ File validi: {success}")
    print(f"❌ File corrotti o errati: {failed}")
    print(f"📁 Totale file analizzati: {len(pt_files)}")

if __name__ == "__main__":
    # Percorsi reali da usare per CLAP2016 ablation LRC no DFE
    train_dir = "embeddings_ablation_clap2016_lrc_no_dfe/train"
    val_dir = "embeddings_ablation_clap2016_lrc_no_dfe/val"

    verify_embedding_dir(train_dir)
    verify_embedding_dir(val_dir)