import os
import torch
import argparse

def check_files(embeddings_dir):
    pt_files = [f for f in os.listdir(embeddings_dir) if f.endswith(".pt")]
    found_issues = []

    for fname in pt_files:
        fpath = os.path.join(embeddings_dir, fname)
        try:
            obj = torch.load(fpath)
            print(f"🔍 {fname}: type = {type(obj)}")

            if isinstance(obj, dict):
                keys = list(obj.keys())
                print(f"    ➤ keys: {keys}")

                if 'age' in keys or 'label' in keys:
                    print(f"⚠️  Potenziale data leakage: chiave sospetta in {fname}: {keys}")
                    found_issues.append((fname, keys))
            else:
                print(f"    ➤ File {fname} non è un dizionario.")
        except Exception as e:
            print(f"❌ Errore nel leggere {fname}: {e}")

    if not found_issues:
        print("\n✅ Nessun campo 'age' o 'label' trovato nei file .pt.")
    else:
        print("\n❗ Verifica necessaria nei seguenti file:")
        for f, keys in found_issues:
            print(f" - {f}: chiavi = {keys}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Cartella contenente i file .pt di embedding da verificare.")
    args = parser.parse_args()

    check_files(args.embeddings_dir)