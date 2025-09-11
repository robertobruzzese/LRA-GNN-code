import os
import torch

def check_embeddings(directory):
    ok, missing, corrupt = 0, 0, 0
    for entry in os.listdir(directory):
        subdir = os.path.join(directory, entry)
        if not os.path.isdir(subdir):
            continue
        try:
            emb_path = os.path.join(subdir, 'embedding.pt')
            age_path = os.path.join(subdir, 'age.pt')
            if not os.path.exists(emb_path) or not os.path.exists(age_path):
                print(f"❌ Mancanti: {entry}")
                missing += 1
                continue
            # Prova a caricare i file
            embedding = torch.load(emb_path)
            age = torch.load(age_path)
            if not isinstance(embedding, torch.Tensor) or not isinstance(age, (int, float, torch.Tensor)):
                print(f"⚠️ Formato non valido: {entry}")
                corrupt += 1
                continue
            ok += 1
        except Exception as e:
            print(f"💥 Errore in {entry}: {e}")
            corrupt += 1

    print(f"\n✅ OK: {ok} | ❌ Mancanti: {missing} | 💥 Corrotti: {corrupt}")

# Usa il tuo percorso
check_embeddings("embeddings_ablation_morph_no_lrc_dfe/train")