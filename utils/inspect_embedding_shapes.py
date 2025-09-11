import os
import torch

folder = "embeddings_ablation_morph_no_lrc_no_dfe/train"

count_128 = 0
count_512 = 0
count_other = 0

print("\n📊 Analisi delle shape dei file .pt nella cartella:")
print("---------------------------------------------------")

for fname in os.listdir(folder):
    if fname.endswith(".pt"):
        path = os.path.join(folder, fname)
        try:
            tensor = torch.load(path)
            if isinstance(tensor, torch.Tensor):
                shape = tuple(tensor.shape)
                if shape == (1, 128):
                    count_128 += 1
                elif shape == (1, 512):
                    count_512 += 1
                else:
                    count_other += 1
                    print(f"⚠️  {fname} -> shape atipica: {shape}")
            else:
                count_other += 1
                print(f"⚠️  {fname} -> non è un tensore torch.Tensor")
        except Exception as e:
            print(f"❌ Errore caricando {fname}: {e}")

# 📈 Risultati
print("\n✅ Risultati finali:")
print(f"- Embedding con shape [1, 128]: {count_128}")
print(f"- Embedding con shape [1, 512]: {count_512}")
print(f"- Altri (non tensor o shape diversa): {count_other}")