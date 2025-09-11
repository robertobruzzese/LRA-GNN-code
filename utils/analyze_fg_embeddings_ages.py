import os
import torch
from collections import Counter
import matplotlib.pyplot as plt

def collect_ages_from_embeddings(folder):
    ages = []
    for fname in os.listdir(folder):
        if fname.endswith(".pt"):
            try:
                path = os.path.join(folder, fname)
                data = torch.load(path, map_location="cpu")
                age = int(float(data['age']))
                ages.append(age)
            except Exception as e:
                print(f"⚠️ Errore in {fname}: {e}")
    return ages

def plot_age_distributions(train_ages, val_ages, save_path="output/embedding_age_distribution_fgnet.png"):
    def to_decade(age): return f"{(age // 10) * 10}s"

    train_decades = [to_decade(a) for a in train_ages]
    val_decades = [to_decade(a) for a in val_ages]

    all_decades = sorted(set(train_decades + val_decades))

    train_counts = Counter(train_decades)
    val_counts = Counter(val_decades)

    train_vals = [train_counts.get(d, 0) for d in all_decades]
    val_vals = [val_counts.get(d, 0) for d in all_decades]

    x = range(len(all_decades))
    plt.figure(figsize=(10, 5))
    plt.bar([i - 0.2 for i in x], train_vals, width=0.4, label='Train', color='lightblue')
    plt.bar([i + 0.2 for i in x], val_vals, width=0.4, label='Validation', color='orange')
    plt.xticks(x, all_decades)
    plt.xlabel("Decade (età)")
    plt.ylabel("Numero di embedding")
    plt.title("Distribuzione per decade nei file .pt di FG-NET (embedding)")
    plt.legend()
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Grafico salvato in {save_path}")
    plt.show()

if __name__ == "__main__":
    train_dir = "embeddings_FGNET/train"
    val_dir = "embeddings_FGNET/val"

    print("📥 Leggo età da embedding TRAIN...")
    train_ages = collect_ages_from_embeddings(train_dir)
    print(f"🔢 Trovate {len(train_ages)} embedding con età")

    print("📥 Leggo età da embedding VAL...")
    val_ages = collect_ages_from_embeddings(val_dir)
    print(f"🔢 Trovate {len(val_ages)} embedding con età")

    plot_age_distributions(train_ages, val_ages)