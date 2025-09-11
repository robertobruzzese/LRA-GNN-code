import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

def analyze_fgnet_ages(image_folder, save_plot=False, plot_path="fgnet_age_distribution.png"):
    """
    Analizza la distribuzione delle età nel dataset FG-NET e salva CSV con risultati.
    
    Parametri:
        image_folder (str): percorso alla cartella con le immagini FG-NET.
        save_plot (bool): se True, salva il grafico invece di mostrarlo.
        plot_path (str): percorso per salvare il grafico (se save_plot=True).
    """
    if not os.path.exists(image_folder):
        print(f"❌ Cartella non trovata: {image_folder}")
        return

    ages = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(".jpg") and "A" in filename:
            try:
                age_str = filename.split("A")[1].split(".")[0]
                age_digits = ''.join(filter(str.isdigit, age_str))  # solo numeri
                age = int(age_digits)
                ages.append(age)
            except Exception as e:
                print(f"⚠️ Impossibile estrarre età da file: {filename} → {e}")
                continue

    print(f"\n📊 Numero totale di immagini: {len(ages)}")
    print("📌 Distribuzione per età (esatta):")
    exact_counts = dict(Counter(sorted(ages)))
    print(exact_counts)

    # 📄 Salva CSV con distribuzione esatta
    exact_df = pd.DataFrame(sorted(exact_counts.items()), columns=["Age", "Count"])
    os.makedirs("output", exist_ok=True)
    exact_df.to_csv("output/age_distribution_fgnet_exact.csv", index=False)
    print("✅ Salvato CSV: output/age_distribution_fgnet_exact.csv")

    decades = [f"{(a // 10) * 10}s" for a in ages]
    decade_counts = Counter(decades)
    print("\n📌 Distribuzione per decade:")
    for decade, count in sorted(decade_counts.items()):
        print(f"  {decade}: {count} immagini")

    # 📄 Salva CSV con distribuzione per decade
    decade_df = pd.DataFrame(sorted(decade_counts.items()), columns=["Decade", "Count"])
    decade_df.to_csv("output/age_distribution_fgnet_decades.csv", index=False)
    print("✅ Salvato CSV: output/age_distribution_fgnet_decades.csv")

    # 📈 Grafico decade
    plt.figure(figsize=(8, 5))
    plt.bar(decade_counts.keys(), decade_counts.values(), color="skyblue")
    plt.title("Distribuzione per decade nel dataset FG-NET")
    plt.xlabel("Decade (età)")
    plt.ylabel("Numero di immagini")
    plt.grid(axis="y")
    plt.tight_layout()

    if save_plot:
        os.makedirs("output", exist_ok=True)
        plt.savefig(os.path.join("output", plot_path), dpi=300)
        print(f"✅ Grafico salvato in output/{plot_path}")
    else:
        plt.show()


def count_embeddings(embedding_root):
    """
    Conta il numero di file .pt in embedding_root/train e embedding_root/val.
    """
    train_dir = os.path.join(embedding_root, "train")
    val_dir = os.path.join(embedding_root, "val")

    def count_pt_files(path):
        if not os.path.exists(path):
            return 0
        return len([f for f in os.listdir(path) if f.endswith(".pt")])

    train_count = count_pt_files(train_dir)
    val_count = count_pt_files(val_dir)

    print(f"\n📦 Numero di embedding in:")
    print(f"🔹 Train → {train_count}")
    print(f"🔹 Val   → {val_count}")

    # 📈 Plot comparativo
    plt.figure(figsize=(6, 4))
    plt.bar(["Train", "Validation"], [train_count, val_count], color=["lightgreen", "orange"])
    plt.title("Numero di embedding FG-NET: Train vs Validation")
    plt.ylabel("Numero di campioni")
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/embedding_count_fgnet.png", dpi=300)
    print("✅ Grafico salvato in output/embedding_count_fgnet.png")
    plt.show()


if __name__ == "__main__":
    # 🧠 Analizza immagini originali
    analyze_fgnet_ages(
        image_folder="datasets/data/FGNET/images/Train/images_preprocessed",
        save_plot=True,
        plot_path="fgnet_age_distribution.png"
    )

    # 📦 Conta embedding in train e val
    count_embeddings(embedding_root="embeddings_FGNET")