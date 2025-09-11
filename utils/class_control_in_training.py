import os
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys

# Aggiungi path per import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.embedding_dataset_ablation_prlae import EmbeddingDataset
from dataset.embedding_dataset_ablation_prlae_x_train_classifier import EmbeddingDatasetPRLAEXClassifier
from models.ablation_train_classifier import extract_embeddings_and_labels, GraphEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--exp_name", type=str, choices=["lrc", "dfe"], required=True)
    args = parser.parse_args()

    dataset_name = args.dataset.upper()
    exp_name = args.exp_name

    if exp_name == "lrc":
        embedding_dir = f"embeddings_ablation_{dataset_name.lower()}_lrc_no_dfe/train"
        enable_lrc = True
        enable_dfe = False
        encoder = GraphEncoder().to("cpu")
        dataset = EmbeddingDatasetPRLAEXClassifier(
            embeddings_dir=embedding_dir,
            dataset_name=dataset_name,
            encoder=encoder,
            device="cpu"
        )
    elif exp_name == "dfe":
        embedding_dir = f"embeddings_ablation_{dataset_name.lower()}_no_lrc_dfe/train"
        enable_lrc = False
        enable_dfe = True
        dataset = EmbeddingDataset(
            embeddings_dir=embedding_dir,
            dataset_name=dataset_name,
            enable_lrc=False,
            enable_dfe=True,
            return_dict=True
        )
    else:
        raise ValueError("exp_name deve essere 'lrc' o 'dfe'")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    _, y_real = extract_embeddings_and_labels(dataloader, device="cpu")

    class_labels = [int(y.item()) // 10 for y in y_real]

    plt.hist(class_labels, bins=range(11), align='left', rwidth=0.8)
    plt.xlabel("Classe (decade)")
    plt.ylabel("Numero di esempi")
    plt.title(f"Distribuzione classi nel training ({dataset_name}, {exp_name.upper()})")
    plt.xticks(range(10))
    plt.grid(True)
    plt.tight_layout()
    plt.show()