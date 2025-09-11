import os
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(".")  # o il path corretto se necessario
from utils.ablation_train_classifier_clap2016_morph_fgnet_utkface import AgeGroupClassifier
from dataset.embedding_dataset_ablation_prlae_x_train_classifier import EmbeddingDatasetPRLAEXClassifier

@torch.no_grad()
def main():
    dataset = "UTKFACE"
    exp_name = "prlae_lrc_no_dfe"
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    embedding_dir = f"embeddings_ablation_{dataset.lower()}_lrc_no_dfe/val"
    checkpoint = f"checkpoints_ablation/{dataset.lower()}/{exp_name}/classifier.pth"

    dataset_obj = EmbeddingDatasetPRLAEXClassifier(
        embeddings_dir=embedding_dir,
        dataset_name=dataset,
        encoder=None,
        device=device
    )

    loader = DataLoader(dataset_obj, batch_size=1, shuffle=False)

    # 🔢 Recupera dimensione embedding dinamicamente
    embedding_dim = dataset_obj[0][0].shape[-1]
    classifier = AgeGroupClassifier(input_dim=embedding_dim).to(device)

    if not os.path.exists(checkpoint):
        print(f"❌ Checkpoint non trovato: {checkpoint}")
        return

    try:
        classifier.load_state_dict(torch.load(checkpoint, map_location=device))
        classifier.eval()
    except RuntimeError as e:
        print(f"❌ Errore nel caricamento del classificatore:\n{str(e)}")
        return

    correct = 0
    total = 0

    print("\n🧪 Esempi di predizioni:")
    for i, (embedding, label) in enumerate(loader):
        embedding, label = embedding.to(device), label.to(device)
        output = classifier(embedding)
        pred = torch.argmax(output, dim=1)

        # Converti età reale in decade
        true_label = (label // 10).long()
        if i < 5:
            print(f"[Sample {i}] Pred: {pred.item()}, True: {label.item()}")
        correct += (pred == true_label).sum().item()
        total += true_label.size(0)
        print(f"[Sample {i}] Pred: {pred.item()}, True: {true_label.item()}")

    acc = correct / total * 100
    print(f"\n✅ Accuracy classificatore su {dataset} val/: {acc:.2f}%")

if __name__ == "__main__":
    main()