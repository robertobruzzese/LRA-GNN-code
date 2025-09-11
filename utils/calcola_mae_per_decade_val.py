import os
import torch
import argparse
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from models.progressive_rl_ablation import ProgressiveRLAgent
from models.classifier import AgeGroupClassifier
from dataset.embedding_dataset_ablation_prlae_x_train_classifier import EmbeddingDatasetPRLAEXClassifier

@torch.no_grad()
def evaluate_mae_per_decade(agent, dataloader, device):
    errors_by_decade = defaultdict(list)

    for sample in dataloader:
        if isinstance(sample, dict):
            embedding = sample["embedding"].to(device)
            true_age = sample["label"].item()
        else:
            embedding, label = sample
            embedding = embedding.to(device)
            true_age = label.item()

        with torch.no_grad():
            group_logits = agent.classifier(embedding)
            group_pred = torch.argmax(group_logits, dim=1)
            age_pred = group_pred.item() * 10 + 5  # centro decade
        mae = abs(age_pred - true_age)
        decade = int(true_age) // 10
        errors_by_decade[decade].append(mae)

    # 📊 Stampa MAE medio per decade
    print("\n📈 MAE per Decade (Validation Set):\n")
    for dec in sorted(errors_by_decade.keys()):
        maes = errors_by_decade[dec]
        avg_mae = np.mean(maes)
        print(f"Decade {dec*10}-{dec*10+9}: MAE = {avg_mae:.2f} (n={len(maes)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset es. UTKFACE")
    args = parser.parse_args()

    dataset = args.dataset.upper()
    exp_name = "lrc_no_dfe"
    agent_folder = "prlae_lrc_no_dfe"

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    embedding_dir = f"embeddings_ablation_{dataset.lower()}_{exp_name}/val"
    checkpoint_dir = os.path.join("checkpoints_ablation", dataset.lower(), agent_folder)

    # 📦 Dataset
    dataset_obj = EmbeddingDatasetPRLAEXClassifier(
        embeddings_dir=embedding_dir,
        dataset_name=dataset,
        encoder=None,
        device=device
    )
    dataloader = DataLoader(dataset_obj, batch_size=1, shuffle=False)

    # 📥 Carica classifier
    classifier_path = os.path.join(checkpoint_dir, "classifier.pth")
    classifier = AgeGroupClassifier(input_dim=512).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()

    # 📥 Carica agente RL
    agent = ProgressiveRLAgent(state_dim=518, action_dim=5, classifier=classifier, device=device)
    best_model = sorted(f for f in os.listdir(checkpoint_dir) if f.startswith("best_agent_"))[-1]
    agent.load(os.path.join(checkpoint_dir, best_model))

    # 🔍 Calcola MAE per decade
    evaluate_mae_per_decade(agent, dataloader, device)