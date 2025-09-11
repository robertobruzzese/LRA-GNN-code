import sys
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# ✅ Fix per importare dal progetto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.embedding_dataset_ablation_prlae_x_train_classifier import EmbeddingDatasetPRLAEXClassifier
from models.classifier import AgeGroupClassifier
from models.progressive_rl_ablation import ProgressiveRLAgent
from evaluate_agent_fn import evaluate_agent

# 📁 Percorsi
dataset = "UTKFACE"
embedding_dir = f"embeddings_ablation_{dataset.lower()}_lrc_no_dfe/train"
checkpoint_dir = f"checkpoints_ablation/{dataset.lower()}/prlae_lrc_no_dfe"
best_agent_path = os.path.join(checkpoint_dir, "best_agent_2025-08-02_09-13-35.pth")

# ⚙️ Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 📦 Dataset
train_dataset = EmbeddingDatasetPRLAEXClassifier(
    embeddings_dir=embedding_dir,
    dataset_name=dataset,
    encoder=None,
    device=device
)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# 🎯 Classificatore
classifier_path = os.path.join(checkpoint_dir, "classifier.pth")
classifier = AgeGroupClassifier(input_dim=512).to(device)
classifier.load_state_dict(torch.load(classifier_path, map_location=device))
classifier.eval()

# 🤖 Agente RL
agent = ProgressiveRLAgent(state_dim=518, action_dim=5, classifier=classifier, device=device)
agent.load(best_agent_path)

# 📊 Valutazione sul training set
print(f"\n📊 Valutazione del best agent su TRAINING SET ({dataset})...\n")
mae, cs5, eps = evaluate_agent(agent, train_loader, device, dataset_name=dataset)

# ✅ Accuracy separata
results = agent.evaluate(model=None, dataloader=train_loader, device=device)
true_labels = results["true_labels"]
predicted_labels = results["predicted_labels"]
acc = accuracy_score(true_labels, predicted_labels) * 100

print(f"\n✅ Accuracy sul training set: {acc:.2f}%")