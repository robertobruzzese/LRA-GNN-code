import torch
from torch.utils.data import DataLoader
from dataset.embedding_dataset import EmbeddingDataset
from models.progressive_rl import ProgressiveRLAgent
from models.lra_gnn import LRA_GNN
from training.rl_environment import RLEnvironment
from models.classifier import AgeGroupClassifier
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np

# === CROSS-DATASET EVALUATION: Use MORPH-trained model on FGNET ===

# 📌 Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 🔁 Load FGNET validation embeddings
embedding_dir = "embeddings_FGNET/val"
checkpoint_dir = os.path.join("checkpoints", "MORPH")  # Use model trained on MORPH

print("🔁 Cross-Evaluation: model trained on MORPH, evaluated on FGNET")

# 🔍 Find latest best_agent
agent_files = glob.glob(os.path.join(checkpoint_dir, "best_agent_*.pth"))
if not agent_files:
    raise FileNotFoundError("❌ No best_agent_*.pth file found in checkpoints!")
agent_files.sort()
agent_path = agent_files[-1]
print(f"📂 Selected agent file: {agent_path}")

# ⚙️ Hyperparameters consistent with training
state_dim = 134
action_dim = 5

# 🔁 Load dataset
dataset = EmbeddingDataset(embedding_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 🎯 Load LRA-GNN model if needed
model = LRA_GNN()
model.to(device)
model.eval()

# 🤖 Load trained RL agent
agent = ProgressiveRLAgent(state_dim=state_dim, action_dim=action_dim)
agent.q_network.load_state_dict(torch.load(agent_path, map_location=device))
agent.q_network.to(device)
agent.q_network.eval()

# ➕ Load classifier
embedding_dim = next(iter(dataloader))[0].shape[1]
classifier_path = os.path.join(checkpoint_dir, "classifier.pth")
classifier = AgeGroupClassifier(input_dim=embedding_dim).to(device)
classifier.load_state_dict(torch.load(classifier_path, map_location=device))
classifier.eval()
agent.classifier = classifier

# 🔬 Evaluate
results = agent.evaluate(model=None, dataloader=dataloader, device=device)
true_labels = results["true_labels"]
predicted_labels = results["predicted_labels"]
acc = accuracy_score(true_labels, predicted_labels) * 100

# 📊 Classification report
all_labels = sorted(set(true_labels) | set(predicted_labels))
target_names = [f"{i*10}s" for i in all_labels]
report_dict = classification_report(true_labels, predicted_labels, labels=all_labels, target_names=target_names, output_dict=True, zero_division=0)
report_table = pd.DataFrame(report_dict).transpose()

# 📈 Plot table
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')
table = ax.table(cellText=report_table.round(2).values,
                 colLabels=report_table.columns,
                 rowLabels=report_table.index,
                 loc='center',
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title("Classification Report - RL Agent on FGNET", pad=20)
os.makedirs("output_cross_fgnet", exist_ok=True)
plt.savefig("output_cross_fgnet/classification_report_table.png", dpi=300)
plt.tight_layout()
plt.show()

# 🔲 Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - FGNET")
plt.tight_layout()
plt.savefig("output_cross_fgnet/confusion_matrix.png", dpi=300)
plt.show()

# 📈 Smoothed accuracy plot
correctness = [int(t == p) * 100 for t, p in zip(true_labels, predicted_labels)]
window_size = 20
moving_avg = np.convolve(correctness, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(moving_avg, label=f"Moving Average (window={window_size})", color='tab:blue')
plt.axhline(y=acc, color='red', linestyle='--', label=f"Overall Accuracy = {acc:.2f}%")
plt.ylabel("Accuracy (%)")
plt.title("Smoothed Accuracy on FGNET Validation Samples")
plt.xlabel("Sample Index")
plt.legend()
plt.grid(True)
plt.savefig("output_cross_fgnet/smoothed_accuracy.png", dpi=300)
plt.tight_layout()
plt.show()
