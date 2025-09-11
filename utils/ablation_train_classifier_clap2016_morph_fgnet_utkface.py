import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import Counter
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from dataset.embedding_dataset_ablation_prlae import EmbeddingDataset
from dataset.embedding_dataset_ablation_prlae_x_train_classifier import EmbeddingDatasetPRLAEXClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss) if self.reduction == 'mean' else focal_loss

class AgeGroupClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)
    
def print_class_distribution(y_tensor, split_name):
    y_classes = [int(age.item()) // 10 for age in y_tensor]
    class_counts = Counter(y_classes)
    print(f"\n📊 Distribuzione classi in {split_name}:")
    for cls in sorted(class_counts.keys()):
        print(f"  Classe {cls * 10:02d}-{cls * 10 + 9:02d}: {class_counts[cls]} campioni")



def extract_embeddings_and_labels(dataloader, device):
    X_list, y_list = [], []
    for batch in dataloader:
        sample = batch[0]

        #print(f"📦 Sample type: {type(sample)}")
        #print(sample)

        if isinstance(sample, dict):
            embeddings = sample["embedding"].to(device)
            ages = sample["age"].to(device)

            if embeddings.dim() == 2:
                embeddings = embeddings.mean(dim=0, keepdim=True)
            elif embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            elif embeddings.dim() > 2:
                print(f"❌ Embedding con dimensione troppo alta: {embeddings.shape}")
                continue

        elif isinstance(sample, (list, tuple)):
            embeddings, ages = sample
            embeddings = embeddings.to(device)
            ages = ages.to(device)

        elif isinstance(sample, torch_geometric.data.Data):
            # Caso DFE-only: embeddings nei nodi (x), label = y
            embeddings = sample.x.to(device)  # [num_nodes, 512]
            ages = sample.y.to(device)        # float (es. 36.9)

            # Pooling: media sugli embedding dei nodi
            if embeddings.dim() == 2:
                embeddings = embeddings.mean(dim=0, keepdim=True)
            else:
                print("❌ Embedding non 2D:", embeddings.shape)
                continue

            if ages.dim() == 0:
                ages = ages.unsqueeze(0)

        else:
            print("⚠️ Sample ignorato:", type(sample))
            continue

        X_list.append(embeddings)
        #y_list.append(ages)
        y_list.append(ages.view(-1))  # ➜ Garantisce 1D, batch-safe #

    if not X_list:
        raise ValueError("❌ Nessun embedding caricato: controlla il dataset o i file in input.")
    print(f"📊 Estrazione completata: {len(X_list)} embeddings")
    print(f"📐 Shape primo embedding: {X_list[0].shape}")
    #X = torch.cat(X_list, dim=0)
    X = torch.stack(X_list, dim=0)  # 👈 mantiene le righe separate: [num_samples, embedding_dim]
    y = torch.cat(y_list, dim=0)

    return X, y

def train_classifier(train_loader, val_loader, input_dim, device='cpu', epochs=1000, class_weights=None, num_classes=10):
    model = AgeGroupClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = FocalLoss(alpha=class_weights)  # << usa l'arg passato
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = FocalLoss(alpha=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    best_val_acc = 0.0
    best_model = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == yb).sum().item()

        train_acc = 100 * correct / len(train_loader.dataset)

        # Val
        model.eval()
        correct_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                correct_val += (logits.argmax(dim=1) == yb).sum().item()
        val_acc = 100 * correct_val / len(val_loader.dataset)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

        scheduler.step()

    final_model = AgeGroupClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    final_model.load_state_dict(best_model)
    return final_model

# 🚀 Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                    choices=["FGNET","MORPH","CLAP2016","UTKFACE"])
    parser.add_argument("--exp_name", type=str, required=True, choices=["lrc", "dfe"])
    args = parser.parse_args()

    dataset_name = args.dataset.upper()
    exp_name = args.exp_name
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if exp_name == "lrc":
        train_dir = f"embeddings_ablation_{dataset_name.lower()}_lrc_no_dfe/train"
        val_dir = f"embeddings_ablation_{dataset_name.lower()}_lrc_no_dfe/val"
        train_dataset = EmbeddingDatasetPRLAEXClassifier(train_dir, dataset_name, encoder=None, device=device)
        val_dataset = EmbeddingDatasetPRLAEXClassifier(val_dir, dataset_name, encoder=None, device=device)
        exp_tag = "prlae_lrc_no_dfe"
        input_dim = 512
    elif exp_name == "dfe":
        train_dir = f"embeddings_ablation_{dataset_name.lower()}_no_lrc_dfe/train"
        val_dir = f"embeddings_ablation_{dataset_name.lower()}_no_lrc_dfe/val"
        train_dataset = EmbeddingDataset(train_dir, dataset_name, return_dict=True, enable_lrc=False, enable_dfe=True)
        val_dataset = EmbeddingDataset(val_dir, dataset_name, return_dict=True, enable_lrc=False, enable_dfe=True)
        exp_tag = "prlae_no_lrc_dfe"
        input_dim = 512
    else:
        raise ValueError("Esperimento non valido")

    #train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    X_train, y_train = extract_embeddings_and_labels(train_loader, device)
    X_val, y_val = extract_embeddings_and_labels(val_loader, device)
    # ✅ Rimuove la dimensione extra nel caso DFE (shape: [N, 1, 512])
    if X_train.dim() == 3 and X_train.shape[1] == 1:
        X_train = X_train.squeeze(1)
    if X_val.dim() == 3 and X_val.shape[1] == 1:
        X_val = X_val.squeeze(1)
    scaler = StandardScaler()
    # ⬇️ INSERISCI QUI
    print(f"📐 X_train shape: {X_train.shape}")
    print(f"📐 X_val shape: {X_val.shape}")
    
    X_train_np = X_train.detach().cpu().numpy()
    X_val_np = X_val.detach().cpu().numpy()
    print(f"✅ Training features: {X_train_np.shape}")
    print(f"✅ Validation features: {X_val_np.shape}")

    if X_train_np.shape[1] != X_val_np.shape[1]:
        raise ValueError(f"❌ Dimensione incoerente: train={X_train_np.shape[1]}, val={X_val_np.shape[1]}")

    if X_train_np.ndim == 1:
        X_train_np = X_train_np.reshape(1, -1)
    if X_val_np.ndim == 1:
        X_val_np = X_val_np.reshape(1, -1)

    X_train = torch.tensor(scaler.fit_transform(X_train_np), dtype=torch.float).to(device)
    X_val = torch.tensor(scaler.transform(X_val_np), dtype=torch.float).to(device)
    #X_train = torch.tensor(scaler.fit_transform(X_train.cpu()), dtype=torch.float).to(device)
   # X_train = torch.tensor(scaler.fit_transform(X_train.detach().cpu().numpy()), dtype=torch.float).to(device)
    #X_val = torch.tensor(scaler.transform(X_val.cpu()), dtype=torch.float).to(device)
    #X_val = torch.tensor(scaler.transform(X_val.detach().cpu().numpy()), dtype=torch.float).to(device)
   # ✅ Stampa distribuzione delle classi
    print_class_distribution(y_train, "train/")
    print_class_distribution(y_val, "val/")

    # 📊 Calcolo pesi classi

    y_train_cls_np = np.array([int(age.item()) // 10 for age in y_train])
    y_train_cls = torch.tensor(y_train_cls_np, dtype=torch.long).to(device)
    y_val_cls_np = np.array([int(age.item()) // 10 for age in y_val])
    y_val_cls = torch.tensor(y_val_cls_np, dtype=torch.long).to(device)
    # Calcolo pesi iniziali
    # 📊 Calcolo pesi classi
    # Classi presenti nei dati
    print(f"✅ X_train shape: {X_train.shape}")
    print(f"✅ y_train shape: {y_train_cls.shape}")
    # ✅ Stima dinamica del numero classi (in decadi)
    present_train = np.unique(y_train_cls_np)
    present_val   = np.unique(y_val_cls_np)
    num_classes   = int(max(present_train.max(), present_val.max()) + 1)

    print(f"🔢 Classi presenti (train): {present_train.tolist()}")
    print(f"🔢 Classi presenti (val):   {present_val.tolist()}")
    print(f"🔢 num_classes = {num_classes}")

    # 📊 Pesi di classe (solo per le classi presenti in train)
    computed = compute_class_weight(
        class_weight='balanced',
        classes=present_train,
        y=y_train_cls_np
    ).astype(np.float32)

    class_weights = np.zeros(num_classes, dtype=np.float32)
    for cls, w in zip(present_train, computed):
        class_weights[int(cls)] = w

    # (opzionale) amplifica ulteriormente le classi molto rare
    class_weights[class_weights > 2.0] *= 1.5

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

    # ✅ DataLoader per training e validation
    train_loader = DataLoader(TensorDataset(X_train, y_train_cls), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val_cls),   batch_size=32)

# 🏋️‍♀️ Training del classificatore
    model = train_classifier(
        train_loader,
        val_loader,
        input_dim=input_dim,
        device=device,
        epochs=1000,
        class_weights=class_weights_tensor,
        num_classes=num_classes
    )
    print(f"📐 Embedding dimension: {input_dim} | Train: {X_train.size(0)} | Val: {X_val.size(0)}")

    ckpt_dir = os.path.join("checkpoints_ablation", dataset_name.lower(), exp_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "classifier.pth"))
    print(f"✅ Classificatore salvato in {ckpt_dir}/classifier.pth")
    from joblib import dump
    dump(scaler, os.path.join(ckpt_dir, "scaler.pkl"))
    print(f"✅ Scaler salvato in {ckpt_dir}/scaler.pkl")
    # 🔍 Valutazione finale con confusion matrix
    # 🔍 Valutazione finale con confusion matrix
    model.eval()
    all_preds, all_true = [], []
    val_classes = torch.tensor([int(age.item()) // 10 for age in y_val])
    val_loader_cm = DataLoader(TensorDataset(X_val, val_classes), batch_size=32)

    with torch.no_grad():
        for xb, yb in val_loader_cm:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    labels_for_cm = list(sorted(set(all_true)))  # oppure: range(num_classes)
    cm = confusion_matrix(all_true, all_preds, labels=labels_for_cm)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix (val set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    report   = classification_report(all_true, all_preds, digits=2)
    macro_f1 = f1_score(all_true, all_preds, average='macro') * 100
    accuracy = (np.array(all_true) == np.array(all_preds)).mean() * 100
    print(f"\n🎯 Accuracy (val set): {accuracy:.2f}%")
    print(f"📊 Macro F1-score (val set): {macro_f1:.2f}%")
    print(f"\n📋 Classification Report:\n{report}")