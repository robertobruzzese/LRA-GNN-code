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
from sklearn.metrics import classification_report, f1_score, accuracy_score

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

from sklearn.metrics import f1_score

def train_classifier(train_loader, val_loader, input_dim, device='cpu',
                     epochs=1000, class_weights=None, y_train_dec=None, num_classes=10, tau=0.5):
    model = AgeGroupClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # prior (solo per val/test)
    priors = np.bincount(y_train_dec, minlength=num_classes).astype(np.float32)
    priors /= max(len(y_train_dec), 1)
    log_prior = torch.tensor(priors, dtype=torch.float32, device=device).clamp_min(1e-12).log()

    best_f1, best_state = 0.0, None
    patience, wait = 100, 0

    for epoch in range(epochs):
        model.train()
        total_loss, corr = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)                    # NO adjustment in train
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            corr += (logits.argmax(1) == yb).sum().item()
        train_acc = 100 * corr / len(train_loader.dataset)
        scheduler.step()

        # VALIDAZIONE (con logit adjustment)
        model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb) - tau * log_prior.unsqueeze(0)
                preds.extend(out.argmax(1).cpu().numpy())
                targs.extend(yb.cpu().numpy())
        val_acc = (np.array(preds) == np.array(targs)).mean() * 100
        val_f1  = f1_score(targs, preds, average='macro') * 100

        if (epoch+1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}% - Val F1: {val_f1:.2f}% - Best F1: {best_f1:.2f}%")

        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"🛑 Early stopping @epoch {epoch+1} (best macro‑F1={best_f1:.2f}%)")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, log_prior  # ritorno anche il log_prior per l'eval finale

# 🚀 Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
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
    present_classes = np.unique(y_train_cls_np)

    # Calcola i pesi solo per le classi presenti
    computed = compute_class_weight(class_weight='balanced',
                                classes=present_classes,
                                y=y_train_cls_np)

    # Inizializza vettore con 0.0 per tutte le 10 classi
    class_weights = np.zeros(10, dtype=np.float32)

    # Inserisci i pesi solo nelle posizioni delle classi presenti
    for cls, w in zip(present_classes, computed):
        class_weights[cls] = w

    # Amplifica ulteriormente le classi con peso elevato (> 2.0)
    class_weights[class_weights > 2.0] *= 1.5

    # Converti in tensore
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    # 🔁 Amplifica ulteriormente le classi con peso elevato
    class_weights = np.array(class_weights)
    class_weights[class_weights > 2.0] *= 1.5

    # 🎯 Converti in tensore
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    # ✅ DataLoader per training e validation
    train_loader = DataLoader(TensorDataset(X_train, y_train_cls), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val_cls), batch_size=32)

# 🏋️‍♀️ Training del classificatore
    #model = train_classifier(train_loader, val_loader, input_dim=input_dim, device=device)
    # y_train_dec serve per i priors/logit adjustment
    y_train_dec = y_train_cls_np.copy()
    num_classes = 10
    tau = 0.5  # prova anche 0.0 e 0.8

    model, log_prior = train_classifier(
        train_loader, val_loader,
        input_dim=input_dim, device=device,
        epochs=1000,
        class_weights=class_weights_tensor,
        y_train_dec=y_train_dec,
        num_classes=num_classes,
        tau=tau
    )
    print(f"📐 Embedding dimension: {input_dim} | Train: {X_train.size(0)} | Val: {X_val.size(0)}")

    ckpt_dir = os.path.join("checkpoints_ablation", dataset_name.lower(), exp_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "classifier.pth"))
    print(f"✅ Classificatore salvato in {ckpt_dir}/classifier.pth")
    # 🔍 Valutazione finale con confusion matrix
    model.eval()
    all_preds = []
    all_true = []

    val_classes = torch.tensor([int(age.item()) // 10 for age in y_val])
    val_loader_cm = DataLoader(TensorDataset(X_val, val_classes), batch_size=32)

    with torch.no_grad():
        for xb, yb in val_loader_cm:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_true.extend(yb.numpy())

    from sklearn.metrics import classification_report, f1_score, accuracy_score

    # 📊 Confusion matrix
    cm = confusion_matrix(all_true, all_preds, labels=range(9))  # 9 classi da 0 a 8
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix (val set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # 📋 F1-score, Precision, Recall, Accuracy
    report = classification_report(all_true, all_preds, digits=2)
    macro_f1 = f1_score(all_true, all_preds, average='macro') * 100
    accuracy = accuracy_score(all_true, all_preds) * 100

    print(f"\n🎯 Accuracy (val set): {accuracy:.2f}%")
    print(f"📊 Macro F1-score (val set): {macro_f1:.2f}%")
    print(f"\n📋 Classification Report:\n{report}")