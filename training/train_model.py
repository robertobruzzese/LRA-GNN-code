import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.metrics import calculate_metrics
import matplotlib.pyplot as plt

DEBUG = False  # imposta a True se vuoi stampe per debug

def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, save_path,
                criterion, scheduler=None, early_stopping_patience=10, show_plot=True):
    model = model.to(device)
    loss_fn = criterion
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    val_mae_list = []
    cs5_list = []
    epsilon_list = []
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            is_list = isinstance(batch, list)
            if is_list:
                batch = [b.to(device) for b in batch]
                y = batch[0].y.view(-1).float()
            else:
                batch = batch.to(device)
                y = batch.y.view(-1).float()

            #outputs = model(batch).view(-1)
            try:
                outputs = model(batch).view(-1)
            except TypeError:
                outputs = model(batch.x, batch.edge_index, batch.edge_attr).view(-1)
            loss = loss_fn(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 🔍 Validazione
        model.eval()
        with torch.no_grad():
            val_loss = 0
            all_preds = []
            all_labels = []
            for batch in val_loader:
                is_list = isinstance(batch, list)
                if is_list:
                    batch = [b.to(device) for b in batch]
                    batch_y = batch[0].y.view(-1)
                else:
                    batch = batch.to(device)
                    batch_y = batch.y.view(-1)

                #preds = model(batch).view(-1)
                try:
                    preds = model(batch).view(-1)
                except TypeError:
                    preds = model(batch.x, batch.edge_index, batch.edge_attr).view(-1)
                loss = loss_fn(preds, batch_y.float())
                val_loss += loss.item()

                all_preds.append(preds)
                all_labels.append(batch_y)

            if len(all_preds) == 0:
                print("❌ Nessuna predizione valida. Interrompo.")
                return 0, 0, 0, 0, 0

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            if all_preds.shape[0] != all_labels.shape[0]:
                min_len = min(all_preds.shape[0], all_labels.shape[0])
                if DEBUG:
                    print(f"⚠️ Dimension mismatch: preds={all_preds.shape}, labels={all_labels.shape} → truncated to {min_len}")
                all_preds = all_preds[:min_len]
                all_labels = all_labels[:min_len]

            metrics = calculate_metrics(all_preds, all_labels)

            val_mae_list.append(metrics['MAE'])
            cs5_list.append(metrics['CS_5'])
            epsilon_list.append(metrics['Epsilon_Error'])

        print(f"\n📉 Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        print(f"📊 Val MAE = {metrics['MAE']:.2f}, CS@5 = {metrics['CS_5']:.2f}%, ϵ = {metrics['Epsilon_Error']:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

        if save_path and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved to {save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement ({epochs_no_improve}/{early_stopping_patience})")

        if epochs_no_improve >= early_stopping_patience:
            print("🛑 Early stopping triggered!")
            break

    if show_plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(val_mae_list, label='Val MAE')
        plt.plot(cs5_list, label='CS@5')
        plt.plot(epsilon_list, label='Epsilon Error')
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return avg_train_loss, avg_val_loss, metrics['MAE'], metrics['CS_5'], metrics['Epsilon_Error']


def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_mae = 0.0
    total_cs5 = 0.0
    total_eps = 0.0
    total = 0

    with torch.no_grad():
        for graph in loader:
            if isinstance(graph, list):
                graph = [g.to(device) for g in graph]
                y = graph[0].y
            else:
                graph = graph.to(device)
                y = graph.y

            #out = model(graph)
            if hasattr(graph, 'x'):
                out = model(graph.x, graph.edge_index, graph.edge_attr)
            else:
                out = model(graph)
            mae = torch.abs(out - y).item()
            cs5 = 100.0 if mae <= 5 else 0.0
            eps = mae / (y.item() + 1e-6)

            total_mae += mae
            total_cs5 += cs5
            total_eps += eps
            total += 1

    return total_mae / total, total_cs5 / total, total_eps / total


def evaluate_model_debug(model, loader, device, criterion, N=10):
    model.eval()
    total_mae = 0.0
    total_cs5 = 0.0
    total_eps = 0.0
    total = 0

    with torch.no_grad():
        for idx, graph in enumerate(loader):
            if isinstance(graph, list):
                graph = [g.to(device) for g in graph]
                y = graph[0].y
            else:
                graph = graph.to(device)
                y = graph.y

            out = model(graph)

            print(f"🟡 Sample {idx}: y.shape = {y.shape}, y = {y}")
            if y.item() <= 0:
                print("⚠️ Valore sospetto per y:", y.item())

            mae = torch.abs(out - y).item()
            eps = mae / (y.item() + 1e-6)
            cs5 = 100.0 if mae <= 5 else 0.0

            print(f"   ➤ Pred = {out.item():.2f}, GT = {y.item():.2f}, MAE = {mae:.2f}, ε = {eps:.4f}")

            total_mae += mae
            total_cs5 += cs5
            total_eps += eps
            total += 1

            if idx >= N - 1:
                break

    return total_mae / total, total_cs5 / total, total_eps / total