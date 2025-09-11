# training/train_model_lrc_dfe.py
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
from utils.metrics import calculate_metrics

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


def _to_device_graph(graph, device):
    """Sposta su device un Data o una list[Data]."""
    if isinstance(graph, list):
        return [g.to(device) for g in graph]
    return graph.to(device)


def train_model_lrc_dfe(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: str,
    criterion,
    scheduler=None,
    early_stopping_patience: int = 10,
    show_plot: bool = True
):
    """
    Training loop compatibile con liste di grafi (LRC+DFE).
    Chiama sempre model(graph) e gestisce y dal primo grafo della lista.
    """
    model = model.to(device)
    loss_fn = criterion
    best_val_loss = float('inf')

    train_losses, val_losses = [], []
    val_mae_list, cs5_list, epsilon_list = [], [], []
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = _to_device_graph(batch, device)
            # target
            if isinstance(batch, list):
                y = batch[0].y.view(-1).float()
            else:
                y = batch.y.view(-1).float()

            optimizer.zero_grad()
            out = model(batch).view(-1)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / max(len(train_loader), 1)
        train_losses.append(avg_train_loss)

        # --- Validazione ---
        avg_val_loss, metrics = _validate_epoch_lrc_dfe(model, val_loader, device, loss_fn)
        val_losses.append(avg_val_loss)
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

    if show_plot and _HAS_PLT:
        _plot_curves(train_losses, val_losses, val_mae_list, cs5_list, epsilon_list)

    return avg_train_loss, avg_val_loss, val_mae_list[-1], cs5_list[-1], epsilon_list[-1]


def _validate_epoch_lrc_dfe(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for graph in loader:
            graph = _to_device_graph(graph, device)
            if isinstance(graph, list):
                y = graph[0].y.view(-1).float()
            else:
                y = graph.y.view(-1).float()

            preds = model(graph).view(-1)
            loss = loss_fn(preds, y)
            total_loss += loss.item()

            all_preds.append(preds)
            all_labels.append(y)

    if len(all_preds) == 0:
        # evita crash in casi limite
        return 0.0, {'MAE': 0.0, 'CS_5': 0.0, 'Epsilon_Error': 0.0}

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    m = calculate_metrics(all_preds, all_labels)

    avg_val_loss = total_loss / max(len(loader), 1)
    return avg_val_loss, m


def evaluate_model_lrc_dfe(model, loader, device, criterion) -> Tuple[float, float, float]:
    """
    Evaluation compatibile con liste di grafi. Restituisce MAE, CS@5, epsilon.
    """
    model.eval()
    total_mae = total_cs5 = total_eps = 0.0
    total = 0

    with torch.no_grad():
        for graph in loader:
            graph = _to_device_graph(graph, device)
            if isinstance(graph, list):
                y = graph[0].y.view(-1).float()
            else:
                y = graph.y.view(-1).float()

            out = model(graph).view(-1)
            # metriche per-sample
            abs_err = torch.abs(out - y)
            mae = abs_err.mean().item()
            cs5 = (abs_err <= 5).float().mean().item() * 100.0
            eps = (abs_err / (y.abs() + 1e-6)).mean().item()

            total_mae += mae
            total_cs5 += cs5
            total_eps += eps
            total += 1

    return total_mae / max(total, 1), total_cs5 / max(total, 1), total_eps / max(total, 1)


def _plot_curves(train_losses, val_losses, val_mae_list, cs5_list, epsilon_list):
    import matplotlib.pyplot as plt
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