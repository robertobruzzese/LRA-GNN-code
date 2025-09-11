import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import os


def evaluate_agent(agent, dataloader, device, dataset_name="DATASET"):
    # 📊 Esecuzione valutazione
    results = agent.evaluate(model=None, dataloader=dataloader, device=device)
    print("📦 Chiavi contenute in results:", results.keys())

    y_true = np.array(results["true_ages"])
    #y_pred = np.array(results["predicted_ages"])
    if "agent_final_ages" in results:
        print("✅ Usando le età finali dell'RL agent (senza regressore).")
        y_pred = np.array(results["agent_final_ages"])
    else:
        print("⚠️ 'agent_final_ages' non trovato. Uso 'predicted_ages' dal regressore.")
        y_pred = np.array(results["predicted_ages"])

    def compute_mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def compute_cs(y_true, y_pred, k=5):
        return np.mean(np.abs(y_true - y_pred) <= k) * 100

    def compute_epsilon_error(y_true, y_pred, sigma=5):
        squared_diff = (y_true - y_pred) ** 2
        exp_term = np.exp(-squared_diff / (2 * sigma ** 2))
        return 1 - np.mean(exp_term)

    # 📐 Metriche
    mae = compute_mae(y_true, y_pred)
    cs5 = compute_cs(y_true, y_pred, k=5)
    eps = compute_epsilon_error(y_true, y_pred)

    print(f"\n📐 MAE: {mae:.2f}")
    print(f"📊 CS@5: {cs5:.2f}%")
    print(f"⚠️ Epsilon-error: {eps:.4f}")

    # 📊 Tabella comparativa
    if dataset_name == "FGNET":
        paper_entry = {"Dataset": "FGNET", "Method": "LRA-GNN (Paper)", "MAE": 2.14, "CS@5 (%)": 91.6, "Param.": "13M"}
    elif dataset_name == "MORPH":
        paper_entry = {"Dataset": "MORPH", "Method": "LRA-GNN (Paper)", "MAE": 2.21, "CS@5 (%)": "-", "Param.": "13M"}
    elif dataset_name == "UTKFACE":
        paper_entry = {"Dataset": "UTKFACE", "Method": "LRA-GNN (Paper)", "MAE": "4.22", "CS@5 (%)": "-", "Param.": "13M"}
    elif dataset_name == "CLAP2016":
        paper_entry = {"Dataset": "CLAP2016", "Method": "LRA-GNN (Paper)", "MAE": "3.11", "CS@5 (%)": "-", "Param.": "13M"}
    else:
        paper_entry = {"Dataset": dataset_name, "Method": "LRA-GNN (Paper)", "MAE": "-", "CS@5 (%)": "-", "Param.": "13M"}

    ours_entry = {
        "Dataset": dataset_name,
        "Method": "LRA-GNN (Ours)",
        "MAE": round(mae, 2),
        "CS@5 (%)": round(cs5, 2),
        "ε-error": round(eps, 4),
        "Param.": "?"  # Puoi aggiornare con i parametri reali
    }

    df = pd.DataFrame([paper_entry, ours_entry])
    print(f"\n📋 Tabella comparativa prestazioni su {dataset_name}:")
    print(df.to_string(index=False))

    os.makedirs("output_ablation", exist_ok=True)
    df.to_csv(f"output_ablation/{dataset_name.lower()}_comparison_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 1.6))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title(f"📋 Performance Comparison on {dataset_name}", pad=12)
    plt.tight_layout()
    plt.savefig(f"output_ablation/{dataset_name.lower()}_results_table.png", dpi=300)
    plt.close()

    # Classification report
    true_labels = results["true_labels"]
    predicted_labels = results["predicted_labels"]
    acc = accuracy_score(true_labels, predicted_labels) * 100

    all_labels = sorted(set(true_labels) | set(predicted_labels))
    target_names = [f"{i*10}s" for i in all_labels]

    report_dict = classification_report(
        true_labels, predicted_labels, labels=all_labels,
        target_names=target_names, output_dict=True, zero_division=0
    )
    report_table = pd.DataFrame(report_dict).transpose()
    report_table.to_csv("output_ablation/classification_report.csv")

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
    plt.title("📊 Classification Report - RL Agent", pad=20)
    plt.tight_layout()
    plt.savefig("output_ablation/classification_report_table.png", dpi=300)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.tight_layout()
    plt.savefig("output_ablation/confusion_matrix.png", dpi=300)
    plt.close()

    # True vs Predicted CSV
    df_compare = pd.DataFrame({"True Age": y_true, "Predicted Age": y_pred})
    df_compare.to_csv("output_ablation/true_vs_predicted.csv", index=False)
    print("✅ Salvataggi completati in output_ablation/")
    print("👉 True ages:", results["true_ages"][:10])
    print("👉 Predicted ages:", results["predicted_ages"][:10])
    return mae, cs5, eps