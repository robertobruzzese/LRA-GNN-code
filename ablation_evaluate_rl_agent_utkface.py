import argparse
import os
import torch
from torch.utils.data import DataLoader

from models.progressive_rl_ablation_utkface import ProgressiveRLAgent
from models.classifier_deep import AgeGroupClassifier as DeepClassifier
from models.classifier_shallow import AgeGroupClassifier as ShallowClassifier
from models.classifier_default import AgeGroupClassifier as DefaultClassifier
from models.classifier_ablation import AgeGroupClassifier as AblationClassifier

from evaluate_agent_fn import evaluate_agent

# Dataset LRC o DFE
from dataset.embedding_dataset_ablation_prlae import EmbeddingDataset  # solo DFE
from dataset.embedding_dataset_ablation_prlae_x_train_classifier import EmbeddingDatasetPRLAEXClassifier  # solo LRC


# --- CLASSIFIER-ONLY EVAL (tutta la VAL) ------------------------------------
def classifier_only_eval(dataloader, classifier, device, scaler=None, dataset_name="DATASET"):
    """
    Valuta SOLO il classificatore su tutto il validation set (decade 0..9).
    - Applica lo scaler (se fornito) allo stesso modo del training.
    - Gestisce i vari formati di sample (dict / tuple / torch_geometric Data).
    - Stampa accuracy, classification report e salva una confusion matrix.
    """
    import numpy as np
    import os
    import torch
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
    import matplotlib.pyplot as plt

    classifier.eval()
    true_decades, pred_decades = [], []

    with torch.no_grad():
        for batch in dataloader:
            sample = batch  # supporta batch_size=1 o collate default

            # --- estrai embedding (xi) e age (yi) in modo robusto ---
            if isinstance(sample, dict):
                xi = sample["embedding"].to(device)
                yi = sample["age"] if "age" in sample else sample["label"]
                yi = yi.to(device).float()
            elif isinstance(sample, (list, tuple)):
                # DataLoader con batch_size=1 spesso ritorna [ (xi, yi) ]
                item = sample[0] if len(sample) == 1 else sample
                if isinstance(item, (list, tuple)):
                    xi, yi = item
                else:
                    xi, yi = item[0], item[1]  # fallback
                xi = xi.to(device)
                yi = yi.to(device).float()
            else:
                # torch_geometric.data.Data o altro
                try:
                    xi = sample.x.to(device)
                    yi = sample.y.to(device).float()
                except Exception as _:
                    raise ValueError(f"[classifier_only_eval] Formato sample non supportato: {type(sample)}")

            # --- pooling/reshape per ottenere shape [1, 512] ---
            if xi.dim() == 3:
                # es. [1, N, 512] -> [1, 512]
                xi = xi.mean(dim=1)
            elif xi.dim() == 2:
                # es. [N, 512] -> [1, 512] (media sui N)
                xi = xi.mean(dim=0, keepdim=True)
            elif xi.dim() == 1:
                xi = xi.unsqueeze(0)
            else:
                raise ValueError(f"[classifier_only_eval] Shape embedding non supportata: {xi.shape}")

            # --- decade vera ---
            gi = int(yi.view(-1)[0].item()) // 10  # 0..9
            true_decades.append(gi)

            # --- applica scaler (se presente) nello stesso modo del training ---
            if scaler is not None:
                xi_np = xi.detach().cpu().numpy()   # [1, 512]
                xi_np = scaler.transform(xi_np)     # standardizzazione
                xi    = torch.tensor(xi_np, dtype=xi.dtype, device=xi.device)

            # --- predizione classifier (decade) ---
            logits = classifier(xi)                 # [1, num_classes]
            pred   = torch.argmax(logits, dim=1).item()
            pred_decades.append(int(pred))

    # --- metriche ---
    acc = accuracy_score(true_decades, pred_decades) * 100.0
    print(f"\n🔎 CLASSIFIER-ONLY (tutta la VAL) — decade accuracy = {acc:.2f}%")

    # classification report
    target_names = [f"{i*10}s" for i in sorted(set(true_decades) | set(pred_decades))]
    print("\n📋 Classification Report (classifier-only):")
    try:
        print(classification_report(true_decades, pred_decades, target_names=target_names, zero_division=0, digits=2))
    except Exception:
        print(classification_report(true_decades, pred_decades, zero_division=0, digits=2))

    # confusion matrix salva png
    os.makedirs("output_ablation", exist_ok=True)
    cm = confusion_matrix(true_decades, pred_decades, labels=sorted(set(true_decades) | set(pred_decades)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{i*10}s" for i in sorted(set(true_decades) | set(pred_decades))])
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Classifier-only Confusion Matrix — {dataset_name}")
    plt.tight_layout()
    out_png = f"output_ablation/{dataset_name.lower()}_classifier_only_confusion.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"📁 Confusion matrix salvata in {out_png}")

    return acc, true_decades, pred_decades
# ---------------------------------------------------------------------------

def main(args):
    dataset = args.dataset.upper()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 📁 Esperimento e cartelle
    if args.enable_lrc and not args.enable_dfe:
        exp_name = "lrc_no_dfe"
        agent_folder = "prlae_lrc_no_dfe"
        embedding_dir = f"embeddings_ablation_{dataset.lower()}_{exp_name}/val"
        dataset_obj = EmbeddingDatasetPRLAEXClassifier(
            embeddings_dir=embedding_dir,
            dataset_name=dataset,
            encoder=None,
            device=device
        )
    elif args.enable_dfe and not args.enable_lrc:
        exp_name = "no_lrc_dfe"
        agent_folder = "prlae_no_lrc_dfe"
        embedding_dir = f"embeddings_ablation_{dataset.lower()}_{exp_name}/val"
        dataset_obj = EmbeddingDataset(
            embeddings_dir=embedding_dir,
            dataset_name=dataset,
            enable_lrc=False,
            enable_dfe=True,
            return_dict=True
        )
        first_sample = dataset_obj[0]
        print("✅ Controllo tipo di sample restituito:")
        print("Tipo:", type(first_sample))
        print("Contenuto:", first_sample)
    else:
        raise ValueError("❌ Specifica solo uno tra --enable_lrc e --enable_dfe")

    checkpoint_dir = os.path.join("checkpoints_ablation", dataset.lower(), agent_folder)

    dataloader = DataLoader(dataset_obj, batch_size=1, shuffle=False)

    # 🔢 Dimensione embedding
    first_sample = dataset_obj[0]
    if isinstance(first_sample, dict):               # DFE
        embedding_dim = first_sample["embedding"].shape[-1]
    elif isinstance(first_sample, (tuple, list)):    # LRC
        embedding_dim = first_sample[0].shape[-1]
    else:
        raise ValueError("❌ Formato embedding sconosciuto.")

    # 🎯 Classificatore
    classifier_path = os.path.join(checkpoint_dir, "classifier.pth")
    dataset_name = dataset

    # 🔧 UTKFACE: architettura ablation (512→128→64→10 con dropout)
    use_utk_ablation = (dataset_name == "UTKFACE")
    use_deep = ((dataset_name == "CLAP2016" and args.enable_dfe) or
                (dataset_name == "MORPH" and args.enable_lrc) or
                (dataset_name == "MORPH" and args.enable_dfe))

    if use_utk_ablation:
        classifier = AblationClassifier(input_dim=embedding_dim).to(device)
    elif use_deep:
        state_dict = torch.load(classifier_path, map_location=device)
        first_layer_key = next(k for k in state_dict if "net.0.weight" in k or "net.0.bias" in k)
        expected_input_dim = state_dict[first_layer_key].shape[1]
        embedding_dim = expected_input_dim
        classifier = DeepClassifier(input_dim=expected_input_dim).to(device)
    else:
        # FGNET → shallow, altrimenti default
        if dataset_name == "FGNET":
            classifier = ShallowClassifier(input_dim=embedding_dim).to(device)
        else:
            classifier = DefaultClassifier(input_dim=embedding_dim).to(device)

    if os.path.exists(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        print(f"✅ Classificatore caricato da: {classifier_path}")
    else:
        raise FileNotFoundError(f"❌ Classificatore mancante: {classifier_path}")

    # 🔁 Scaler (come nel training del classifier)
    from joblib import load
    scaler_path = os.path.join(checkpoint_dir, "scaler.pkl")
    scaler = load(scaler_path) if os.path.exists(scaler_path) else None
    print(f"🔍 scaler path: {scaler_path} | exists={os.path.exists(scaler_path)}")

    # 🔎 Sanity check: classifier-only su tutta la VAL
    print("\n🧪 Avvio sanity-check CLASSIFIER-ONLY su tutta la VAL…")
    _ = classifier_only_eval(
        dataloader=dataloader,
        classifier=classifier,
        device=device,
        scaler=scaler,
        dataset_name=dataset
    )
    print("🧪 Fine sanity-check CLASSIFIER-ONLY.\n")

    # 🤖 Agente RL
    state_dim = embedding_dim + 6
    action_dim = 5
    agent = ProgressiveRLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        classifier=classifier,
        device=device
    )
    agent.q_network.to(device)

    # attacca scaler all’agente PRIMA del load/evaluate
    agent.scaler = scaler

    # 📥 Carica best model RL
    candidates = [f for f in os.listdir(checkpoint_dir) if f.startswith("best_agent_")]
    if not candidates:
        raise FileNotFoundError(f"❌ Nessun best_agent_ trovato in {checkpoint_dir}")
    best_model_path = os.path.join(checkpoint_dir, sorted(candidates)[-1])
    agent.load(best_model_path)
    agent.q_network.eval()
    agent.target_network.eval()
    print(f"✅ Modello RL caricato da: {best_model_path}")

    # 📊 Valutazione
    print(f"\n📊 Valutazione ablation PRLAE su {dataset}...\n")
    evaluate_agent(agent, dataloader, device, dataset_name=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Nome del dataset (MORPH, FGNET, UTKFACE, CLAP2016)")
    parser.add_argument("--enable_lrc", action="store_true", help="Abilita LRC (PRLAE+LRC)")
    parser.add_argument("--enable_dfe", action="store_true", help="Abilita DFE (PRLAE+DFE)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    main(args)