# ablation_train_rl_utkface.py

import argparse
import os
import re
import glob
from datetime import datetime

import torch
from torch import nn
from torch_geometric.loader import DataLoader as PyGDataLoader

from models.progressive_rl_ablation import ProgressiveRLAgent
from training.train_rl_ablation import train_prlae
from dataset.embedding_dataset_ablation_prlae import EmbeddingDataset

# Classificatori
from models.classifier_ablation import AgeGroupClassifier as AblationClassifier
from models.classifier import AgeGroupClassifier as DefaultClassifier
from models.classifier_deep import AgeGroupClassifier as DeepClassifier
from models.classifier_shallow import AgeGroupClassifier as ShallowClassifier

# -------------------------
# Wrapper: applica StandardScaler SOLO al ramo classifier (start-row)
# -------------------------
class ScaledClassifier(nn.Module):
    def __init__(self, base_classifier: nn.Module, scaler, device):
        super().__init__()
        self.base = base_classifier
        self.scaler = scaler
        self.device = device

    def forward(self, x: torch.Tensor):
        # x atteso [B, D]
        if self.scaler is not None:
            x_np = x.detach().cpu().numpy()
            x_np = self.scaler.transform(x_np)  # mantiene shape [B, D]
            x = torch.from_numpy(x_np).to(self.device).type_as(x)
        return self.base(x)

# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Nome del dataset (MORPH, FGNET, UTKFACE, CLAP2016)")
    parser.add_argument("--enable_lrc", action="store_true", help="Abilita LRC")
    parser.add_argument("--enable_dfe", action="store_true", help="Abilita DFE")
    args = parser.parse_args()

    dataset_name = args.dataset.upper()
    if dataset_name != "UTKFACE":
        print("⚠️  Questo script è specializzato per UTKFACE. Per altri dataset usa il trainer generico.")
        # Non esco con errore: magari vuoi comunque testarlo.
        # Se preferisci forzare: uncomment -> raise SystemExit(1)

    print("🚀 Launching UTKFACE RL training (ablation):")
    print(f"    dataset   = {dataset_name}")
    print(f"    enable_lrc = {args.enable_lrc}")
    print(f"    enable_dfe = {args.enable_dfe}")

    # -------------------------
    # Paths & device
    # -------------------------
    if args.enable_lrc and not args.enable_dfe:
        exp_name = "prlae_lrc_no_dfe"
        embedding_name = "lrc_no_dfe"
    elif args.enable_dfe and not args.enable_lrc:
        exp_name = "prlae_no_lrc_dfe"
        embedding_name = "no_lrc_dfe"
    else:
        raise ValueError("❌ Devi specificare **solo uno** tra --enable_lrc e --enable_dfe")

    embedding_dir = f"embeddings_ablation_{args.dataset.lower()}_{embedding_name}/train"
    if not os.path.exists(embedding_dir):
        raise FileNotFoundError(f"❌ Directory embeddings non trovata: {embedding_dir}")

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Dataset & Loader
    # -------------------------
    embedding_dataset = EmbeddingDataset(
        embeddings_dir=embedding_dir,
        dataset_name=args.dataset,
        enable_lrc=args.enable_lrc,
        enable_dfe=args.enable_dfe
    )
    print(f"📁 Controllo directory embeddings: {embedding_dir}")
    print("🧪 Sample count:", len(embedding_dataset))

    embedding_loader = PyGDataLoader(
        embedding_dataset,
        batch_size=1,
        shuffle=True
    )

    # -------------------------
    # Determina embedding_dim
    # -------------------------
    sample = embedding_dataset[0]
    if isinstance(sample, (list, tuple)):
        graph = sample[0]
    else:
        graph = sample

    if not hasattr(graph, "x") or graph.x is None:
        raise ValueError("❌ Il grafo non contiene l'attributo 'x'")
    embedding_dim = graph.x.shape[1]
    state_dim = embedding_dim + 6
    action_dim = 5

    # -------------------------
    # Classifier + Scaler (solo UTK)
    # -------------------------
    checkpoint_dir = os.path.join("checkpoints_ablation", dataset_name.lower(), exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    classifier_path = os.path.join(checkpoint_dir, "classifier.pth")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"❌ Nessun classifier.pth trovato in {classifier_path}")

    # Per UTKFace usiamo l'architettura di ablation (512→128→64→10 con dropout)
    # Se i tuoi ckpt sono di un'altra architettura, cambia qui il costruttore.
    base_classifier = AblationClassifier(input_dim=embedding_dim).to(device)
    try:
        base_classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        base_classifier.eval()
        print(f"✅ Classificatore caricato da {classifier_path}")
    except RuntimeError as e:
        # Fallback automatico (tentativo) se l'architettura non matcha
        print("⚠️  Mismatch architettura. Provo Default/Deep/Shallow in fallback…")
        tried = False
        for alt in (DefaultClassifier, DeepClassifier, ShallowClassifier):
            try:
                alt_cls = alt(input_dim=embedding_dim).to(device)
                alt_cls.load_state_dict(torch.load(classifier_path, map_location=device))
                alt_cls.eval()
                base_classifier = alt_cls
                tried = True
                print(f"✅ Caricato con architettura alternativa: {alt.__name__}")
                break
            except Exception:
                continue
        if not tried:
            raise RuntimeError(f"❌ Errore nel caricamento del classificatore: {e}")

    # Carica lo StandardScaler del classifier (se salvato)
    scaler = None
    try:
        from joblib import load
        scaler_path = os.path.join(checkpoint_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = load(scaler_path)
            print(f"✅ Scaler caricato da {scaler_path}")
        else:
            print("ℹ️  scaler.pkl non trovato: userò il classifier senza standardizzazione (solo per il ramo classifier).")
    except Exception as e:
        print(f"⚠️  Impossibile caricare scaler.pkl ({e}). Procedo senza scaler.")

    # Wrappa il classifier: lo scaler si applica SOLO qui
    classifier = ScaledClassifier(base_classifier, scaler, device).to(device)
    classifier.eval()

    # -------------------------
    # Instanzia agente RL
    # -------------------------
    agent = ProgressiveRLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        classifier=classifier  # <— passa il wrapper scalato
    )
    agent.q_network.to(device)

    # -------------------------
    # Resume da partial checkpoints (se presenti)
    # -------------------------
    def extract_episode_num(path):
        m = re.search(r"rl_agent_partial_(\d+)_", path)
        return int(m.group(1)) if m else -1

    partials = glob.glob(os.path.join(checkpoint_dir, "rl_agent_partial_*.pth"))
    partials = sorted(partials, key=extract_episode_num)
    print("📁 Checkpoints parziali rilevati:", partials)
    if partials:
        last_ckpt = partials[-1]
        agent.load(last_ckpt)
        print(f"📥 Checkpoint caricato da {last_ckpt}")
        start_step = extract_episode_num(last_ckpt)
    else:
        print("🚀 Nessun checkpoint parziale trovato. Inizio training da zero.")
        start_step = 0

    # -------------------------
    # Training loop (chiama train_prlae esistente)
    # -------------------------
    best_model_dir = checkpoint_dir
    best_accuracy = 0.0

    for step in range(start_step, 200, 50):
        best_accuracy = train_prlae(
            agent=agent,
            dataloader=embedding_loader,
            device=device,
            dataset_name=dataset_name,
            num_episodes=50,
            start_episode=step,
            save_every=10,
            best_accuracy=best_accuracy,
            best_model_dir=best_model_dir,
            classifier=classifier,
            embedding_dim=embedding_dim
        )

    # Salvataggio finale dell’agente
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_path = os.path.join(checkpoint_dir, f"rl_agent_{timestamp}.pth")
    agent.save(final_path)
    print(f"\n💾 RL agent salvato in: {final_path}")
    print("\n🏁 Training RL completato (UTKFACE)!")
    

if __name__ == "__main__":
    main()