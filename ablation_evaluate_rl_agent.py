import argparse
import os
import torch
from torch.utils.data import DataLoader
from models.progressive_rl_ablation import ProgressiveRLAgent
from models.classifier_extended import AgeGroupClassifier as ExtendedClassifier
from models.classifier_deep import AgeGroupClassifier as DeepClassifier
from models.classifier_matching import AgeGroupClassifier as MatchingClassifier
from models.classifier_shallow import AgeGroupClassifier as ShallowClassifier
from models.classifier_default import AgeGroupClassifier as DefaultClassifier
from models.classifier_ablation import AgeGroupClassifier as AblationClassifier


from evaluate_agent_fn import evaluate_agent

# Dataset LRC o DFE
from dataset.embedding_dataset_ablation_prlae import EmbeddingDataset  # solo DFE
from dataset.embedding_dataset_ablation_prlae_x_train_classifier import EmbeddingDatasetPRLAEXClassifier  # solo LRC

def main(args):
    dataset = args.dataset.upper()
       # ⚙️ Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # 📁 Esperimento e cartelle
    if args.enable_lrc and not args.enable_dfe:
        exp_name = "lrc_no_dfe"
        agent_folder = "prlae_lrc_no_dfe"
        embedding_dir = f"embeddings_ablation_{dataset.lower()}_{exp_name}/val"
        dataset_obj = EmbeddingDatasetPRLAEXClassifier(
            embeddings_dir=embedding_dir,
            dataset_name=dataset,
            encoder=None,  # Encoder non necessario in fase di evaluation (embedding già estratti)
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
                # 🔍 DEBUG: Verifica formato restituito dal dataset
        first_sample = dataset_obj[0]
        print("✅ Controllo tipo di sample restituito:")
        print("Tipo:", type(first_sample))
        print("Contenuto:", first_sample)
    else:
        raise ValueError("❌ Specifica solo uno tra --enable_lrc e --enable_dfe")

    checkpoint_dir = os.path.join("checkpoints_ablation", dataset.lower(), agent_folder)

    dataloader = DataLoader(dataset_obj, batch_size=1, shuffle=False)
    
    # 🔍 DEBUG: ispeziona i primi sample
    for i, item in enumerate(dataloader):
        if isinstance(item, dict):
            #age = item["age"]
            age = item["age"] if "age" in item else item["label"]
        elif isinstance(item, (tuple, list)):
            _, age = item
        else:
            raise ValueError(f"[DEBUG] Formato non previsto: {type(item)}")

        print(f"[DEBUG #{i}] Type age: {type(age)} | Value: {age}")
        if i > 5: break
    # 🔢 Dimensione embedding
    first_sample = dataset_obj[0]
    if isinstance(first_sample, dict):  # caso DFE
        embedding_dim = first_sample["embedding"].shape[-1]
    elif isinstance(first_sample, (tuple, list)):  # caso LRC
        embedding_dim = first_sample[0].shape[-1]
    else:
        raise ValueError("❌ Formato embedding sconosciuto.")

    # ⚙️ Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 🎯 Classificatore
    # 🎯 Classificatore
    classifier_path = os.path.join(checkpoint_dir, "classifier.pth")

    # 🧠 Selezione del tipo di classificatore
    dataset_name = dataset  # già in maiuscolo

    # ⚠️ CLAP2016 usa DeepClassifier SOLO SE DFE è abilitato
    use_deep = ((dataset_name == "CLAP2016" and args.enable_dfe) or
                (dataset_name == "MORPH" and args.enable_lrc) or
                (dataset_name == "MORPH" and args.enable_dfe) 
                
                )
    # 🔧 UTKFACE: i ckpt sono 512→128→64→10 con dropout (architettura ablation)
    use_utk_ablation = (dataset_name == "UTKFACE")

    # ⚠️ UTKFACE usa ExtendedClassifier SOLO SE LRC è abilitato
    #use_extended = (dataset_name == "UTKFACE" and args.enable_lrc)
    #use_extended = (dataset_name == "CLAP2016" and args.enable_dfe)
    #use_extended = (   (dataset_name == "UTKFACE" and args.enable_dfe) )
    #use_default = (   (dataset_name == "UTKFACE" and args.enable_lrc) )
    use_shallow = (
        (dataset_name == "FGNET" and args.enable_lrc) or
        (dataset_name == "FGNET" and args.enable_dfe)
)
   # from models.classifier import ExtendedClassifier, DeepClassifier, DefaultClassifier  # assicurati che ci siano

# 🎯 Selezione finale
    if use_utk_ablation:
        classifier = AblationClassifier(input_dim=embedding_dim).to(device)
    elif use_deep:
        # ✅ Ricava l'input_dim corretto dai pesi salvati
        state_dict = torch.load(classifier_path, map_location=device)
        first_layer_key = next(k for k in state_dict if "net.0.weight" in k or "net.0.bias" in k)
        expected_input_dim = state_dict[first_layer_key].shape[1]
        embedding_dim = expected_input_dim  # ⚠️ ← forza la dimensione giusta
        classifier = DeepClassifier(input_dim=expected_input_dim).to(device)
        #classifier = DeepClassifier(input_dim=embedding_dim).to(device
    elif use_shallow:
        classifier = ShallowClassifier(input_dim=embedding_dim).to(device)
    else:
        classifier = DefaultClassifier(input_dim=embedding_dim).to(device)

    # 📥 Caricamento pesi
    if os.path.exists(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        print(f"✅ Classificatore caricato da: {classifier_path}")
    else:
        raise FileNotFoundError(f"❌ Classificatore mancante: {classifier_path}")
    # 🤖 Agente RL
    state_dim = embedding_dim + 6
    action_dim = 5
    agent = ProgressiveRLAgent(state_dim=state_dim, action_dim=action_dim, classifier=classifier, device=device)
    agent.q_network.to(device)
    from joblib import load
    scaler_path = os.path.join(checkpoint_dir, "scaler.pkl")
    scaler = load(scaler_path) if os.path.exists(scaler_path) else None
    agent.scaler = scaler  # passa lo scaler all’agente
    # 📥 Carica best model RL
    candidates = [f for f in os.listdir(checkpoint_dir) if f.startswith("best_agent_")]
    if not candidates:
        raise FileNotFoundError(f"❌ Nessun best_agent_ trovato in {checkpoint_dir}")
    best_model_path = os.path.join(checkpoint_dir, sorted(candidates)[-1])
    agent.load(best_model_path)
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

    # ⚙️ Dispositivo (spostato fuori per usarlo anche per il dataset LRC)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    main(args)