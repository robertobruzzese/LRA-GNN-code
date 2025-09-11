#train_lra_gnn.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader


from models.lra_gnn import LRA_GNN
from training.train_model import train_model
from dataset.age_estimation_dataset import AgeEstimationDataset
from dataset.age_estimation_dataset_fgnet import AgeEstimationDatasetFGNET
from dataset.age_estimation_dataset_morph import AgeEstimationDatasetMorph
from dataset.age_estimation_dataset_utkface import AgeEstimationDatasetUTKFace
from dataset.age_estimation_dataset_clap2016 import AgeEstimationDatasetClap2016

# saver piatto unificato (NON sovrascrivere il nome "save_embeddings")
from utils.save_embeddings_flat import save_embeddings as save_flat_embeddings


def train_lra_gnn(
    dataset_name="MORPH",
    enable_lrc=False,
    enable_dfe=False,
    force_ablation=True,
    embeddings_dir=None,
    checkpoints_dir=None,
    use_prlae=False,
    ablation=False,
    save_graph_embeddings=True,   # <-- rinominato: prima era `save_embeddings`
    save_flat=False               # <-- opzionale per FULL
):
    print("🧩 train_lra_gnn - FLAGS:")
    print(f"    enable_lrc = {enable_lrc}")
    print(f"    enable_dfe = {enable_dfe}")

    exp_name_parts = []
    if enable_lrc: exp_name_parts.append("lrc")
    if enable_dfe: exp_name_parts.append("dfe")
    exp_name = "_".join(exp_name_parts) if exp_name_parts else "gcn_only"

    is_ablation = force_ablation or enable_lrc or enable_dfe

    DATASET_NAME = dataset_name.upper()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"📁 DATASET_NAME: {DATASET_NAME}")
    print(f"⚙️  Esperimento: {exp_name}")

    if embeddings_dir is None or checkpoints_dir is None:
        if is_ablation:
            checkpoint_dir = os.path.join("checkpoints_ablation", DATASET_NAME.lower(), exp_name)
            embedding_dir  = f"embeddings_ablation_{DATASET_NAME.lower()}_{exp_name}"
        else:
            checkpoint_dir = os.path.join("checkpoints", DATASET_NAME)
            embedding_dir  = f"embeddings_{DATASET_NAME.lower()}"
    else:
        checkpoint_dir = checkpoints_dir
        embedding_dir  = embeddings_dir

    os.makedirs(os.path.join(embedding_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(embedding_dir, "val"),   exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_lra_gnn_{DATASET_NAME.lower()}.pth")

    if is_ablation:
        train_dir = os.path.join(embedding_dir, "train")
        val_dir   = os.path.join(embedding_dir, "val")

        if DATASET_NAME == "UTKFACE":
            train_dataset = AgeEstimationDatasetUTKFace(train_dir, DATASET_NAME, "train", enable_lrc=enable_lrc, enable_dfe=enable_dfe)
            val_dataset   = AgeEstimationDatasetUTKFace(val_dir,   DATASET_NAME, "val",   enable_lrc=enable_lrc, enable_dfe=enable_dfe)
        elif DATASET_NAME == "CLAP2016":
            train_dataset = AgeEstimationDatasetClap2016(train_dir, DATASET_NAME, "train", enable_lrc=enable_lrc, enable_dfe=enable_dfe)
            val_dataset   = AgeEstimationDatasetClap2016(val_dir,   DATASET_NAME, "val",   enable_lrc=enable_lrc, enable_dfe=enable_dfe)
        elif DATASET_NAME == "FGNET":
            train_dataset = AgeEstimationDatasetFGNET(train_dir, DATASET_NAME, "train", enable_lrc=enable_lrc, enable_dfe=enable_dfe)
            val_dataset   = AgeEstimationDatasetFGNET(val_dir,   DATASET_NAME, "val",   enable_lrc=enable_lrc, enable_dfe=enable_dfe)
        elif DATASET_NAME == "MORPH":
            train_dataset = AgeEstimationDatasetMorph(train_dir, DATASET_NAME, "train", enable_lrc=enable_lrc, enable_dfe=enable_dfe)
            val_dataset   = AgeEstimationDatasetMorph(val_dir,   DATASET_NAME, "val",   enable_lrc=enable_lrc, enable_dfe=enable_dfe)
        else:
            raise ValueError(f"Dataset non supportato: {DATASET_NAME}")
    else:
        dataset_dir = f"embeddings_{DATASET_NAME.lower()}"
        train_dir, val_dir = os.path.join(dataset_dir, "train"), os.path.join(dataset_dir, "val")
        train_dataset = AgeEstimationDataset(train_dir, DATASET_NAME, "train")
        val_dataset   = AgeEstimationDataset(val_dir,   DATASET_NAME, "val")

    print(f"📊 Train set: {len(train_dataset)} samples")
    print(f"📊 Validation set: {len(val_dataset)} samples")
    os.makedirs("logs", exist_ok=True)
    with open("logs/train_val_samples", "w") as f:
        f.write(f"📊 Train set: {len(train_dataset)} samples\n")
        f.write(f"📊 Validation set: {len(val_dataset)} samples\n")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)

    in_channels = 128 if (is_ablation and not (enable_lrc or enable_dfe)) else 512
    print(f"📐 Determinazione in_channels: enable_lrc={enable_lrc}, enable_dfe={enable_dfe} → in_channels={in_channels}")

    model = LRA_GNN(
        num_layers=12, num_heads=8,
        in_channels=in_channels, hidden_channels=512, out_channels=1,
        enable_lrc=enable_lrc, enable_dfe=enable_dfe
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    train_loss, val_loss, val_mae, cs5, eps = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=50,
        save_path=checkpoint_path,
        scheduler=scheduler,
        early_stopping_patience=10,
        show_plot=True
    )

    print(f"\n✅ Training LRA-GNN completato per {DATASET_NAME}!")

    # ------ saver storici per-dataset (BACKWARD-COMPATIBLE) ------
    if save_graph_embeddings:
        if DATASET_NAME == "MORPH":
            from utils.save_embeddings_morph import save_embeddings as save_ds_embeddings
        elif DATASET_NAME == "FGNET":
            from utils.save_embeddings_fgnet import save_embeddings as save_ds_embeddings
        elif DATASET_NAME == "UTKFACE":
            from utils.save_embeddings_utkface import save_embeddings as save_ds_embeddings
        elif DATASET_NAME == "CLAP2016":
            from utils.save_embeddings_clap2016 import save_embeddings as save_ds_embeddings
        else:
            raise ValueError(f"❌ Dataset non supportato: {dataset_name}")

        save_ds_embeddings(model, train_loader, device, save_dir=os.path.join(embedding_dir, "train"))
        save_ds_embeddings(model, val_loader,   device, save_dir=os.path.join(embedding_dir, "val"))

    # ------ saver piatto unificato (opzionale, SOLO FULL) ------
    if save_flat and not is_ablation:
        n_tr = save_flat_embeddings(model, train_loader, device, save_dir=os.path.join(embedding_dir, "train"))
        n_va = save_flat_embeddings(model, val_loader,   device, save_dir=os.path.join(embedding_dir, "val"))
        print(f"💾 Embeddings piatti salvati: train={n_tr}, val={n_va}")
    elif save_flat and is_ablation:
        print("ℹ️  --save_flat ignorato in modalità ablation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training LRA-GNN")
    parser.add_argument("--dataset", type=str, default="MORPH",
                        help="Nome del dataset: MORPH, FGNET, UTKFACE o CLAP2016")
    parser.add_argument("--enable_lrc", action="store_true", default=False, help="Abilita il modulo LRC")
    parser.add_argument("--enable_dfe", action="store_true", default=False, help="Abilita il modulo DFE")
    parser.add_argument("--save_flat", action="store_true",
                        help="(FULL only) Salva vettori piatti 512D a livello split/train,val")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--force_ablation", dest="force_ablation", action="store_true",
                       help="Forza la modalità ablation")
    group.add_argument("--no_force_ablation", dest="force_ablation", action="store_false",
                       help="Disattiva la modalità ablation")
    parser.add_argument("--no_save_embeddings", action="store_true",
                        help="Disabilita il salvataggio degli embeddings per-dataset")
    parser.set_defaults(force_ablation=True)

    args = parser.parse_args()

    train_lra_gnn(
        dataset_name=args.dataset,
        enable_lrc=args.enable_lrc,
        enable_dfe=args.enable_dfe,
        force_ablation=args.force_ablation,
        save_graph_embeddings=not args.no_save_embeddings,  # <-- boolean chiaro e senza conflitti
        save_flat=args.save_flat,
    )