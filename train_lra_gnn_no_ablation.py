#train_lra_gnn_no_ablation.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from utils.save_embeddings_morph import save_embeddings
from models.lra_gnn import LRA_GNN
from training.train_model import train_model
from dataset.age_estimation_dataset import AgeEstimationDataset
from dataset.age_estimation_dataset_fgnet import AgeEstimationDatasetFGNET
from dataset.age_estimation_dataset_morph import AgeEstimationDatasetMorph
from dataset.age_estimation_dataset_utkface import AgeEstimationDatasetUTKFace
from dataset.age_estimation_dataset_clap2016 import AgeEstimationDatasetClap2016

def train_lra_gnn(
    dataset_name="MORPH",
    enable_lrc=False,
    enable_dfe=False,
    force_ablation=True,
    embeddings_dir=None,
    checkpoints_dir=None,
    use_prlae=False,
    ablation=False
):
    print("🧩 train_lra_gnn - FLAGS:")
    print(f"    enable_lrc = {enable_lrc}")
    print(f"    enable_dfe = {enable_dfe}")

    exp_name_parts = []
    if enable_lrc:
        exp_name_parts.append("lrc")
    if enable_dfe:
        exp_name_parts.append("dfe")
    exp_name = "_".join(exp_name_parts) if exp_name_parts else "gcn_only"

    is_ablation = force_ablation or enable_lrc or enable_dfe

    DATASET_NAME = dataset_name.upper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📁 DATASET_NAME: {DATASET_NAME}")
    print(f"⚙️  Esperimento: {exp_name}")

    if embeddings_dir is None or checkpoints_dir is None:
        if is_ablation:
            checkpoint_dir = os.path.join("checkpoints_ablation", DATASET_NAME.lower(), exp_name)
            embedding_dir = f"embeddings_ablation_{DATASET_NAME.lower()}_{exp_name}"
        else:
            checkpoint_dir = os.path.join("checkpoints", DATASET_NAME)
            embedding_dir = f"embeddings_{DATASET_NAME.lower()}"
    else:
        checkpoint_dir = checkpoints_dir
        embedding_dir = embeddings_dir

    os.makedirs(os.path.join(embedding_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(embedding_dir, "val"), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_lra_gnn_{DATASET_NAME.lower()}.pth")

    if is_ablation:
        train_dir = os.path.join(embedding_dir, "train")
        val_dir = os.path.join(embedding_dir, "val")

        if dataset_name == "UTKFACE":
            train_dataset = AgeEstimationDatasetUTKFace(
                root_dir=train_dir,
                dataset_name=DATASET_NAME,
                embedding_split="train",
                enable_lrc=enable_lrc,
                enable_dfe=enable_dfe
            )

            val_dataset = AgeEstimationDatasetUTKFace(
                root_dir=val_dir,
                dataset_name=DATASET_NAME,
                embedding_split="val",
                enable_lrc=enable_lrc,
                enable_dfe=enable_dfe
            )
        elif dataset_name == "CLAP2016":
            train_dataset = AgeEstimationDatasetClap2016(
                root_dir=train_dir,
                dataset_name=DATASET_NAME,
                embedding_split="train",
                enable_lrc=enable_lrc,
                enable_dfe=enable_dfe
            )
            val_dataset = AgeEstimationDatasetClap2016(
                root_dir=val_dir,
                dataset_name=DATASET_NAME,
                embedding_split="val",
                enable_lrc=enable_lrc,
                enable_dfe=enable_dfe
            )
        elif dataset_name == "FGNET":
            train_dataset = AgeEstimationDatasetFGNET(
                root_dir=train_dir,
                dataset_name=DATASET_NAME,
                embedding_split="train",
                enable_lrc=enable_lrc,
                enable_dfe=enable_dfe
            )
            val_dataset = AgeEstimationDatasetFGNET(
                root_dir=val_dir,
                dataset_name=DATASET_NAME,
                embedding_split="val",
                enable_lrc=enable_lrc,
                enable_dfe=enable_dfe
            )
        elif dataset_name == "MORPH":
           train_dataset = AgeEstimationDatasetMorph(
                root_dir=train_dir,
                dataset_name=DATASET_NAME,
                embedding_split="train",
                enable_lrc=enable_lrc,
                enable_dfe=enable_dfe
            )
           val_dataset = AgeEstimationDatasetMorph(
                root_dir=val_dir,
                dataset_name=DATASET_NAME,
                embedding_split="val",
                enable_lrc=enable_lrc,
                enable_dfe=enable_dfe
            )
    else:
        if DATASET_NAME in ["UTKFACE", "CLAP2016", "FGNET"]:
            dataset_dir = f"embeddings_{DATASET_NAME.lower()}"
            train_dir = os.path.join(dataset_dir, "train")
            val_dir = os.path.join(dataset_dir, "val")

            train_dataset = AgeEstimationDataset(train_dir, DATASET_NAME, "train")
            val_dataset = AgeEstimationDataset(val_dir, DATASET_NAME, "val")
        else:  # MORPH completo da immagini
            data_path = f"datasets/data/{DATASET_NAME}"
            dataset = AgeEstimationDataset(
                root_dir=data_path,
                dataset_name=DATASET_NAME,
                enable_lrc=enable_lrc,
                enable_dfe=enable_dfe
            )
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_len = len(train_dataset)
    val_len = len(val_dataset)

    print(f"📊 Train set: {train_len} samples")
    print(f"📊 Validation set: {val_len} samples")
    os.makedirs("logs", exist_ok=True)
    with open("logs/train_val_samples", "w") as f:
        f.write(f"📊 Train set: {train_len} samples\n")
        f.write(f"📊 Validation set: {val_len} samples\n")

    #train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    in_channels = 128 if not (enable_lrc or enable_dfe) else 512
    print(f"📐 Determinazione in_channels: enable_lrc={enable_lrc}, enable_dfe={enable_dfe} → in_channels={in_channels}")

    model = LRA_GNN(
        num_layers=12,
        num_heads=8,
        in_channels=in_channels,
        hidden_channels=512,
        out_channels=1,
        enable_lrc=enable_lrc,
        enable_dfe=enable_dfe
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

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

    if dataset_name.lower() == "morph":
        from utils.save_embeddings_morph import save_embeddings
        save_embeddings(model, train_loader, device, save_dir=os.path.join(embedding_dir, "train"))
        save_embeddings(model, val_loader, device, save_dir=os.path.join(embedding_dir, "val"))

    elif dataset_name.lower() == "fgnet":
        from utils.save_embeddings_fgnet import save_embeddings
        save_embeddings(model, train_loader, device, save_dir=os.path.join(embedding_dir, "train"))
        save_embeddings(model, val_loader, device, save_dir=os.path.join(embedding_dir, "val"))

    elif dataset_name.lower() == "utkface":
        from utils.save_embeddings_utkface import save_embeddings
        save_embeddings(model, train_loader, device, save_dir=os.path.join(embedding_dir, "train"))
        save_embeddings(model, val_loader, device, save_dir=os.path.join(embedding_dir, "val"))

    elif dataset_name.lower() == "clap2016":
        from utils.save_embeddings_clap2016 import save_embeddings
        save_embeddings(model, train_loader, device, save_dir=os.path.join(embedding_dir, "train"))
        save_embeddings(model, val_loader, device, save_dir=os.path.join(embedding_dir, "val"))

    else:
        raise ValueError(f"❌ Dataset non supportato: {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training LRA-GNN")
    parser.add_argument("--dataset", type=str, default="MORPH", help="Nome del dataset: MORPH, FGNET, UTKFACE o CLAP2016")
    parser.add_argument("--enable_lrc", action="store_true", default=False, help="Abilita il modulo LRC")
    parser.add_argument("--enable_dfe", action="store_true", default=False, help="Abilita il modulo DFE")
    args = parser.parse_args()

    train_lra_gnn(dataset_name=args.dataset, enable_lrc=args.enable_lrc, enable_dfe=args.enable_dfe)