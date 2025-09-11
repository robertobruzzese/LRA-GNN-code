import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
import pandas as pd
import numpy as np

def regular_data(info_file):
    """Legge un file TXT con il formato: <nome_immagine> <età>."""
    ages = []
    face_frame = []
    with open(info_file, 'r') as fin:
        for line in fin:
            split_content = line.split()
            face_frame.append(split_content[0])  # Nome file immagine
            ages.append(float(split_content[1]))  # Età

    return ages, face_frame

def load_annotations(info_file, dataset_type):
    """Carica le annotazioni in base al formato del dataset (txt, csv, json)."""
    if dataset_type == 'txt':  
        return regular_data(info_file)
    
    elif dataset_type == 'csv':  # Per dataset IMDB-WIKI o simili
        df = pd.read_csv(info_file)
        return df['age'].tolist(), df['image_path'].tolist()
    
    elif dataset_type == 'json':  # Supporto per dataset JSON
        import json
        with open(info_file, 'r') as fin:
            data = json.load(fin)
            ages = [item['age'] for item in data]
            face_frame = [item['image_path'] for item in data]
        return ages, face_frame

    else:
        raise ValueError(f"Formato dataset '{dataset_type}' non supportato")

class AgeEstimationDataset(Dataset):
    """Dataset per la stima dell'età a partire da immagini di volti."""

    def __init__(self, info_file, root_dir, dataset_type='txt', transform=None):
        self.ages, self.face_frame = load_annotations(info_file, dataset_type)
        self.ages = torch.Tensor(self.ages)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """Restituisce il numero totale di immagini nel dataset."""
        return len(self.face_frame)

    def __getitem__(self, idx):
        """Restituisce un'immagine e la relativa età."""
        img_name = os.path.join(self.root_dir, self.face_frame[idx])
        image = Image.open(img_name).convert('RGB')

        age = self.ages[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, age

def load_data(train_batch_size, list_root, pic_root_dir, RANDOM_SEED, val_batch_size=10):
    """Carica i dati e restituisce i DataLoader per training e validazione."""
    
    transform_train_224 = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomCrop([224, 224]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val_224 = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Caricamento del dataset per il training
    transformed_train_dataset = AgeEstimationDataset(
        info_file=list_root + '_train.txt',
        root_dir=pic_root_dir,
        dataset_type='txt',  # Se usi un CSV o JSON, cambia questo parametro
        transform=transform_train_224
    )

    # Caricamento del dataset per la validazione
    transformed_valid_dataset = AgeEstimationDataset(
        info_file=list_root + '_val.txt',
        root_dir=pic_root_dir,
        dataset_type='txt',
        transform=transform_val_224
    )

    # Creazione dei DataLoader
    train_loader = DataLoader(
        transformed_train_dataset, batch_size=train_batch_size,
        shuffle=True, num_workers=4, worker_init_fn=np.random.seed(RANDOM_SEED)
    )

    val_loader = DataLoader(
        transformed_valid_dataset, batch_size=val_batch_size,
        shuffle=False, num_workers=4, worker_init_fn=np.random.seed(RANDOM_SEED)
    )

    return train_loader, val_loader
