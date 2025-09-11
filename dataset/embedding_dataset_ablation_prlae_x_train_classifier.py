import os
import re
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class EmbeddingDatasetPRLAEXClassifier(Dataset):
    def __init__(self, embeddings_dir, dataset_name, encoder, device):
        self.embeddings_dir = embeddings_dir
        self.dataset_name = dataset_name.upper()
        self.encoder = encoder.to(device) if encoder is not None else None
        self.device = device

        # Caricamento età per CLAP2016
        if self.dataset_name == "CLAP2016":
            import pandas as pd
            csv_path = os.path.join(embeddings_dir, "metadata.csv")  # ✅ ora legge da train/metadata.csv o val/metadata.csv
            df = pd.read_csv(csv_path)
            self.age_dict = {row['image'].split('.')[0]: row['mean'] for _, row in df.iterrows()}

        self.samples = []
        self.sample_paths = []
        for entry in os.listdir(embeddings_dir):
            sample_dir = os.path.join(embeddings_dir, entry)
            if not os.path.isdir(sample_dir):
                continue
            if all(os.path.exists(os.path.join(sample_dir, f'graph_lrc_{i}.pt')) for i in range(8)):
                self.samples.append(entry)
                self.sample_paths.append(sample_dir)

        self.samples.sort()
        self.sample_paths.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        sample_dir = self.sample_paths[idx]

        # ✅ Estrai l’età dal NOME del file (non dal .pt)
        if self.dataset_name == "MORPH":
            match = re.search(r'[FfMm](\d{1,3})', sample_id)
            age = int(match.group(1)) if match else -1
        elif self.dataset_name == "FGNET":
            match = re.search(r'[Aa](\d{1,3})', sample_id)
            age = int(match.group(1)) if match else -1
        elif self.dataset_name == "UTKFACE":
            try:
                age = int(sample_id.split('_')[0])
            except:
                age = -1
        elif self.dataset_name == "CLAP2016":
            age = self.age_dict.get(sample_id, -1)
        else:
            raise ValueError(f"Dataset non supportato: {self.dataset_name}")

        if age == -1:
            raise ValueError(f"Età non trovata per: {sample_id}")

        # ↪️ Modalità 1: encoder presente
        if self.encoder is not None:
            embeddings = []
            for i in range(8):
                graph_path = os.path.join(sample_dir, f'graph_lrc_{i}.pt')
                data = torch.load(graph_path).to(self.device)
                with torch.no_grad():
                    emb = self.encoder(data)
                    embeddings.append(emb)
            embedding_finale = torch.stack(embeddings).mean(dim=0)

        # ↪️ Modalità 2: encoder assente → carico embedding ma NON uso .pt["age"]
        else:
            emb_path = sample_dir + ".pt"
            if not os.path.exists(emb_path):
                raise FileNotFoundError(f"❌ File embedding non trovato: {emb_path}")
            embedding_dict = torch.load(emb_path)
            embedding_finale = embedding_dict["embedding"].to(self.device)

        return embedding_finale.cpu(), torch.tensor(age, dtype=torch.float32)