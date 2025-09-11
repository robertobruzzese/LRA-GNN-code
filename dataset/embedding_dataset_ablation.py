import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_dir, return_dict=False, enable_lrc=False):
        self.embeddings_dir = embeddings_dir
        self.return_dict = return_dict
        self.enable_lrc = enable_lrc

        self.samples = []
        for entry in os.listdir(embeddings_dir):
            sample_dir = os.path.join(embeddings_dir, entry)
            if not os.path.isdir(sample_dir):
                continue

            if enable_lrc:
                # Verifica che ci siano tutti gli 8 grafi
                graphs_ok = all(os.path.exists(os.path.join(sample_dir, f'graph_lrc_{i}.pt')) for i in range(8))
                if not graphs_ok:
                    continue
                if not os.path.exists(os.path.join(sample_dir, 'age.pt')):
                    continue
            else:
                if not os.path.exists(os.path.join(sample_dir, 'embedding.pt')) or not os.path.exists(os.path.join(sample_dir, 'age.pt')):
                    continue

            self.samples.append(entry)

        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        sample_dir = os.path.join(self.embeddings_dir, sample_id)

        if self.enable_lrc:
            graph_list = []
            for i in range(8):
                graph = torch.load(os.path.join(sample_dir, f'graph_lrc_{i}.pt'))
                graph_list.append(graph)
            age = torch.load(os.path.join(sample_dir, 'age.pt'))
            return graph_list, torch.tensor(age, dtype=torch.float32)

        else:
            embedding = torch.load(os.path.join(sample_dir, 'embedding.pt'))
            age = torch.load(os.path.join(sample_dir, 'age.pt'))

            if self.return_dict:
                return {
                    'embedding': embedding,
                    'label': torch.tensor(age, dtype=torch.float32)
                }
            else:
                return embedding, torch.tensor(age, dtype=torch.float32)