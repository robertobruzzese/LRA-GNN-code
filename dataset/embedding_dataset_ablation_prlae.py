import os
import re
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_dir, dataset_name, return_dict=False, enable_lrc=False, enable_dfe=False):
        self.embeddings_dir = embeddings_dir
        self.dataset_name = dataset_name.upper()
        self.return_dict = return_dict
        self.enable_lrc = enable_lrc
        self.enable_dfe = enable_dfe

        assert (enable_lrc != enable_dfe), "❌ Specificare esattamente uno tra enable_lrc e enable_dfe"

        # Solo per CLAP2016: carica il dizionario età da CSV
        if self.dataset_name == "CLAP2016":
            csv_path = os.path.join(embeddings_dir, "metadata.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"❌ CSV mancante per CLAP2016: {csv_path}")
            df = pd.read_csv(csv_path)
            #self.age_dict = {row['filename']: row['mean'] for _, row in df.iterrows()}
            #self.age_dict = {row['image']: row['mean'] for _, row in df.iterrows()}
            self.age_dict = {row['image'].split('.')[0]: row['mean'] for _, row in df.iterrows()}

        # Lista delle sottocartelle (una per immagine)
        self.samples = []
        self.sample_paths = []  # ✅ aggiunto
        for entry in os.listdir(embeddings_dir):
            sample_dir = os.path.join(embeddings_dir, entry)
            if not os.path.isdir(sample_dir):
                continue

            if enable_lrc:
                graphs_ok = all(os.path.exists(os.path.join(sample_dir, f'graph_lrc_{i}.pt')) for i in range(8))
                if not graphs_ok:
                    continue
            elif enable_dfe:
                # ✅ Verifica presenza di almeno un file .pt
                pt_files = [f for f in os.listdir(sample_dir) if f.endswith('.pt')]
                if len(pt_files) == 0:
                    continue

            self.samples.append(entry)
            self.sample_paths.append(sample_dir)  # ✅ aggiunto

        self.samples.sort()
        self.sample_paths.sort()  # ✅ ordina i path in parallelo ai sample_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        sample_dir = self.sample_paths[idx]

        # 🧠 Estrazione età in base al dataset
        if self.dataset_name == "MORPH":
            age_match = re.search(r'[FfMm](\d{1,3})', sample_id)
            if not age_match:
                raise ValueError(f"❌ Età non trovata nel nome: {sample_id}")
            age = int(age_match.group(1))

        elif self.dataset_name == "FGNET":
            age_match = re.search(r'[Aa](\d{1,3})', sample_id)
            if not age_match:
                raise ValueError(f"❌ Età non trovata nel nome: {sample_id}")
            age = int(age_match.group(1))

        elif self.dataset_name == "UTKFACE":
            try:
                age = int(sample_id.split('_')[0])
            except:
                raise ValueError(f"❌ Formato non valido per UTKFACE: {sample_id}")

        elif self.dataset_name == "CLAP2016":
            if sample_id not in self.age_dict:
                raise ValueError(f"❌ Età non trovata in CSV per: {sample_id}")
            age = self.age_dict[sample_id]

        else:
            raise ValueError(f"❌ Dataset non supportato: {self.dataset_name}")

        # 🔁 Caso PRLAE + LRC → lista di 8 grafi

        if self.enable_lrc:
            graph_list = [torch.load(os.path.join(sample_dir, f"graph_lrc_{i}.pt")) for i in range(8)]
            
            # Verifica coerenza topologia
            edge_index_ref = graph_list[0].edge_index
            if not all(torch.equal(g.edge_index, edge_index_ref) for g in graph_list):
                raise ValueError(f"❌ edge_index diversi tra le 8 head per {sample_id}")

            # Media tra le 8 head (USA I TENSORI)
            #mean_x = torch.stack([g.x for g in graph_list]).mean(dim=0)
            #mean_edge_attr = torch.stack([g.edge_attr for g in graph_list]).mean(dim=0)
            #edge_index = edge_index_ref  # identico per tutte

            # Costruisci grafo medio
            #mean_graph = Data(x=mean_x, edge_index=edge_index, edge_attr=mean_edge_attr)
            #mean_embedding = mean_graph.x.mean(dim=0)  # (512,) embedding globale per PRLAE
            #return mean_embedding, torch.tensor(age, dtype=torch.float32)
            # Media tra le 8 head (USA I GRAFI)
            mean_x = torch.stack([g.x for g in graph_list]).mean(dim=0)
            mean_edge_attr = torch.stack([g.edge_attr for g in graph_list]).mean(dim=0)
            edge_index = edge_index_ref  # identico per tutte

            # Costruisci grafo medio con età come target
            mean_graph = Data(
                x=mean_x,
                edge_index=edge_index,
                edge_attr=mean_edge_attr,
                y=torch.tensor(age, dtype=torch.float32)
            )
            return mean_graph
        # 🔁 Caso PRLAE + DFE → embedding.pt
        elif self.enable_dfe:
            deep_features_path = os.path.join(sample_dir, "deep_features.pt")

            if not os.path.exists(deep_features_path):
                raise FileNotFoundError(f"❌ deep_features.pt mancante in {sample_dir}")

            # 🎯 Se in evaluation (return_dict=True), restituisci solo embedding e age
            if self.return_dict:
                embedding = torch.load(deep_features_path)
                return {
                    "embedding": embedding,
                    "age": torch.tensor(age, dtype=torch.float32),
                }

            # 🧱 Altrimenti, in fase di training, serve anche il grafo
            graph_path = os.path.join(sample_dir, "graph_rw.pt")

            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"❌ graph_rw.pt mancante in {sample_dir}")

            x = torch.load(deep_features_path)  # (N, 512)
            graph = torch.load(graph_path)      # contiene edge_index, edge_attr

            if not isinstance(x, torch.Tensor):
                raise ValueError(f"❌ deep_features deve essere un tensore, ma è {type(x)}")
            if not isinstance(graph, Data):
                raise ValueError(f"❌ graph_rw.pt deve essere un oggetto torch_geometric.data.Data")

            # 🔧 Costruzione del grafo finale
            data = Data(
                x=x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr if 'edge_attr' in graph else None,
                y=torch.tensor(age, dtype=torch.float32)
            )

            return data
        else:
            raise ValueError("❌ Specificare almeno uno tra enable_lrc o enable_dfe") 