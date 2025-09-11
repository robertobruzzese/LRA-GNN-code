import os
import torch
import pandas as pd
import re
from torch_geometric.data import Dataset, Data

class AgeEstimationDatasetClap2016(Dataset):
    def __init__(self, root_dir, dataset_name="MORPH", embedding_split="train", transform=None,
                 enable_lrc=False, enable_dfe=False):
        super().__init__()
        self.dataset_name = dataset_name.upper()
        self.transform = transform
        self.samples = []
        self.enable_lrc = enable_lrc
        self.enable_dfe = enable_dfe

        if self.dataset_name == "MORPH":
            self.embeddings_dir = os.path.join("embeddings_morph", embedding_split)
            age_regex = r'[MF](\d{1,3})'
            csv_df = None

        elif self.dataset_name == "FGNET":
            self.embeddings_dir = root_dir
            age_regex = r'A(\d{1,3})'
            csv_df = None

        elif self.dataset_name == "UTKFACE":
            self.embeddings_dir = root_dir
            csv_df = None
            def extract_age_from_filename(fname):
                # Rimuovi eventuali estensioni e chip
                base = fname.replace(".jpg", "").replace(".chip", "")
                match = re.match(r'^(\d+)_', base)
                return int(match.group(1)) if match else None

        elif self.dataset_name == "CLAP2016":
            self.embeddings_dir = root_dir
            csv_path = os.path.join("datasets", "data", "CLAP2016", f"CLAP_complete_{embedding_split}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"❌ CSV non trovato: {csv_path}")
            csv_df = pd.read_csv(csv_path)
            #csv_df["image"] = csv_df["image"].str.replace(".jpg", "", regex=False)
            image_to_age = dict(zip(csv_df["image"], csv_df["mean"]))
            print(f"📌 Esempio chiavi CSV: {list(image_to_age.keys())[:10]}")
        else:
            raise ValueError(f"Dataset sconosciuto: {self.dataset_name}")

        for fname in os.listdir(self.embeddings_dir):
            folder_path = os.path.join(self.embeddings_dir, fname)
            if not os.path.isdir(folder_path):
                continue

            try:
                if self.dataset_name in ["MORPH", "FGNET"]:
                    match = re.search(age_regex, fname)
                    if not match:
                        print(f"⚠️ Età non trovata in {fname}")
                        continue
                    age = int(match.group(1))
                    if age is None:
                        print(f"Ignoro {fname} per età non parsabile")
                        continue

                elif self.dataset_name == "CLAP2016":
                    #image_name_jpg = fname + ".jpg"
                    #if image_name_jpg not in image_to_age:
                    if fname + ".jpg" not in image_to_age:
                        print(f"⚠️ Età non trovata per {fname}, salto.")
                        continue
                    #age = float(image_to_age[image_name_jpg])
                    age = float(image_to_age[fname + ".jpg"])

                if self.enable_lrc:
                    all_exist = all(os.path.exists(os.path.join(folder_path, f"graph_lrc_{i}.pt")) for i in range(8))
                    if not all_exist:
                        print(f"⚠️ Mancano alcuni grafi LRC in {fname}")
                        continue
                    self.samples.append({"image_name": fname, "age": age, "graph_path": None})
                    continue

                if self.enable_dfe:
                    graph_path = os.path.join(folder_path, "deep_features.pt")
                else:
                    graph_path = os.path.join(folder_path, "graph_rw.pt")

                if os.path.exists(graph_path):
                    try:
                        graph = torch.load(graph_path)
                        if isinstance(graph, Data):
                            if graph.edge_index.size(0) == 2 and graph.edge_index.size(1) > 0:
                                if not self.enable_lrc and not self.enable_dfe and graph.x.size(1) == 512:
                                    graph.x = graph.x[:, :128]
                                self.samples.append({"image_name": fname, "age": age, "graph_path": graph_path})
                            else:
                                print(f"⚠️ Ignoro grafo vuoto: {fname}")
                                os.makedirs("logs", exist_ok=True)
                                with open("logs/empty_graphs.log", "a") as f:
                                    f.write(f"{fname}\n")
                        elif isinstance(graph, torch.Tensor):
                            if not self.enable_dfe and graph.size(1) == 512:
                                graph = graph[:, :128]
                            self.samples.append({"image_name": fname, "age": age, "graph_path": graph_path})
                        else:
                            print(f"⚠️ Tipo file non gestito in {fname}: {type(graph)}")

                    except Exception as e:
                        print(f"❌ Errore caricamento grafo {fname}: {e}")
                else:
                    print(f"⚠️ Skipping {fname}: missing file {graph_path}")

            except Exception as e:
                print(f"⚠️ Errore parsing {fname}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_name = sample["image_name"]
        age = sample["age"]

        # === Caso 1: LRC only (usa media degli 8 grafi)
        if self.enable_lrc and not self.enable_dfe:
            graphs = []
            for i in range(8):
                g_path = os.path.join(self.embeddings_dir, image_name, f"graph_lrc_{i}.pt")
                g = torch.load(g_path)
                if isinstance(g, Data) and hasattr(g, "x"):
                    assert g.x.size(1) == 512, f"❌ graph_lrc_{i}.pt ha x.shape[1] = {g.x.size(1)} invece di 512"
                    graphs.append(g.x)
                else:
                    raise TypeError(f"❌ Il file {g_path} non è un oggetto Data valido.")
            x_avg = torch.stack(graphs).mean(dim=0)
            return Data(x=x_avg, y=torch.tensor([age], dtype=torch.float), image_name=image_name)

        # === Caso 2: LRC + DFE (usa solo un grafo, es. graph_lrc_1.pt)
        elif self.enable_lrc and self.enable_dfe:
            graph_list = []
            for i in range(8):
                g_path = os.path.join(self.embeddings_dir, image_name, f"graph_lrc_{i}.pt")
                g = torch.load(g_path)
                if isinstance(g, Data) and hasattr(g, "x"):
                    assert g.x.size(1) == 512, f"❌ graph_lrc_{i}.pt ha x.shape[1] = {g.x.size(1)} invece di 512"
                    g.y = torch.tensor([age], dtype=torch.float)
                    g.image_name = image_name
                    graph_list.append(g)
                else:
                    raise TypeError(f"❌ Il file {g_path} non è un oggetto Data valido.")
            return graph_list

# === Caso 3: DFE only (usa direttamente graph_rw.pt)
        elif not self.enable_lrc and self.enable_dfe:
            graph_path = os.path.join(self.embeddings_dir, image_name, "graph_rw.pt")
            graph = torch.load(graph_path)

            assert isinstance(graph, Data), f"❌ graph_rw.pt deve essere un oggetto Data, trovato: {type(graph)}"
            assert hasattr(graph, "x"), f"❌ Il grafo non contiene il campo x"
            assert graph.x.size(1) == 512, f"❌ Dimensione delle feature errata: trovato {graph.x.size(1)}, atteso 512"

            graph.y = torch.tensor([age], dtype=torch.float)
            graph.image_name = image_name
            return graph
        # === Caso 4: GCN only (RW), con riduzione a 128
        else:
            g_path = sample["graph_path"]
            g = torch.load(g_path)

            if isinstance(g, Data):
                if hasattr(g, "x") and isinstance(g.x, torch.Tensor):
                    if g.x.size(1) == 512:
                        g.x = g.x[:, :128]
                    g.y = torch.tensor([age], dtype=torch.float)
                    g.image_name = image_name
                    return g
                else:
                    raise TypeError(f"❌ RW Data senza x valido per {image_name}")
            elif isinstance(g, torch.Tensor):
                if g.size(1) == 512:
                    g = g[:, :128]
                return Data(x=g, y=torch.tensor([age], dtype=torch.float), image_name=image_name)
            else:
                raise TypeError(f"❌ RW tipo non supportato: {type(g)}")