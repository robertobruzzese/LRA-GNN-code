import os, re, torch
from torch_geometric.data import Dataset, Data

class AgeEstimationDatasetUTKFace(Dataset):
    def __init__(self, root_dir, dataset_name="UTKFACE", embedding_split="train",
                 transform=None, enable_lrc=False, enable_dfe=False):
        super().__init__()
        self.dataset_name = dataset_name.upper()
        self.transform = transform
        self.samples = []
        self.enable_lrc = enable_lrc
        self.enable_dfe = enable_dfe
        self.embeddings_dir = root_dir

        def extract_age_from_filename(fname):
            base = fname.replace(".jpg", "").replace(".chip", "")
            m = re.match(r'^(\d+)_', base)
            return int(m.group(1)) if m else None

        for fname in os.listdir(self.embeddings_dir):
            folder = os.path.join(self.embeddings_dir, fname)
            if not os.path.isdir(folder): 
                continue
            age = extract_age_from_filename(fname)
            if age is None:
                print(f"⚠️ Età non trovata in {fname}")
                continue

            # LRC (con o senza DFE): richiede i 8 grafi LRC
            if self.enable_lrc:
                if not all(os.path.exists(os.path.join(folder, f"graph_lrc_{i}.pt")) for i in range(8)):
                    print(f"⚠️ Mancano alcuni grafi LRC in {fname}")
                    continue
                self.samples.append({"image_name": fname, "age": age})
                continue

            # DFE only → usa SEMPRE graph_rw.pt (allineato a CLAP/FGNET)
            if self.enable_dfe:
                graph_path = os.path.join(folder, "graph_rw.pt")
                if os.path.exists(graph_path):
                    self.samples.append({"image_name": fname, "age": age, "graph_path": graph_path})
                else:
                    print(f"⚠️ Skipping {fname}: missing graph_rw.pt")
                continue

            # RW only
            graph_path = os.path.join(folder, "graph_rw.pt")
            if os.path.exists(graph_path):
                self.samples.append({"image_name": fname, "age": age, "graph_path": graph_path})
            else:
                print(f"⚠️ Skipping {fname}: missing graph_rw.pt")

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_name, age = sample["image_name"], sample["age"]

        # LRC only → media delle 8 X
        if self.enable_lrc and not self.enable_dfe:
            xs = []
            for i in range(8):
                g = torch.load(os.path.join(self.embeddings_dir, image_name, f"graph_lrc_{i}.pt"))
                assert isinstance(g, Data) and hasattr(g, "x")
                assert g.x.size(1) == 512, f"x dim {g.x.size(1)}≠512 in graph_lrc_{i}.pt"
                xs.append(g.x)
            x_avg = torch.stack(xs).mean(dim=0)
            return Data(x=x_avg, y=torch.tensor([age], dtype=torch.float), image_name=image_name)

        # LRC + DFE → lista di 8 Data
        if self.enable_lrc and self.enable_dfe:
            lst = []
            for i in range(8):
                g = torch.load(os.path.join(self.embeddings_dir, image_name, f"graph_lrc_{i}.pt"))
                assert isinstance(g, Data) and hasattr(g, "x")
                assert g.x.size(1) == 512, f"x dim {g.x.size(1)}≠512 in graph_lrc_{i}.pt"
                g.y = torch.tensor([age], dtype=torch.float)
                g.image_name = image_name
                lst.append(g)
            return lst

        # DFE only → usa graph_rw.pt (Data con x dim 512)
        if not self.enable_lrc and self.enable_dfe:
            g = torch.load(sample["graph_path"])
            assert isinstance(g, Data) and hasattr(g, "x"), "graph_rw.pt deve essere Data con .x"
            assert g.x.size(1) == 512, f"x dim {g.x.size(1)}≠512 in graph_rw.pt"
            g.y = torch.tensor([age], dtype=torch.float)
            g.image_name = image_name
            return g

        # RW only (GCN base), con riduzione a 128 se serve
        g = torch.load(sample["graph_path"])
        if isinstance(g, Data):
            assert hasattr(g, "x")
            if g.x.size(1) == 512:
                g.x = g.x[:, :128]
            g.y = torch.tensor([age], dtype=torch.float)
            g.image_name = image_name
            return g
        elif isinstance(g, torch.Tensor):
            if g.size(1) == 512:
                g = g[:, :128]
            return Data(x=g, y=torch.tensor([age], dtype=torch.float), image_name=image_name)
        else:
            raise TypeError(f"❌ RW tipo non supportato: {type(g)}")