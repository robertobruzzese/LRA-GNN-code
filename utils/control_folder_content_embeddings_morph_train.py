
import torch, os
folder = 'embeddings/train/052229_1M43'  # 🔁 cambia con la tua cartella

required_files = [
    'graph_initial.pt',
    'graph_rw.pt',
    'patches_tensor.pt',
    'patch_to_node.pt',
    'deep_features.pt'
]

for fname in required_files:
    fpath = os.path.join(folder, fname)
    if not os.path.exists(fpath):
        print(f'❌ {fname} MANCANTE')
        continue
    try:
        obj = torch.load(fpath)
        if isinstance(obj, torch.Tensor):
            print(f'✅ {fname}: Tensor shape {tuple(obj.shape)}')
        elif hasattr(obj, \"x\"):
            print(f'✅ {fname}: Data with {obj.x.shape[0]} nodes, {obj.edge_index.shape[1]} edges')
        else:
            print(f'✅ {fname}: Oggetto tipo {type(obj)}')
    except Exception as e:
        print(f'❌ {fname}: Errore di lettura ({e})')
"