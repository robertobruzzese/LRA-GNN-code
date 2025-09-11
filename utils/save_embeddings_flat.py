# utils/save_embeddings_flat.py
import os, sys, torch

CANDIDATES = ("deep_features.pt", "deep_features_from_rw.pt", "node_features.pt")
EMB_DIM = 512

def _load_disk_embedding(sample_dir: str):
    """Carica l'embedding 512D SOLO dai file su disco, in ordine deterministico."""
    for fn in CANDIDATES:
        p = os.path.join(sample_dir, fn)
        if not os.path.exists(p):
            continue
        t = torch.load(p, map_location="cpu")
        if not torch.is_tensor(t):
            continue
        # [N,512] → mean pool
        if t.ndim == 2 and t.shape[1] == EMB_DIM:
            return t.mean(dim=0).to(torch.float32)
        # [512]
        if t.ndim == 1 and t.shape[0] == EMB_DIM:
            return t.to(torch.float32)
    return None

def _normalize_stem(stem):
    """Converte qualunque wrapper in stringa 'ID' e rimuove doppie estensioni (.jpg.chip, .jpg, .png, .pt)."""
    if isinstance(stem, (list, tuple)):
        if not stem:
            return None
        stem = stem[0]
    if torch.is_tensor(stem):
        try:
            stem = stem.item()
        except Exception:
            stem = str(stem)

    stem = os.path.basename(str(stem))

    # rimuovi estensioni note, includendo il caso .jpg.chip
    for ext in (".jpg.chip", ".jpeg.chip", ".png.chip", ".chip", ".jpg", ".jpeg", ".png", ".pt"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break  # rimuoviamo al massimo una voce della lista (quella più specifica)

    return stem

def _image_stem_from_graph(graph):
    """Legge graph.image_name; gestisce Batch (lista interna) o lista di Data."""
    g0 = graph[0] if isinstance(graph, list) else graph
    stem = getattr(g0, "image_name", None)
    if stem is None:
        return None
    return _normalize_stem(stem)

def save_embeddings(model, loader, device, save_dir: str):
    """
    Salva i vettori piatti 512D nello split directory (save_dir).
    NON usa il modello, NON usa graph.x: SOLO file su disco dentro le sottocartelle.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved = skipped = 0

    for graph in loader:
        stem = _image_stem_from_graph(graph)
        if not stem:
            skipped += 1
            print("⚠️  skip: graph senza 'image_name' (o vuoto)", file=sys.stderr)
            continue

        sample_dir = os.path.join(save_dir, stem)  # es: embeddings_*/train/<ID>

        if not os.path.isdir(sample_dir):
            # 🔁 Fallback per dataset con cartelle tipo "<ID>.jpg.chip"
            # Prova varianti comuni
            tried = [sample_dir]
            found = False
            for suffix in (".jpg.chip", ".jpeg.chip", ".png.chip", ".chip"):
                alt = sample_dir + suffix
                tried.append(alt)
                if os.path.isdir(alt):
                    sample_dir = alt
                    found = True
                    break

            # Ultima spiaggia: usa il nome grezzo (senza normalizzazione)
            if not found:
                g0 = graph[0] if isinstance(graph, list) else graph
                raw = os.path.basename(str(getattr(g0, "image_name", "")))
                if raw:
                    alt2 = os.path.join(save_dir, raw)
                    tried.append(alt2)
                    if os.path.isdir(alt2):
                        sample_dir = alt2
                        found = True

            if not found:
                skipped += 1
                print(f"⚠️  skip: cartella assente: {sample_dir} (provati: {', '.join(tried)})", file=sys.stderr)
                continue

        emb = _load_disk_embedding(sample_dir)
        if emb is None or emb.ndim != 1 or emb.shape[0] != EMB_DIM:
            skipped += 1
            print(f"⚠️  skip: deep features non validi in {sample_dir}", file=sys.stderr)
            continue

        out_path = os.path.join(save_dir, f"{stem}.pt")  # es: embeddings_morph/train/<ID>.pt
        torch.save(emb.contiguous(), out_path)
        saved += 1

    print(f"✅ flat-saver: salvati={saved} | skippati={skipped} in {save_dir}")
    return saved