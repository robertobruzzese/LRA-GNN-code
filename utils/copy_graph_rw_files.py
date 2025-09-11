import os
import shutil
from tqdm import tqdm

def copy_graph_rw_files(split):
    src_base = f"embeddings_morph/{split}"
    dst_base = f"embeddings_ablation_morph_no_lrc_dfe/{split}"

    if not os.path.exists(src_base):
        print(f"❌ Origine non trovata: {src_base}")
        return
    if not os.path.exists(dst_base):
        print(f"❌ Destinazione non trovata: {dst_base}")
        return

    count = 0
    for sample_id in tqdm(os.listdir(src_base), desc=f"📦 Copia {split}"):
        src_dir = os.path.join(src_base, sample_id)
        dst_dir = os.path.join(dst_base, sample_id)

        # ✅ Salta se non è una cartella
        if not os.path.isdir(src_dir):
            continue

        src_file = os.path.join(src_dir, "graph_rw.pt")
        dst_file = os.path.join(dst_dir, "graph_rw.pt")

        if os.path.exists(src_file) and os.path.isdir(dst_dir):
            shutil.copy(src_file, dst_file)
            count += 1
        else:
            print(f"⚠️ Skipped: {sample_id} (file o cartella mancante)")

    print(f"✅ Copiati {count} file 'graph_rw.pt' in {split}")

if __name__ == "__main__":
    copy_graph_rw_files("train")
    copy_graph_rw_files("val")