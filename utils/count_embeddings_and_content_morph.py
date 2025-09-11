import os

def check_embedding_folders(root_dir: str, required_files=None):
    if required_files is None:
        required_files = [
            "graph_initial.pt",
            "graph_rw.pt",
            "patches_tensor.pt",
            "patch_to_node.pt",
            "deep_features.pt"
        ]

    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    total = len(subfolders)
    complete = 0
    missing = 0
    empty = 0

    print(f"🔍 Trovate {total} sottocartelle in {root_dir}\n")

    for folder in subfolders:
        folder_path = os.path.join(root_dir, folder)
        problems = []

        for filename in required_files:
            file_path = os.path.join(folder_path, filename)
            if not os.path.exists(file_path):
                problems.append(f"❌ Manca {filename}")
            elif os.path.getsize(file_path) == 0:
                problems.append(f"⚠️ Vuoto {filename}")

        if not problems:
            complete += 1
        else:
            missing += 1
            print(f"📂 {folder}:")
            for p in problems:
                print(f"   {p}")
            print()

    print("\n📊 Statistiche embedding:")
    print(f"Totali cartelle: {total}")
    print(f"Complete:        {complete}")
    print(f"Incomplete:      {missing}")
    print(f"(tra cui con file vuoti): controllato sopra")

# ✅ Esegui qui
check_embedding_folders("embeddings_morph/train")
check_embedding_folders("embeddings_morph/val")