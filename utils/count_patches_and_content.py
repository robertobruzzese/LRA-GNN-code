import os
import csv

def check_patch_files(patches_dir: str):
    patch_files = [f for f in os.listdir(patches_dir) if f.endswith("_patches.csv")]
    print(f"🔍 Trovati {len(patch_files)} file patch in {patches_dir}")
    total = len(patch_files)
    valid = 0
    empty = 0
    corrupted = 0

    for f in patch_files:
        path = os.path.join(patches_dir, f)
        try:
            with open(path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)
                rows = list(reader)

                if len(rows) == 0:
                    print(f"⚠️ Vuoto: {f}")
                    empty += 1
                else:
                    valid += 1
        except Exception as e:
            print(f"❌ Corrotto o illeggibile: {f} – {str(e)}")
            corrupted += 1

    print("\n📊 Statistiche patch CSV")
    print(f"Totali:         {total}")
    print(f"Valide:         {valid}")
    print(f"Vuote:          {empty}")
    print(f"Corrotte:       {corrupted}")

# ✅ Esegui qui con la tua directory
check_patch_files("datasets/data/UTKFace/images/Train/patches")
check_patch_files("datasets/data/UTKFace/images/Validation/patches")