import os

def check_split(split_dir, split_name, log_dir="logs"):
    complete = 0
    incomplete = 0
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{split_name}_incomplete.log")

    with open(log_path, "w") as log_file:
        for subdir in sorted(os.listdir(split_dir)):
            full_path = os.path.join(split_dir, subdir)
            if os.path.isdir(full_path):
                pt_files = [f for f in os.listdir(full_path) if f.endswith(".pt")]
                if len(pt_files) < 13:
                    log_file.write(f"Incompleto: {subdir} ({len(pt_files)} file)\n")
                    print(f"❌ Incompleto: {subdir} ({len(pt_files)} file)")
                    incomplete += 1
                else:
                    complete += 1

    print(f"\n📂 {split_name.upper()}:")
    print(f"✅ Cartelle complete: {complete}")
    print(f"❌ Cartelle incomplete: {incomplete}")
    print(f"📝 Log salvato in: {log_path}\n")

# === Path dei due split ===
train_dir = "embeddings_clap2016/train"
val_dir = "embeddings_clap2016/val"

# === Check Entrambi ===
check_split(train_dir, "train")
check_split(val_dir, "val")