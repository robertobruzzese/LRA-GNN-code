import argparse
import os
import cv2
import numpy as np
import face_alignment

# --- Argomenti ---
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True, help='Percorso immagine input')
parser.add_argument('--output', type=str, required=True, help='Percorso file .pts output')
args = parser.parse_args()

# --- Landmark ridotti ---
KEY_LANDMARKS = [
    30, 31, 35, 36, 39, 42, 45, 48, 54, 62,
    27, 28, 29, 32, 33, 34, 37, 38, 40, 41,
    43, 44, 46, 47, 49, 50, 51, 52, 53, 55, 56, 57
]

# --- Configurazione ---
TARGET_SIZE = (224, 224)
import torch

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

# --- Carica immagine ---
image = cv2.imread(args.image)
if image is None:
    print(f"❌ Immagine non trovata: {args.image}")
    exit(1)

# --- Estrai landmark ---
landmarks_all = fa.get_landmarks(image)
if landmarks_all is None:
    print(f"⚠️ Nessun volto trovato in: {args.image}")
    exit(2)

landmarks = np.array(landmarks_all[0])
h, w = image.shape[:2]
landmarks[:, 0] = (landmarks[:, 0] / w) * TARGET_SIZE[0]
landmarks[:, 1] = (landmarks[:, 1] / h) * TARGET_SIZE[1]

# --- Landmark ridotti ---
reduced = landmarks[KEY_LANDMARKS]

# --- Salva su file .pts ---
with open(args.output, "w") as f:
    f.write(f"version: 1\nn_points: {len(reduced)}\n{{\n")
    for x, y in reduced:
        f.write(f"{x:.2f} {y:.2f}\n")
    f.write("}\n")

print(f"✅ Landmark salvati in {args.output}")