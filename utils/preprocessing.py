import cv2
import numpy as np
from mtcnn import MTCNN
import os

def preprocess_image(image_path, save_path):
    """Rileva il volto, lo allinea e lo ridimensiona a 224×224 px."""
    
    detector = MTCNN()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(image)
    if not results:
        raise ValueError(f"❌ Nessun volto rilevato in {image_path}!")

    # 📌 Estrarre bounding box e landmark
    face = results[0]
    x, y, w, h = face["box"]
    keypoints = face["keypoints"]

    left_eye, right_eye = keypoints["left_eye"], keypoints["right_eye"]

    # 📌 Calcola angolo di rotazione per allineare gli occhi
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # 📌 Centro del volto per la rotazione
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # 📌 Ruota l'immagine
    M = cv2.getRotationMatrix2D(center, angle, 1)
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    # 📌 Ritaglia il volto
    cropped_face = aligned_image[y:y+h, x:x+w]

    # 📌 Ridimensiona a 224 × 224 px
    resized_face = cv2.resize(cropped_face, (224, 224))

    # 📌 Salva l'immagine preprocessata
    cv2.imwrite(save_path, cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR))
    abs_path = os.path.abspath(save_path)
    cv2.imwrite(save_path, cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR))
    print(f"✅ Immagine preprocessata salvata in: {abs_path}")

    print(f"✅ Immagine preprocessata salvata in {save_path}")
