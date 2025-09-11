import os
import torch
import matplotlib.pyplot as plt
from collections import Counter

embedding_dir = "embeddings_morph/val"
ages = []
genders = []

for filename in os.listdir(embedding_dir):
    if filename.endswith(".pt"):
        path = os.path.join(embedding_dir, filename)
        data = torch.load(path)

        if isinstance(data, dict):  # caso embedding salvato come dict
            age = data.get('age')
            gender = data.get('gender')  # 0 = male, 1 = female (di solito)
            if age is not None:
                ages.append(age)
            if gender is not None:
                genders.append(gender)

# 📊 Età
plt.hist(ages, bins=20, edgecolor='black')
plt.title("Distribuzione delle età (val set - MORPH)")
plt.xlabel("Età")
plt.ylabel("Frequenza")
plt.show()

# 📊 Genere
gender_counter = Counter(genders)
labels = ['Maschi', 'Femmine']
counts = [gender_counter.get(0, 0), gender_counter.get(1, 0)]

plt.bar(labels, counts, color=['blue', 'pink'])
plt.title("Distribuzione del genere (val set - MORPH)")
plt.ylabel("Numero di campioni")
plt.show()