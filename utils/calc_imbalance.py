import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.rl_environment import RLEnvironment
import matplotlib.pyplot as plt


# Inizializza l'ambiente
env = RLEnvironment()

# Ottieni i conteggi per ciascun gruppo
counts = env.group_counts  # È già una lista

print(f"Numero totale di campioni: {sum(counts)}")
print(f"Conteggi per riga (classe): {counts}")

# Plot dell'istogramma
plt.hist(counts, bins=20)
plt.title("Distribuzione dei campioni per classe (row)")
plt.xlabel("Numero di campioni per classe")
plt.ylabel("Frequenza")
plt.grid(True)
plt.show()
