# training/rl_environment_prl.py
import random

class RLEnvironment:
    """
    Ambiente PRL 'paper-faithful':
      - Griglia 10x10 (di default)
      - Stato interno: (r, c)
      - Episodio termina se: raggiungi il target OPPURE superi max_steps
      - Reward: sola formula del paper (nessuno shaping extra)
    """
    def __init__(self, num_rows=10, num_cols=10, max_steps=100):
        self.num_rows  = int(num_rows)
        self.num_cols  = int(num_cols)
        self.max_steps = int(max_steps)
        self.t = 0  # step counter

    def reset(self, x, target_row, target_col, imbalance_ratio, actual_age,
              start_row=None, start_col=None):
        # posizione iniziale
        self.r = int(start_row) if start_row is not None else random.randint(0, self.num_rows - 1)
        self.c = int(start_col) if start_col is not None else random.randint(0, self.num_cols - 1)

        # target e meta-dati (usati solo per reward/terminazione)
        self.target_row = int(target_row)
        self.target_col = int(target_col)
        self.imbalance_ratio = float(imbalance_ratio)

        # l'embedding non entra nella reward, ma lo teniamo per compatibilità
        self.x = x
        self.actual_age = float(actual_age)

        # azzera il contatore passi
        self.t = 0

    def get_state(self):
        return {'x': self.x, 'r': self.r, 'c': self.c}

    def decode_action(self, action):
        a = int(action)
        if   a == 0 and self.r > 0:                     # up
            self.r -= 1
        elif a == 1 and self.r < self.num_rows - 1:     # down
            self.r += 1
        elif a == 2 and self.c > 0:                     # left
            self.c -= 1
        elif a == 3 and self.c < self.num_cols - 1:     # right
            self.c += 1
        # a == 4 (stay) o azione invalida → nessun movimento
        return self.r, self.c

    def step(self, action):
        # 1) applica azione
        self.decode_action(action)
        self.t += 1

        # 2) reward (solo formula del paper)
        reward = self.calculate_reward(
            predicted_r=self.r,
            predicted_c=self.c,
            target_r=self.target_row,
            target_c=self.target_col,
            imbalance_ratio=self.imbalance_ratio
        )

        # 3) nuovo stato (opzionale)
        new_state = (self.r, self.c)

        # 4) done: raggiunto target OPPURE max_steps esauriti
        done = self.check_done() or (self.t >= self.max_steps)

        return new_state, reward, done

    def check_done(self):
        return self.r == self.target_row and self.c == self.target_col

    def calculate_reward(self, predicted_r, predicted_c, target_r, target_c, imbalance_ratio):
        # Manhattan distance iota, come nel paper
        iota = abs(predicted_r - target_r) + abs(predicted_c - target_c)
        if predicted_r == target_r and predicted_c == target_c:
            return +imbalance_ratio   # riga & colonna giuste
        elif predicted_r == target_r:
            return - iota * (imbalance_ratio ** 0.5)  # riga giusta, colonna sbagliata
        else:
            return - iota * imbalance_ratio           # riga sbagliata