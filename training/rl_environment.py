import torch
import random
import numpy as np


class RLEnvironment:
    def __init__(self, num_rows=10, num_cols=10):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.group_counts = None
        self.majority_class_count = None
        #self.reset()

    #def reset(self, x=None, target_row=None, target_col=None, imbalance_ratio=None):
    #    self.r = np.random.randint(10)
    #    self.c = np.random.randint(10)
    #    self.x = x
    #    self.target_row = target_row
    #    self.target_col = target_col

        # Se lo ricevi da fuori, lo salvi
    #    self.imbalance_ratio = imbalance_ratio

    #    print(f"[RESET] Start: ({self.r}, {self.c}) → Target: ({self.target_row}, {self.target_col})")

    def reset(self, x, target_row, target_col, imbalance_ratio, actual_age, start_row=None):
        self.r = start_row if start_row is not None else random.randint(0, 9)
        self.c = torch.randint(0, 10, (1,)).item()
        self.target_row = target_row
        self.target_col = target_col
        self.imbalance_ratio = imbalance_ratio
        self.x = x

        # 👇 salva per il calcolo della reward
        self.actual_age = actual_age
        self.age_group = target_row


    def get_state(self):
        return {
            'x': self.x,
            'r': self.r,
            'c': self.c
        }

    def step(self, action):
        # 1. Applica l'azione
        ri, ci = self.decode_action(action)
        self.r = ri
        self.c = ci

        # 2. Calcola il reward
        #reward = self.calculate_reward()

        # 2. Calcola il reward con la nuova formula
        predicted_age = self.r * 10 + self.c
        #reward = self.calculate_reward(predicted_age, self.actual_age, self.imbalance_ratio)
        reward = self.calculate_reward(
            predicted_r=self.r,
            predicted_c=self.c,
            target_r=self.target_row,
            target_c=self.target_col,
            imbalance_ratio=self.imbalance_ratio
        )

        # 3. Calcola nuovo stato (opzionale)
        new_state = (self.r, self.c)

        # 4. Verifica se l’episodio è finito
        done = self.check_done()

        return new_state, reward, done
    
    def check_done(self):
        return self.r == self.target_row and self.c == self.target_col


    def decode_action(self, action):
        if action == 0 and self.r > 0:       # up
            return self.r - 1, self.c
        elif action == 1 and self.r < 9:     # down
            return self.r + 1, self.c
        elif action == 2 and self.c > 0:     # left
            return self.r, self.c - 1
        elif action == 3 and self.c < 9:     # right
            return self.r, self.c + 1
        else:  # stay or move non valida (es. bordo)
            return self.r, self.c


##
#    def calculate_reward(self):
#        gi = self.target_row
#        ei = self.target_col
#        ri = self.r
#        ci = self.c

#        imbalance_ratio = self.imbalance_ratio  # 👈 usa quello già calcolato!

#        distance = abs(ri - gi) + abs(ci - ei)

#        if ri == gi and ci == ei:
#            reward = imbalance_ratio
#            desc = "✅ Correct row and column"
#        elif ri == gi:
#            reward = 1.0 - distance * (imbalance_ratio ** 0.5)
#            desc = "🟡 Correct row, wrong column"
#        else:
#            reward = -distance * imbalance_ratio
#            desc = "❌ Wrong row"

#        print(f"[REWARD] r={ri}, c={ci}, g={gi}, e={ei}")
#        print(f"         Imbalance Ratio: {imbalance_ratio:.4f}")
#        print(f"         Distance: {distance}")
#        print(f"         Reward: {reward:.4f} → {desc}")

#        return reward

    #def calculate_reward(self, predicted_age, actual_age, imbalance_ratio):
  
        # Reward function based on the formula provided in the original paper
    #    distance = abs(predicted_age - actual_age)
        #imbalance_ratio = 1 / (age_group + 1)  # Example imbalance ratio
    #    reward = -distance * imbalance_ratio
    #    return reward
    #def calculate_reward(self, predicted_r, predicted_c, target_r, target_c, imbalance_ratio, penalty_coeff=0.1):
    #    if predicted_r == target_r and predicted_c == target_c:
    #        return +imbalance_ratio
    #    elif predicted_r == target_r and predicted_c != target_c:
    #        return -penalty_coeff * (imbalance_ratio ** 0.5)
    #    else:
    #        return -penalty_coeff * (imbalance_ratio)
        
    def calculate_reward(self, predicted_r, predicted_c, target_r, target_c, imbalance_ratio):
    # Compute distance from ground truth as defined in the paper
        iota = abs(predicted_r - target_r) + abs(predicted_c - target_c)

        if predicted_r == target_r and predicted_c == target_c:
            return +imbalance_ratio  # correct row and column
        elif predicted_r == target_r:
            return -iota * (imbalance_ratio ** 0.5)  # correct row, wrong column
        else:
            return -iota * imbalance_ratio  # wrong row

