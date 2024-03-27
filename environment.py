import numpy as np
import random

class SpaceRoombaEnvironment:
    def __init__(self, dimension=35, alpha=0.5):
        self.dimension = dimension
        self.alpha = alpha
        self.grid = np.full((dimension, dimension), '.', dtype=str)
        self.prob_matrix = np.full((dimension, dimension), 1.0 / (dimension * dimension))
        self.bot_pos = None
        self.crew_pos = None
        self.alien_pos = None
        self.bot_pos = self.place_random()
        self.crew_pos = self.place_random(exclude=self.bot_pos)
        self.alien_pos = self.place_random(exclude=[self.bot_pos, self.crew_pos])
        self.update_grid()

    def place_random(self, exclude=None):
        if exclude is None:
            exclude = []
        elif not isinstance(exclude, list):
            exclude = [exclude]  # Convert single item to list
        while True:
            pos = (random.randint(0, self.dimension-1), random.randint(0, self.dimension-1))
            if pos not in exclude:
                return pos

    def distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def update_grid(self):
        self.grid[:, :] = '.'  # Clear grid
        self.grid[self.bot_pos] = 'B'  # Mark bot
        self.grid[self.crew_pos] = 'C'  # Mark crew member
        self.grid[self.alien_pos] = 'A'  # Mark alien

    def print_grid(self):
        for row in self.grid:
            print(' '.join(row))
        print('\n')

    def sense_crew(self):
        distance = self.distance(self.bot_pos, self.crew_pos)
        prob_of_beep = np.exp(-self.alpha * (distance - 1))
        return random.random() < prob_of_beep

    def update_prob_matrix_after_move(self):
        beep_detected = self.sense_crew()
        new_prob_matrix = np.zeros_like(self.prob_matrix)
        
        for x in range(self.dimension):
            for y in range(self.dimension):
                distance = abs(x - self.bot_pos[0]) + abs(y - self.bot_pos[1])
                prob_of_beep_given_pos = np.exp(-self.alpha * (distance - 1))
                
                if beep_detected:
                    # Increase probability based on closeness if beep is detected
                    new_prob_matrix[x, y] = self.prob_matrix[x, y] * prob_of_beep_given_pos
                else:
                    # Decrease probability for closer positions if no beep is detected
                    new_prob_matrix[x, y] = self.prob_matrix[x, y] * (1 - prob_of_beep_given_pos)

        # Normalize the updated probabilities
        total_prob = np.sum(new_prob_matrix)
        if total_prob > 0:
            self.prob_matrix = new_prob_matrix / total_prob
        else:
            # Handle edge case where total_prob is 0 to avoid division by zero
            self.prob_matrix = np.full_like(self.prob_matrix, 1.0 / (self.dimension * self.dimension))

    def move_alien_randomly(self):
        directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
        random.shuffle(directions)
        for dx, dy in directions:
            new_pos = (self.alien_pos[0] + dx, self.alien_pos[1] + dy)
            if 0 <= new_pos[0] < self.dimension and 0 <= new_pos[1] < self.dimension:
                self.alien_pos = new_pos
                break
        self.update_grid()
