import numpy as np
import random

class SpaceRoombaEnvironment:
    
    def __init__(self, dimension=35, alpha=0.01, k=5):
        self.dimension = dimension
        self.alpha = alpha
        self.k = k
        self.grid = np.full((dimension, dimension), '.', dtype=str)
        self.prob_matrix = np.full((dimension, dimension), 1.0 / (dimension**2))
        self.alien_prob_matrix = np.zeros((dimension, dimension))

        # Initialize positions with placeholders
        self.bot_pos = None
        self.crew_pos = None
        self.alien_pos = None

        # Set positions ensuring no attribute is referenced before it's assigned
        self.bot_pos = self.place_random()
        self.crew_pos = self.place_random(exclude=self.bot_pos)
        self.alien_pos = self.place_random(exclude=[self.bot_pos, self.crew_pos], outside_k=True)
        
        self.update_grid()
        self.update_alien_prob_matrix_initial()

    def place_random(self, exclude=None, outside_k=False):
        if exclude is None:
            exclude = []
        while True:
            pos = (random.randint(0, self.dimension - 1), random.randint(0, self.dimension - 1))
            if pos in exclude:
                continue  # Skip positions that are in the exclude list
            if outside_k:
                if self.distance(pos, self.bot_pos) <= 2 * self.k + 1:
                    continue  # Skip positions within the k radius
            return pos
        
    def distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def update_grid(self):
        self.grid[:, :] = '.'  # Clear grid
        self.grid[self.bot_pos] = 'B'
        self.grid[self.crew_pos] = 'C'
        self.grid[self.alien_pos] = 'A'

    def print_grid(self):
        print('\n'.join([' '.join(row) for row in self.grid]))
        print('\n')

    def sense_crew(self):
        distance = self.distance(self.bot_pos, self.crew_pos)
        prob_of_beep = np.exp(-self.alpha * (distance - 1))
        rand_val = random.random()
        result = (rand_val < prob_of_beep)
        print(f"Distance: {distance}, Prob of Beep: {prob_of_beep}, Result: {result}, Random Value: {rand_val}")
        return result

    def move_alien_randomly(self):
        directions = [(0, -1), (-1, 0), (1, 0), (0, 1)] 
        random.shuffle(directions)
        for dx, dy in directions:
            new_pos = (self.alien_pos[0] + dx, self.alien_pos[1] + dy)
            if 0 <= new_pos[0] < self.dimension and 0 <= new_pos[1] < self.dimension:
                self.alien_pos = new_pos
                break
        self.update_alien_prob_matrix_after_move()
        
    def update_alien_prob_matrix_initial(self):
        # Initially mark only the alien's starting position in the probability matrix
        self.alien_prob_matrix = np.zeros((self.dimension, self.dimension))
        self.alien_prob_matrix[self.alien_pos] = 1.0

    def update_alien_prob_matrix_after_move(self):
        # Spread the probability from the alien's current position to adjacent cells
        temp_matrix = np.zeros((self.dimension, self.dimension))
        for x in range(self.dimension):
            for y in range(self.dimension):
                if self.distance((x, y), self.alien_pos) <= 1:
                    temp_matrix[x, y] = 1.0 / 5  # Adjacent cells + current cell
        self.alien_prob_matrix = temp_matrix

class Bot2(SpaceRoombaEnvironment):
    def run(self):
        steps = 0
        while True:
            beep_detected = self.sense_crew()
            self.update_prob_matrix_after_move(beep_detected)
            self.decide_move_based_on_utility()
            self.move_alien_randomly()
            print(f"Step {steps}:")
            self.print_grid()

            if self.bot_pos == self.crew_pos:
                print(f"Bot 2 rescued the crew member in {steps} steps!")
                break
            elif self.bot_pos == self.alien_pos:
                print(f"Bot 2 was destroyed by the alien after {steps + 1} steps.")
                break
            
            steps += 1

    def update_prob_matrix_after_move(self, beep_detected):
        new_prob_matrix = np.zeros_like(self.prob_matrix)
        
        for x in range(self.dimension):
            for y in range(self.dimension):
                distance = self.distance((x, y), self.bot_pos)
                prob_of_beep = np.exp(-self.alpha * (distance - 1))
                if beep_detected:
                    new_prob_matrix[x, y] = self.prob_matrix[x, y] * prob_of_beep
                else:
                    new_prob_matrix[x, y] = self.prob_matrix[x, y] * (1 - prob_of_beep)
        
        total_prob = np.sum(new_prob_matrix)
        if total_prob > 0:
            self.prob_matrix = new_prob_matrix / total_prob
        else:
            self.prob_matrix = np.full_like(self.prob_matrix, 1.0 / (self.dimension**2))

    def decide_move_based_on_utility(self):
        best_move = None
        best_utility = float('-inf')
        
        for dx, dy in [(0, -1), (-1, 0), (1, 0), (0, 1)]: #Took out staying in place
            next_pos = (self.bot_pos[0] + dx, self.bot_pos[1] + dy)
            if 0 <= next_pos[0] < self.dimension and 0 <= next_pos[1] < self.dimension:
                utility = self.calculate_utility(next_pos)
                if utility > best_utility:
                    best_utility = utility
                    best_move = next_pos

        if best_move:
            self.bot_pos = best_move
            self.update_grid()

    def calculate_utility(self, pos):
        crew_prob = self.prob_matrix[pos]
        alien_prob = self.alien_prob_matrix[pos]
        # Adjust the utility calculation as needed based on your scenario
        utility = crew_prob * 100 - alien_prob * 200  # Example weighting
        return utility

if __name__ == "__main__":
    bot = Bot2(dimension=35, alpha=0.01, k=1)
    bot.run()
