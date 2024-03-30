import random
import numpy as np

class SpaceRoombaEnvironment:
    
    def __init__(self, dimension=35, alpha=0.01, k=5):
        self.dimension = dimension
        self.alpha = alpha
        self.k = k
        self.grid = np.full((dimension, dimension), '.', dtype=str)
        self.prob_matrix = np.full((dimension, dimension), 1.0 / (dimension**2))
        self.alien_prob_matrix = np.zeros((dimension, dimension))
        self.crew_prob_matrix = np.full((dimension, dimension), 1 / (dimension**2 - 1))
        self.alien_prob_matrix = np.full((dimension, dimension), 1 / (dimension**2 - (2*k + 1)**2))
        

        # Initialize positions with placeholders
        self.bot_pos = None
        self.crew_pos = None
        self.alien_pos = None

        # Set positions ensuring no attribute is referenced before it's assigned
        self.bot_pos = self.place_random()
        self.crew_pos = self.place_random(exclude=self.bot_pos)
        self.alien_pos = self.place_random(exclude=[self.bot_pos, self.crew_pos], outside_k=True)
        
        
        self.update_grid()
        self.update_prob_matrices()
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


class Bot1(SpaceRoombaEnvironment):
    
    def sense_environment(self):
        beep_detected = random.random() < np.exp(-self.alpha * (self.distance(self.bot_pos, self.crew_pos) - 1))
        alien_sensed = self.distance(self.bot_pos, self.alien_pos) <= (2 * self.k + 1)
        return beep_detected, alien_sensed
    
    def update_prob_matrices(self, beep_detected, alien_sensed):
        # Update the crew member probability matrix based on beep detection
        for x in range(self.dimension):
            for y in range(self.dimension):
                distance = self.distance((x, y), self.bot_pos)
                prob_of_beep = np.exp(-self.alpha * (distance - 1))
                if beep_detected:
                    self.crew_prob_matrix[x, y] *= prob_of_beep
                else:
                    self.crew_prob_matrix[x, y] *= (1 - prob_of_beep)

        # Normalize the crew member probability matrix
        self.crew_prob_matrix /= np.sum(self.crew_prob_matrix)

        # Update the alien probability matrix only if the alien is sensed
        if alien_sensed:
            for x in range(self.dimension):
                for y in range(self.dimension):
                    if self.distance((x, y), self.bot_pos) <= (2 * self.k + 1):
                        self.alien_prob_matrix[x, y] *= 2  # Increase probability if within sensing range
                    else:
                        self.alien_prob_matrix[x, y] *= 0.5  # Decrease probability if outside sensing range

        # Normalize the alien probability matrix
        self.alien_prob_matrix /= np.sum(self.alien_prob_matrix)
        
    def move_based_on_prob(self):
        best_move = self.bot_pos
        best_score = float('-inf')

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  
            nx, ny = self.bot_pos[0] + dx, self.bot_pos[1] + dy
            if 0 <= nx < self.dimension and 0 <= ny < self.dimension:
                # Calculate a simple score: crew probability minus alien probability
                score = self.crew_prob_matrix[nx, ny] - self.alien_prob_matrix[nx, ny]
                if score > best_score:
                    best_score = score
                    best_move = (nx, ny)

        self.bot_pos = best_move
        self.update_grid()
    
    def run(self):
        steps = 0
        while True:
            beep_detected, alien_sensed = self.sense_environment()
            self.update_prob_matrices(beep_detected, alien_sensed)
            self.move_based_on_prob()
            
            print(f"Step {steps}:")
            self.print_grid()

            if self.bot_pos == self.crew_pos:
                print(f"Bot 1 rescued the crew member in {steps} steps!")
                break
            
            self.move_alien_randomly()
            
            if self.bot_pos == self.alien_pos:
                print(f"Bot 1 was destroyed by the alien after {steps + 1} steps.")
                break
            
            steps += 1

if __name__ == "__main__":
    bot = Bot1(dimension=35, alpha=0.01, k=1)
    bot.run()

