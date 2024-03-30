import numpy as np
import random

class SpaceRoombaEnvironment:
    def __init__(self, dimension=35, alpha=0.5, k=3):
        self.dimension = dimension
        self.grid = np.full((dimension, dimension), '.', dtype=str)
        self.alpha = alpha
        self.k = k
        
        # Initially define bot, crew, and alien positions as None
        self.bot_pos = None
        self.crew_positions = []
        self.alien_pos = None
        
        # First, place the bot at a random position
        self.bot_pos = (random.randint(0, self.dimension - 1), random.randint(0, self.dimension - 1))
        
        # Then, initialize crew positions ensuring they are distinct from the bot's position
        self.crew_positions = [self.place_random(exclude=[self.bot_pos]) for _ in range(2)]
        
        # Finally, place the alien, ensuring it's outside the bot's k radius and not overlapping with crew positions
        self.alien_pos = self.place_random(outside_k=True, exclude=self.crew_positions + [self.bot_pos])

        # Update alien probability matrix initial state and the grid
        self.crew_prob_matrix = np.full((dimension, dimension), 1.0 / (dimension**2))
        self.alien_prob_matrix = np.zeros((dimension, dimension))
        self.update_alien_prob_matrix_initial()
        self.update_grid()

    def place_random(self, outside_k=False, exclude=[]):
        while True:
            pos = (random.randint(0, self.dimension - 1), random.randint(0, self.dimension - 1))
            if pos in exclude:
                continue
            if outside_k and self.distance(pos, self.bot_pos) <= 2 * self.k + 1:
                continue
            return pos

    def distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def update_grid(self):
        self.grid.fill('.')
        self.grid[self.bot_pos] = 'B'
        for pos in self.crew_positions:
            self.grid[pos] = 'C'
        self.grid[self.alien_pos] = 'A'

    def print_grid(self):
        print('\n'.join([' '.join(row) for row in self.grid]))
        print()

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
        for i in range(self.dimension):
            for j in range(self.dimension):
                if self.distance((i, j), self.bot_pos) > 2 * self.k + 1:
                    self.alien_prob_matrix[i, j] = 1.0 / (self.dimension * self.dimension - (2 * self.k + 1) ** 2)
        total_prob = np.sum(self.alien_prob_matrix)
        self.alien_prob_matrix /= total_prob  # Normalize the probabilities
    
    def update_alien_prob_matrix_after_move(self):
        # Spread the probability from the alien's current position to adjacent cells
        temp_matrix = np.zeros((self.dimension, self.dimension))
        for x in range(self.dimension):
            for y in range(self.dimension):
                if self.distance((x, y), self.alien_pos) <= 1:
                    temp_matrix[x, y] = 1.0 / 5  # Adjacent cells + current cell
        self.alien_prob_matrix = temp_matrix

class Bot3(SpaceRoombaEnvironment):
    def __init__(self, dimension=35, alpha=0.5, k=3):
        super().__init__(dimension, alpha, k)

    def sense_crew(self):
        beep_probabilities = [np.exp(-self.alpha * (self.distance(self.bot_pos, pos) - 1)) for pos in self.crew_positions]
        return any(random.random() < prob for prob in beep_probabilities)

    def update_prob_matrix_after_move(self, beep_detected):
        new_prob_matrix = np.zeros((self.dimension, self.dimension))
        for x in range(self.dimension):
            for y in range(self.dimension):
                distance = min([self.distance((x, y), pos) for pos in self.crew_positions])
                prob_of_beep = np.exp(-self.alpha * (distance - 1))
                if beep_detected:
                    new_prob_matrix[x, y] = self.crew_prob_matrix[x, y] * prob_of_beep
                else:
                    new_prob_matrix[x, y] = self.crew_prob_matrix[x, y] * (1 - prob_of_beep)
        total_prob = np.sum(new_prob_matrix)
        if total_prob > 0:
            self.crew_prob_matrix = new_prob_matrix / total_prob
        else:
            self.crew_prob_matrix = np.full((self.dimension, self.dimension), 1.0 / (self.dimension**2 - 1))
        self.update_alien_prob_matrix()

    def update_alien_prob_matrix(self):
        temp_matrix = np.zeros((self.dimension, self.dimension))
        directions = [(0, -1), (-1, 0), (1, 0), (0, 1), (0, 0)]  # Including staying in place
        for dx, dy in directions:
            adj_x, adj_y = self.alien_pos[0] + dx, self.alien_pos[1] + dy
            if 0 <= adj_x < self.dimension and 0 <= adj_y < self.dimension:
                temp_matrix[adj_x, adj_y] = 1
        self.alien_prob_matrix = temp_matrix / sum(temp_matrix.flatten())

    def decide_move_based_on_prob(self):
        best_move = None
        best_score = float('-inf')
        for dx, dy in [(0, -1), (-1, 0), (1, 0), (0, 1), (0, 0)]:
            nx, ny = self.bot_pos[0] + dx, self.bot_pos[1] + dy
            if 0 <= nx < self.dimension and 0 <= ny < self.dimension:
                # Factor in both finding crew and avoiding alien
                score = self.crew_prob_matrix[nx, ny] - 5 * self.alien_prob_matrix[nx, ny]
                if score > best_score:
                    best_score = score
                    best_move = (nx, ny)
        if best_move:
            self.bot_pos = best_move
        self.update_grid()

    def run(self):
        while len(self.crew_positions) > 0:
            beep_detected = self.sense_crew()
            self.update_prob_matrix_after_move(beep_detected)
            self.decide_move_based_on_prob()
            self.move_alien_randomly()
            self.update_grid()
            if self.bot_pos in self.crew_positions:
                print(f"Rescued a crew member at {self.bot_pos}.")
                self.crew_positions.remove(self.bot_pos)
                if len(self.crew_positions) == 0:
                    print("All crew members rescued.")
                    break
            if self.bot_pos == self.alien_pos:
                print("Bot was destroyed by the alien. Game Over.")
                break
            self.print_grid()

if __name__ == "__main__":
    
    bot3 = Bot3(dimension=35, alpha=0.01, k=1)
    bot3.run()
