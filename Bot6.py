import numpy as np
import random

class SpaceRoombaEnvironment:
    def __init__(self, dimension=35, alpha=0.5, k=3):
        self.dimension = dimension
        self.alpha = alpha
        self.k = k
        self.grid = np.full((dimension, dimension), '.', dtype=str)
        
        # Initially define bot, crew, and alien positions as None
        self.bot_pos = None
        self.crew_positions = []
        self.alien_positions = []
        
        # First, place the bot at a random position
        self.bot_pos = (random.randint(0, self.dimension - 1), random.randint(0, self.dimension - 1))
        
        self.crew_positions = [self.place_random() for _ in range(2)]
        self.alien_positions = [self.place_random(outside_k=True) for _ in range(2)]
        self.crew_prob_matrix = np.full((dimension, dimension), 1.0 / (dimension**2 - 1))
        self.alien_prob_matrix = np.full((dimension, dimension), 1.0 / (dimension**2 - 1))
        self.update_grid()

    def place_random(self, outside_k=False):
        while True:
            pos = (random.randint(0, self.dimension - 1), random.randint(0, self.dimension - 1))
            if pos not in [self.bot_pos] + self.crew_positions + self.alien_positions:
                if outside_k and self.distance(pos, self.bot_pos) <= 2 * self.k + 1:
                    continue
                return pos

    def distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def move_aliens_randomly(self):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
        new_positions = []
        for alien_pos in self.alien_positions:
            valid_moves = [(alien_pos[0] + dx, alien_pos[1] + dy) for dx, dy in directions
                        if 0 <= alien_pos[0] + dx < self.dimension and 0 <= alien_pos[1] + dy < self.dimension]
            new_positions.append(random.choice(valid_moves))
        self.alien_positions = new_positions


    def update_grid(self):
        self.grid.fill('.')
        self.grid[self.bot_pos] = 'B'
        for pos in self.crew_positions:
            self.grid[pos] = 'C'
        for pos in self.alien_positions:
            self.grid[pos] = 'A'

    def print_grid(self):
        print('\n'.join([' '.join(row) for row in self.grid]))
        print()

class Bot6(SpaceRoombaEnvironment):
    def __init__(self, dimension=35, alpha=0.5, k=3):
        super().__init__(dimension, alpha, k)
        # Initialize probability matrices for crew members
        self.crew_prob_matrices = [np.full((dimension, dimension), 1.0 / (dimension**2)) for _ in range(2)]
        # Initialize probability matrices for aliens
        self.alien_prob_matrices = [np.full((dimension, dimension), 1.0 / (dimension**2)) for _ in range(2)]

    def sense_environment(self):
        beep_detected = any(random.random() < np.exp(-self.alpha * (self.distance(self.bot_pos, crew_pos) - 1)) for crew_pos in self.crew_positions)
        alien_sensed = any(self.distance(self.bot_pos, alien_pos) <= (2 * self.k + 1) for alien_pos in self.alien_positions)
        return beep_detected, alien_sensed

    def update_prob_matrices(self, beep_detected, alien_sensed):
        # Update Crew Probability Matrices
        for crew_matrix in self.crew_prob_matrices:
            for x in range(self.dimension):
                for y in range(self.dimension):
                    dist = self.distance((x, y), self.bot_pos)
                    prob_beep = np.exp(-self.alpha * (dist - 1))
                    if beep_detected:
                        crew_matrix[x, y] *= prob_beep
                    else:
                        crew_matrix[x, y] *= (1 - prob_beep)
            # Normalize
            crew_matrix /= np.sum(crew_matrix)

        # Assuming alien_sensed is True if either alien is within detection range; does not distinguish between one or two aliens
        for alien_matrix in self.alien_prob_matrices:
            # If an alien is sensed, adjust probabilities within the detection range
            if alien_sensed:
                for x in range(self.dimension):
                    for y in range(self.dimension):
                        dist = self.distance((x, y), self.bot_pos)
                        # Increase probability within a small vicinity of the bot
                        if dist <= (2 * self.k + 1):
                            alien_matrix[x, y] += 0.1  # This can be adjusted based on expected alien movement
            # Normalize
            alien_matrix /= np.sum(alien_matrix)


    def move_based_on_prob(self):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]  # Including staying in place
        best_move = self.bot_pos
        best_score = float('-inf')

        exploration_bonus = 0.01  # Encourage exploration by adding a small bonus to non-stationary moves

        for dx, dy in directions:
            nx, ny = self.bot_pos[0] + dx, self.bot_pos[1] + dy
            if 0 <= nx < self.dimension and 0 <= ny < self.dimension:
                crew_score = sum(matrix[nx, ny] for matrix in self.crew_prob_matrices)
                alien_score = sum(matrix[nx, ny] for matrix in self.alien_prob_matrices)
                score = crew_score - (5 * alien_score)
                
                # Apply exploration bonus for moves that are not staying in place
                if (dx, dy) != (0, 0):
                    score += exploration_bonus
                
                if score > best_score:
                    best_score = score
                    best_move = (nx, ny)

        if best_move != self.bot_pos:
            self.bot_pos = best_move
        self.update_grid()



    def run(self):
        while len(self.crew_positions) > 0:
            beep_detected, alien_sensed = self.sense_environment()
            self.update_prob_matrices(beep_detected, alien_sensed)
            self.move_based_on_prob()
            self.move_aliens_randomly()
            print(self.bot_pos)
            self.print_grid()

            # Check for crew rescues
            for pos in list(self.crew_positions):
                if self.bot_pos == pos:
                    print(f"Rescued a crew member at {pos}.")
                    self.crew_positions.remove(pos)

            # Check for alien encounter
            if self.bot_pos in self.alien_positions:
                print("Bot6 was destroyed by the alien. Game over.")
                break

            if not self.crew_positions:
                print("All crew members have been rescued. Bot6 mission successful.")
                break

if __name__ == "__main__":
    bot6 = Bot6(dimension=35, alpha=0.01, k=1)
    bot6.run()
