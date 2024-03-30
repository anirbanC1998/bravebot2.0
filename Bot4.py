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

class Bot4(SpaceRoombaEnvironment):
    def __init__(self, dimension=35, alpha=0.5, k=3):
        super().__init__(dimension, alpha, k)
        # Assume crew_positions is defined in SpaceRoombaEnvironment along with bot_pos and alien_pos
        self.crew1_prob_matrix = np.full((dimension, dimension), 1.0 / (dimension**2))
        self.crew2_prob_matrix = np.full((dimension, dimension), 1.0 / (dimension**2))
        self.alien_prob_matrix = self.initialize_alien_prob_matrix()
        self.update_grid()
    
    def sense_crew(self):
        beep_probabilities = [np.exp(-self.alpha * (self.distance(self.bot_pos, pos) - 1)) for pos in self.crew_positions]
        return any(random.random() < prob for prob in beep_probabilities)

    def initialize_alien_prob_matrix(self):
        # Alien is initially equally likely to be anywhere outside the bot's immediate k radius
        prob_matrix = np.full((self.dimension, self.dimension), 1.0)
        for i in range(max(0, self.bot_pos[0]-self.k), min(self.dimension, self.bot_pos[0]+self.k+1)):
            for j in range(max(0, self.bot_pos[1]-self.k), min(self.dimension, self.bot_pos[1]+self.k+1)):
                prob_matrix[i, j] = 0
        prob_matrix /= np.sum(prob_matrix)  # Normalize
        return prob_matrix

    def update_alien_prob_matrix(self):
        new_prob_matrix = np.zeros((self.dimension, self.dimension))
        for dx, dy in [(0, -1), (-1, 0), (1, 0), (0, 1)]:
            nx, ny = self.alien_pos[0] + dx, self.alien_pos[1] + dy
            if 0 <= nx < self.dimension and 0 <= ny < self.dimension:
                new_prob_matrix[nx, ny] = 1.0 / 5  # Spread probability equally to adjacent cells
        self.alien_prob_matrix = new_prob_matrix

    def move_alien_randomly(self):
        directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = self.alien_pos[0] + dx, self.alien_pos[1] + dy
            if 0 <= nx < self.dimension and 0 <= ny < self.dimension:
                self.alien_pos = (nx, ny)
                break
        self.update_alien_prob_matrix()  # Update the alien's probability matrix after it moves
    
    def update_prob_matrix_after_move(self, beep_detected):
    # Assuming beep affects the probability of both crew members equally
        for matrix in [self.crew1_prob_matrix, self.crew2_prob_matrix]:
            new_matrix = np.zeros((self.dimension, self.dimension))
            for x in range(self.dimension):
                for y in range(self.dimension):
                    # Assuming the probability of a beep decreases with distance
                    distance = self.distance((x, y), self.bot_pos)
                    prob_of_beep = np.exp(-self.alpha * (distance - 1))
                    if beep_detected:
                        # Increase probability for closer cells
                        new_matrix[x, y] = matrix[x, y] * prob_of_beep
                    else:
                        # Decrease probability for closer cells
                        new_matrix[x, y] = matrix[x, y] * (1 - prob_of_beep)
            total_prob = np.sum(new_matrix)
            if total_prob > 0:
                matrix[:, :] = new_matrix / total_prob  # Normalize the matrix

    # For simplicity, this version does not explicitly update the alien probability matrix in response to movement.
    # Implement similar logic as for the crew if the alien's movement patterns are to be considered.

    def decide_move_based_on_prob(self):
        best_move = None
        best_score = -np.inf
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]: 
            nx, ny = self.bot_pos[0] + dx, self.bot_pos[1] + dy
            if 0 <= nx < self.dimension and 0 <= ny < self.dimension:
                # Sum the probabilities for both crew members and subtract the alien's probability to avoid it
                crew_prob = self.crew1_prob_matrix[nx, ny] + self.crew2_prob_matrix[nx, ny]
                alien_avoidance = -self.alien_prob_matrix[nx, ny]
                score = crew_prob + alien_avoidance
                if score > best_score:
                    best_score = score
                    best_move = (nx, ny)
        
        if best_move:
            self.bot_pos = best_move
            # After moving, might need to update the grid or perform additional checks
            self.update_grid()
   

    def run(self):
        while len(self.crew_positions) > 0:
            beep_detected = self.sense_crew()
            self.update_prob_matrix_after_move(beep_detected)
            self.decide_move_based_on_prob()
            self.move_alien_randomly()
            self.update_grid()
            self.print_grid()
            
            # Check for crew rescues and remove rescued members from the list
            if self.bot_pos in self.crew_positions:
                print(f"Rescued a crew member at {self.bot_pos}.")
                self.crew_positions.remove(self.bot_pos)
                if not self.crew_positions:  # All crew members rescued
                    print("All crew members have been rescued. Bot4 mission successful.")
                    break
            
            if self.bot_pos == self.alien_pos:
                print("Bot4 was destroyed by the alien. Game over.")
                break

if __name__ == "__main__":

    bot4 = Bot4(dimension=35, alpha=0.01, k=1)
    bot4.run()