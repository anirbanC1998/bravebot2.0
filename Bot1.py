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
        
    
    def print_crew_prob_matrix(self):
        # Print the crew probability matrix
        print("Crew Probability Matrix:")
        for y in range(self.dimension):
            for x in range(self.dimension):
                print(f"{self.crew_prob_matrix[x, y]:.2f} ", end="")
            print()
        print("\n")
    
    def sense_environment(self):
        beep_detected = (0 <= np.exp(-self.alpha * (self.distance(self.bot_pos, self.crew_pos) - 1)))
        alien_sensed = self.distance(self.bot_pos, self.alien_pos) <= (2 * self.k + 1)
        return beep_detected, alien_sensed
    
    def update_prob_matrices(self, beep_detected, alien_sensed):
        # Initialize a matrix to accumulate new probabilities for the crew
        new_crew_prob_matrix = np.zeros((self.dimension, self.dimension))
        
       # Calculate total probability for normalization
        total_crew_prob = 0
        
        # Update probabilities based on beep detection
        for x in range(self.dimension):
            for y in range(self.dimension):
                distance = self.distance((x, y), self.bot_pos)

                    # Adjust the probability based on the distance and whether a beep was detected
                if beep_detected:
                    # The closer the bot is to the crew, the higher the probability should be when a beep is detected
                    new_crew_prob_matrix[x, y] = (np.exp(-self.alpha * (distance - 1))) + 0.1
                else:
                    # Without a beep, the probability update might be less straightforward. You might choose to skip updates or decrease them slightly.
                    continue
                
         # Normalize the updated crew probability matrix to ensure total probabilities sum to 1
        prob_sum = np.sum(new_crew_prob_matrix)
        if prob_sum > 0:
            self.crew_prob_matrix = new_crew_prob_matrix / prob_sum
            # Handle the case where total probability is zero to avoid division by zero
            self.crew_prob_matrix = np.full((self.dimension, self.dimension), 1/(self.dimension**2))

            # Alien probability update is less critical for Bot1's immediate behavior but should reflect risk areas
            if alien_sensed:
                for x in range(self.dimension):
                    for y in range(self.dimension):
                        if self.distance((x, y), self.bot_pos) <= (2 * self.k + 1):
                            # Increase probability for cells within sensing range
                            self.alien_prob_matrix[x, y] = min(self.alien_prob_matrix[x, y] * 1.1, 1)
                        else:
                            continue

            # Since alien probabilities are adjusted in place, normalization may not be strictly necessary but can be done for consistency
            alien_prob_sum = np.sum(self.alien_prob_matrix)
            if alien_prob_sum > 0:
                self.alien_prob_matrix /= alien_prob_sum


    def move_based_on_prob(self):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left
        best_move = None
        best_score = float('-inf')  # Start with the lowest possible score

        for dx, dy in directions:
            nx, ny = self.bot_pos[0] + dx, self.bot_pos[1] + dy
            if 0 <= nx < self.dimension and 0 <= ny < self.dimension:
                # Calculate a score that prefers higher crew probabilities and lower alien probabilities
                score = self.crew_prob_matrix[nx, ny]# - alien matrix, took it out for testing

                # Update the best move if this direction has a better score
                if score > best_score:
                    best_score = score
                    best_move = (dx, dy)

        # Make the move if a better option has been found
        if best_move:
            self.bot_pos = (self.bot_pos[0] + best_move[0], self.bot_pos[1] + best_move[1])
        else:
            # If no preferable move is identified, the bot can stay in place or you might choose a random direction
            # Staying in place might be sensible if all moves seem equally bad or good.
            print("No preferable move identified. Bot is staying in place to reassess.")

        self.update_grid()


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
    
    def run(self):
        steps = 0
        while True:
            beep_detected, alien_sensed = self.sense_environment()
            self.update_prob_matrices(beep_detected, alien_sensed)
            self.move_based_on_prob()
            self.move_alien_randomly()
            print(f"Position Bot: {self.bot_pos}")
            print(f"Position Crew: {self.crew_pos}")
            
            print(f"Step: {steps}.")
            self.print_grid()
            #self.print_crew_prob_matrix()

            if self.bot_pos == self.crew_pos:
                print(f"Bot 1 rescued the crew member in {steps} steps!")
                break
            
            
            if self.bot_pos == self.alien_pos:
                print(f"Bot 1 was destroyed by the alien after {steps + 1} steps.")
                break
            
            steps += 1

if __name__ == "__main__":
    bot = Bot1(dimension=35, alpha=0.001, k=1)
    bot.run()

