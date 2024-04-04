import random
import numpy as np


class Bot5:

    # Bot 5 is similar to Bot4, but it computes a joint probability matrix for the crew members and takes the utility function from Bot2

    def __init__(self, dimension=10, alpha=0.01, k=5):
        self.dimension = dimension
        self.alpha = alpha
        self.k = k
        self.grid = np.full((dimension, dimension), '#', dtype=str)  # Fill it with blocked cells
        self.initialize_ship_layout()
        # Initialize positions with placeholders
        self.bot_pos = None
        self.crew_positions = None
        self.alien_pos = None
        # Set positions ensuring no attribute is referenced before it's assigned
        self.bot_pos = self.place_random()
        self.crew_positions = [self.place_random(exclude=[self.bot_pos]),
                               self.place_random(exclude=[self.bot_pos])]  # 2 crew
        self.alien_pos = self.place_random(exclude=[self.bot_pos], outside_k=True)

        # Make sure each crew has a probability matrix to work with
        self.crew_prob_matrix = np.zeros((dimension, dimension))  # Joint probability matrix for both crew members
        self.alien_prob_matrix = np.zeros((dimension, dimension))
        self.visited_matrix = np.zeros((self.dimension, self.dimension))
        
        #To prevent move cycling
        self.last_positions = []
        self.random_move_count = 0

        self.update_grid()

        # Initializes both alien and crew prob matrices
        self.update_prob_matrices_initial()

    def initialize_ship_layout(self):
        # Open a random cell
        start_x, start_y = random.randint(1, self.dimension - 2), random.randint(1, self.dimension - 2)
        self.grid[start_x, start_y] = '.'

        open_list = [(start_x, start_y)]
        while open_list:
            # Find all blocked cells with exactly one open neighbor
            candidates = []
            for x, y in open_list:
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if self.is_valid(nx, ny) and self.grid[nx, ny] == '#' and self.count_open_neighbors(nx, ny) == 1:
                        candidates.append((nx, ny))

            if not candidates:
                break  # Exit if no candidates are found

            # Randomly select one and open it
            new_open = random.choice(candidates)
            self.grid[new_open] = '.'
            open_list.append(new_open)

            # Remove duplicates from open_list
            open_list = list(set(open_list))

        # Open additional cells to reduce dead ends
        self.reduce_dead_ends()

    def is_valid(self, x, y):
        return 0 <= x < self.dimension and 0 <= y < self.dimension

    def count_open_neighbors(self, x, y):
        return sum(self.grid[x + dx, y + dy] == '.' for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)] if
                   self.is_valid(x + dx, y + dy))

    def reduce_dead_ends(self):
        dead_ends = [(x, y) for x in range(1, self.dimension - 1) for y in range(1, self.dimension - 1) if
                     self.grid[x, y] == '.' and self.count_open_neighbors(x, y) == 1]
        for x, y in random.sample(dead_ends, len(dead_ends) // 2):  # Approx. half of dead ends
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                if self.is_valid(x + dx, y + dy) and self.grid[x + dx, y + dy] == '#':
                    self.grid[x + dx, y + dy] = '.'
                    break

    def place_random(self, exclude=None, outside_k=False):
        if exclude is None:
            exclude = []
        open_positions = [(x, y) for x in range(self.dimension) for y in range(self.dimension) if
                          self.grid[x, y] == '.' and (x, y) not in exclude]
        if open_positions:
            while True:
                pos = random.choice(open_positions)
                if pos in exclude:
                    continue  # Skip positions that are in the exclude list
                if outside_k:
                    if self.distance(pos, self.bot_pos) <= 2 * self.k + 1:
                        continue  # Skip positions within the k radius
                return pos

    def distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def update_grid(self):
        # Resets the grid while preserving walls
        temp_grid = np.full((self.dimension, self.dimension), '.', dtype=str)
        for x in range(self.dimension):
            for y in range(self.dimension):
                if self.grid[x, y] == '#':  # Keep the walls
                    temp_grid[x, y] = '#'
        temp_grid[self.bot_pos] = 'B'
        for i in range(2):
            if (self.crew_positions[i] == None):
                continue  # Do not print, crew rescued
            temp_grid[self.crew_positions[i]] = 'C'
        temp_grid[self.alien_pos] = 'A'
        self.grid = temp_grid

    def print_grid(self):
        for y in range(self.dimension):
            for x in range(self.dimension):
                print(self.grid[x, y], end=" ")
            print()

    print("\n")

    def update_prob_matrices_initial(self):
        crew_unoccupied_cells = self.dimension ** 2 - len(
        [pos for pos in self.crew_positions if pos is not None]) - 1  # Minus 1 for the bot's position

        alien_cells = self.dimension**2 - ((2*self.k+1)**2) - 1
        for x in range(self.dimension):
            for y in range(self.dimension):
                if self.grid[x, y] != '#':  # Check if the cell is not a wall
                    # Update alien probability matrix
                    if self.distance((x, y), self.bot_pos) > 2 * self.k + 1:
                        self.alien_prob_matrix[x, y] = 1 / alien_cells
                    else:
                        self.alien_prob_matrix[x, y] = 0  # Cells within the k-radius are initially improbable for the alien

                    # Split the crew probability between the two possible crew member locations
                    # For simplicity, we equally distribute probability among all non-wall cells
                    self.crew_prob_matrix[x, y] = 1 / crew_unoccupied_cells

        # Normalize each matrix to ensure their probabilities sum to 1
        self.crew_prob_matrix /= np.sum(self.crew_prob_matrix)
        self.alien_prob_matrix /= np.sum(self.alien_prob_matrix)

    """
    def print_crew_prob_matrix(self):
        # Print the crew probability matrix
        print("Crew Probability Matrix:")
        for y in range(self.dimension):
            for x in range(self.dimension):
                print(f"{self.crew_prob_matrix[x, y]:.5f} ", end="")
            print()
        print("\n")
    """

    def sense_environment(self):
        # Determine if a beep is detected based on the closest crew member
        distances = [self.distance(self.bot_pos, pos) for pos in self.crew_positions if pos is not None]
        min_distance = min(distances, default=None)
        beep_detected = False

        for x in range(self.dimension):
            for y in range(self.dimension):
                # check if crew member actually exists in cells d-Manhattan distance away
                if (self.distance(self.bot_pos, (
                x, y)) == min_distance):  # Check for min distance, but the Bot3 does not know which crew it is
                    if (self.grid[x, y] == 'C'):  # Check if the cell contains the crew
                        beep_detected = True
                        break
            else:  # Make sure the outer loop isn't broken if the inner loop is
                continue
            # inner loop broken, break the outer
            break
        if beep_detected:  # Assess reliability on uniform random. Now we know that C exists on our path. Beep detected for any crew member
            beep_detected = any(
                random.random() <= np.exp(-self.alpha * (self.distance(self.bot_pos, crew_pos) - 1)) for crew_pos in
                self.crew_positions if crew_pos is not None)
        # Check if Alien is within the radius
        alien_sensed = self.distance(self.bot_pos, self.alien_pos) <= (2 * self.k + 1)
        return beep_detected, alien_sensed

    def update_prob_matrices(self, beep_detected, alien_sensed):

        # Temporary matrices to hold the updated probabilities, keeps track of past probabilities
        new_alien_prob_matrix = np.zeros_like(self.alien_prob_matrix)
        
        for _ , crew_pos in enumerate(self.crew_positions):
            if crew_pos is None:
                continue  # skip updating the rescued crew prob matrix
            # Update crew probability matrix using Bayesian updating
            new_crew_prob_matrix = np.zeros_like(self.crew_prob_matrix)
            
            for x in range(self.dimension):
                for y in range(self.dimension):
                    # Skip walls
                    if self.grid[x, y] == '#':
                        self.crew_prob_matrix[x, y] = 0
                        continue
                    
                    distance = self.distance((x, y), self.bot_pos)
                    #beep_probability = np.exp(-self.alpha * (distance - 1))  # for each crew member

                    if beep_detected:
                        if distance < self.distance(self.bot_pos, crew_pos):
                            # Update based on the likelihood of detecting a beep given the crew is at (x, y)
                            factor = np.fromfunction(lambda x, y: np.exp(
                                -self.alpha * (np.abs(x - self.bot_pos[0]) + np.abs(y - self.bot_pos[1]) - 1)),
                                                    shape=(self.dimension, self.dimension))
                            self.crew_prob_matrix *= factor
                            new_crew_prob_matrix[x, y] = self.crew_prob_matrix[x, y]
                        else:
                            self.crew_prob_matrix *= 1 - np.fromfunction(lambda x, y: np.exp(
                                -self.alpha * (np.abs(x - self.bot_pos[0]) + np.abs(y - self.bot_pos[1]) - 1)),
                                                                        shape=(self.dimension, self.dimension))
                            new_crew_prob_matrix[x, y] = self.crew_prob_matrix[x, y]

                    # Apply exploration incentive for unvisited cells, Hansel & Gretel approach
                    if self.visited_matrix[x, y] == 0:
                            new_crew_prob_matrix[x, y] = self.crew_prob_matrix[x, y] * 10
                    new_crew_prob_matrix[self.bot_pos] = self.crew_prob_matrix[x, y] - 1
                    # adjust penalty for not going back

            # Normalize the crew probability matrix to ensure probabilities sum to 1
            total_crew_prob = np.sum(new_crew_prob_matrix)
            if total_crew_prob > 0:
                self.crew_prob_matrix = new_crew_prob_matrix / total_crew_prob

        # Update alien probability matrix using Bayesian updating
        if alien_sensed:
            for x in range(self.dimension):
                for y in range(self.dimension):
                    # Skip walls
                    if self.grid[x, y] == '#':
                        self.alien_prob_matrix[x, y] = 0
                        continue

                    if self.distance((x, y), self.bot_pos) <= (2 * self.k + 1):
                        # If alien is sensed and within range, increase probability
                        new_alien_prob_matrix[x, y] = self.alien_prob_matrix[x, y] * 2
                    else:
                        # Decrease likelihood for positions outside of sensing range
                        new_alien_prob_matrix[x, y] = self.alien_prob_matrix[x, y] * 0.01

            # Normalize the alien probability matrix to ensure probabilities sum to 1
            total_alien_prob = np.sum(new_alien_prob_matrix)
            if total_alien_prob > 0:
                self.alien_prob_matrix = new_alien_prob_matrix / total_alien_prob

    def move_based_on_prob(self):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left
        best_move = None
        best_utility = float('-inf')  # Initialize with lowest possible score for crew
        
        #Check for Move Cycling
        if self.random_move_count > 0:
            safe_moves = [move for move in directions if self.is_move_safe(self.bot_pos[0] + move[0], self.bot_pos[1] + move[1])]
            if safe_moves:
                chosen_move = random.choice(safe_moves)
                self.bot_pos = (self.bot_pos[0] + chosen_move[0], self.bot_pos[1] + chosen_move[1])
                self.random_move_count -= 1
                print("Moving randomly to prevent move cycling")
            else:
                print("Staying in place due to no safe moves. Still preventing move cycling")
        else:

            for dx, dy in directions:
                nx, ny = self.bot_pos[0] + dx, self.bot_pos[1] + dy
                # Ensure the move is within bounds and not into a wall.
                if 0 <= nx < self.dimension and 0 <= ny < self.dimension and self.grid[nx, ny] != '#' and self.grid[
                    nx, ny] != 'A':
                    if (self.grid[nx, ny] == 'C'):
                        best_move = (dx, dy)
                        best_utility = 1.0
                        break
                    current_utility = self.calculate_move_utility((nx, ny))
                    # Calculate utility based utility function
                    # Choose the move with the highest utility that maximizes crew probability and minimizes alien risk
                    if current_utility > best_utility:
                        best_utility = current_utility
                        best_move = (dx, dy)

            
            # If best move leads back to a recently visited cell, start random move sequence
            if best_move and (self.bot_pos[0] + best_move[0], self.bot_pos[1] + best_move[1]) in self.last_positions:
                self.random_move_count = 3  # Number of random moves to make
                self.move_bot_randomly(directions)  # Define this method to handle random movement
            # Execute the best move if found
            elif best_move and best_utility > 0.0:
                self.visited_matrix[self.bot_pos] = 1  # Mark the current position as visited
                self.update_position_and_history(best_move)
            else:
                print("Going random, not move cycling")
                self.move_bot_randomly(directions)
            

        self.update_grid()

    # Ensure the move_alien_randomly and other relevant methods also respect walls.

    def move_alien_randomly(self):
        directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]  # Up, Left, Right, Down
        random.shuffle(directions)
        for dx, dy in directions:
            new_pos = (self.alien_pos[0] + dx, self.alien_pos[1] + dy)
            # Ensure new position is within bounds and not a wall
            if 0 <= new_pos[0] < self.dimension and 0 <= new_pos[1] < self.dimension and self.grid[new_pos] != '#':
                self.alien_pos = new_pos
                break
        self.update_grid()  # Update grid to reflect new alien position

    def calculate_move_utility(self, pos):
        if self.grid[pos] == '#':  # Ignore walls
            return float('-inf')  # Assign very low utility

        crew_prob = self.crew_prob_matrix[pos]
        alien_risk = self.alien_prob_matrix[pos]
        visited_penalty = 0 if self.visited_matrix[pos] == 0 else -1  # Penalize visited cells
        exploration_bonus = 0.1 if self.visited_matrix[pos] == 0 else 0  # Encourage exploration

        # Combine the components to calculate utility
        utility = crew_prob - (alien_risk * 2) + exploration_bonus + visited_penalty

        # Proximity bonus: Calculate the distance to the highest probability crew location and adjust utility
        highest_prob_pos = np.unravel_index(np.argmax(self.crew_prob_matrix), self.crew_prob_matrix.shape)
        distance_to_highest_prob = self.distance(pos, highest_prob_pos)
        proximity_bonus = 1 / (distance_to_highest_prob + 1)  # Avoid division by zero

        return utility + proximity_bonus
    
    #Keep track of Bot movement history
    def update_position_and_history(self, best_move):
        # Update bot's position
        new_pos = (self.bot_pos[0] + best_move[0], self.bot_pos[1] + best_move[1])
        self.bot_pos = new_pos
        # Update history
        if len(self.last_positions) >= 2:
            self.last_positions.pop(0)  # Remove the oldest position if there are already 2
        self.last_positions.append(new_pos)

    #Move bot randomly if it move cycles.
    def move_bot_randomly(self, directions):
        safe_moves = [move for move in directions if self.is_move_safe(self.bot_pos[0] + move[0], self.bot_pos[1] + move[1])]
        if safe_moves:
            chosen_move = random.choice(safe_moves)
            self.bot_pos = (self.bot_pos[0] + chosen_move[0], self.bot_pos[1] + chosen_move[1])
            print("Random move due no good move, ties everywhere")
        else:
            print("Staying in place due to no safe moves.")
    
    def is_move_safe(self, x, y):
        return 0 <= x < self.dimension and 0 <= y < self.dimension and self.grid[x, y] != 'A' and self.grid[x, y] != '#'
    

    def run(self):
        steps = 0
        crew_saved = 0
        while steps < 10000: #Game ends when alien catches roomba, or crew is saved
            beep_detected, alien_sensed = self.sense_environment()
            self.update_prob_matrices(beep_detected, alien_sensed)
            self.move_based_on_prob()
            self.move_alien_randomly()
            # print(f"Crew Distance: {self.distance(self.bot_pos, self.crew_pos)}")
            #print(f"Alien Distance: {self.distance(self.bot_pos, self.alien_pos)}")
           # print(f"Beep Detected: {beep_detected}, Alien Sensed: {alien_sensed}")
            #print(f"Position Bot: {self.bot_pos}")
            #for i in range(2):
                #print(f"Position of Crew: {self.crew_positions[i]}")
            #print(f"Position Alien: {self.alien_pos}")
            #self.print_grid()

            # self.print_crew_prob_matrix()
            for i, crew_pos in enumerate(
                    self.crew_positions):  # Need to keep track of C rescued, if all crew_pos is None, every crew is rescued
                if crew_pos and self.bot_pos == crew_pos:
                    print(f"Bot 5 rescued crew member at position {crew_pos}")
                    crew_saved +=1
                    self.crew_positions[i] = None

            self.update_grid()
            print(f"Step: {steps}.")
            if all(crew_pos is None for crew_pos in self.crew_positions):  # End simulation if everyone is rescued
                return (True, steps, crew_saved)

            if self.bot_pos == self.alien_pos:
                print(f"Bot 5 was destroyed by the alien after {steps + 1} steps.")
                return (False, steps, crew_saved)

            steps += 1
        return (False, steps, crew_saved)

if __name__ == "__main__":
    bot = Bot5(dimension=10, alpha=0.05, k=1)
    result, steps, crew_saved = bot.run()
    print(result, steps)
