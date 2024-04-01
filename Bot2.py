import random
import numpy as np


class Bot2:

    def __init__(self, dimension=35, alpha=0.01, k=5):
        self.dimension = dimension
        self.alpha = alpha
        self.k = k
        self.grid = np.full((dimension, dimension), '#', dtype=str)  # Fill it with blocked cells
        self.initialize_ship_layout()
        # Initialize positions with placeholders
        self.bot_pos = None
        self.crew_pos = None
        self.alien_pos = None
        # Set positions ensuring no attribute is referenced before it's assigned
        self.bot_pos = self.place_random()
        self.crew_pos = self.place_random(exclude=self.bot_pos)
        self.alien_pos = self.place_random(exclude=[self.bot_pos, self.crew_pos], outside_k=True)

        self.crew_prob_matrix = np.zeros((dimension, dimension))
        self.alien_prob_matrix = np.zeros((dimension, dimension))
        self.visited_matrix = np.zeros((self.dimension, self.dimension))

        self.update_grid()

        # Intializes both alien and crew prob matrices
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
        temp_grid[self.crew_pos] = 'C'
        temp_grid[self.alien_pos] = 'A'
        self.grid = temp_grid

    def print_grid(self):
        for y in range(self.dimension):
            for x in range(self.dimension):
                print(self.grid[x, y], end=" ")
            print()

    print("\n")

    def update_prob_matrices_initial(self):
        for x in range(self.dimension):
            for y in range(self.dimension):
                if self.grid[x, y] == '.':
                    if self.distance((x, y), self.bot_pos) > 2 * self.k + 1:
                        self.alien_prob_matrix[x, y] = 1 / (self.dimension ** 2 - (2 * self.k + 1) ** 2)
                    self.crew_prob_matrix[x, y] = 1 / (self.dimension ** 2 - 1)
        self.crew_prob_matrix /= self.crew_prob_matrix.sum()
        self.alien_prob_matrix /= self.alien_prob_matrix.sum()

    def print_crew_prob_matrix(self):
        # Print the crew probability matrix
        print("Crew Probability Matrix:")
        for y in range(self.dimension):
            for x in range(self.dimension):
                print(f"{self.crew_prob_matrix[x, y]:.2f} ", end="")
            print()
        print("\n")

    def sense_environment(self):
        d = self.distance(self.bot_pos, self.crew_pos)
        beep_detected = False

        for x in range(self.dimension):
            for y in range(self.dimension):
                # check if crew member actually exists in cells d-Manhattan distance away
                if (self.distance(self.bot_pos, (x, y)) == d):
                    if (self.grid[x, y] == 'C'):  # Check if the cell contains the crew
                        beep_detected = True
                        break
            else:  # Make sure the outer loop isn't broken if the inner loop is
                continue
            # inner loop broken, break the outer
            break
        if beep_detected:  # Assess reliability on uniform random. Now we know that C exists on our path.
            beep_detected = (random.random() <= np.exp(-self.alpha * (self.distance(self.bot_pos, self.crew_pos) - 1)))
        # Check if Alien is within the radius
        alien_sensed = self.distance(self.bot_pos, self.alien_pos) <= (2 * self.k + 1)
        return beep_detected, alien_sensed

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
        self.update_alien_prob_matrix_after_move()

    def update_alien_prob_matrix_after_move(self):
        # Initialize a new probability matrix
        new_prob_matrix = np.zeros((self.dimension, self.dimension))
        for x in range(self.dimension):
            for y in range(self.dimension):
                if self.grid[x, y] == '#':  # Ignore walls
                    continue
                distance = self.distance((x, y), self.alien_pos)
                if distance <= 1:
                    # Assign equal probability to adjacent cells and the alien's current cell
                    new_prob_matrix[x, y] = 1.0 / (1 + len([d for d in [(0, -1), (-1, 0), (1, 0), (0, 1)] if
                                                            0 <= x + d[0] < self.dimension and 0 <= y + d[
                                                                1] < self.dimension and self.grid[
                                                                x + d[0], y + d[1]] != '#']))

        self.alien_prob_matrix = new_prob_matrix / np.sum(new_prob_matrix)  # Normalize the matrix

    def update_prob_matrices(self, beep_detected, alien_sensed):
        # Temporary matrices to hold the updated probabilities
        new_crew_prob_matrix = np.zeros_like(self.crew_prob_matrix)
        new_alien_prob_matrix = np.zeros_like(self.alien_prob_matrix)

        # Update crew probability matrix using Bayesian updating
        for x in range(self.dimension):
            for y in range(self.dimension):
                # Skip walls
                if self.grid[x, y] == '#':
                    self.crew_prob_matrix[x, y] = 0
                    continue

                exploration_bonus = 0.1  # Value to incentivize exploration; adjust as needed
                distance = self.distance((x, y), self.bot_pos)
                beep_probability = np.exp(-self.alpha * (distance - 1))

                if beep_detected:
                    if distance < self.distance(self.bot_pos, self.crew_pos):
                        # Update based on the likelihood of detecting a beep given the crew is at (x, y)
                        new_crew_prob_matrix[x, y] = self.crew_prob_matrix[x, y] * beep_probability
                    else:
                        new_crew_prob_matrix[x, y] = self.crew_prob_matrix[x, y] * (1 - beep_probability)

                # Apply exploration incentive for unvisited cells
                if self.visited_matrix[x, y] == 0:
                    new_crew_prob_matrix[x, y] *= (1 + exploration_bonus)
                else:
                    # Penalize revisiting cells to discourage backtracking
                    new_crew_prob_matrix[x, y] *= 0.5  # Penalize; adjust penalty as appropriate

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
                        new_alien_prob_matrix[x, y] = self.alien_prob_matrix[x, y] * 1.1
                    else:
                        # Decrease likelihood for positions outside of sensing range
                        new_alien_prob_matrix[x, y] = self.alien_prob_matrix[x, y] * 0.1

            # Normalize the alien probability matrix to ensure probabilities sum to 1
            total_alien_prob = np.sum(new_alien_prob_matrix)
            if total_alien_prob > 0:
                self.alien_prob_matrix = new_alien_prob_matrix / total_alien_prob

    def move_based_on_prob(self):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left
        best_move = None
        best_utility = float('-inf')

        for dx, dy in directions:
            nx, ny = self.bot_pos[0] + dx, self.bot_pos[1] + dy

            # Ensure the move is within bounds and not into a wall.
            if 0 <= nx < self.dimension and 0 <= ny < self.dimension and self.grid[nx, ny] != '#' and self.grid[
                nx, ny] != 'A':
                current_utility = self.calculate_move_utility((nx, ny))
                if (self.grid[nx, ny] == 'C'):
                    best_move = (dx, dy)
                    break

                if current_utility > best_utility:
                    best_utility = current_utility
                    best_move = (dx, dy)

        # Execute the best move if found
        if best_move and best_utility > float('-inf'):
            self.bot_pos = (self.bot_pos[0] + best_move[0], self.bot_pos[1] + best_move[1])
            self.visited_matrix[self.bot_pos] = 1  # Mark as visited
            self.update_grid()  # Update the grid to reflect the new bot position
        else:
            print("Staying in place")

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

    def run(self):
        steps = 0
        while steps <= 10000:
            beep_detected, alien_sensed = self.sense_environment()
            self.update_prob_matrices(beep_detected, alien_sensed)
            self.move_based_on_prob()
            self.move_alien_randomly()
            print(f"Crew Distance: {self.distance(self.bot_pos, self.crew_pos)}")
            print(f"Alien Distance: {self.distance(self.bot_pos, self.alien_pos)}")
            print(f"Beep Detected: {beep_detected}, Alien Sensed: {alien_sensed}")
            print(f"Position Bot: {self.bot_pos}")
            print(f"Position Crew: {self.crew_pos}")
            print(f"Position Alien: {self.alien_pos}")
            print(f"Step: {steps}.")
            self.print_grid()

            # self.print_crew_prob_matrix()

            if self.bot_pos == self.crew_pos:
                print(f"Bot 2 rescued the crew member in {steps} steps!")
                return (True, steps)

            if self.bot_pos == self.alien_pos:
                print(f"Bot 2 was destroyed by the alien after {steps + 1} steps.")
                return (False, steps)

            steps += 1
        return (False, steps)


if __name__ == "__main__":
    bot = Bot2(dimension=35, alpha=0.05, k=1)
    result, steps = bot.run()
    print(result, steps)
