import numpy as np
import random


class SpaceRoombaGame:
    def __init__(self, dimension=35, k=1, alpha=0.5):
        self.dimension = dimension
        self.grid = np.zeros((dimension, dimension))
        self.k = k
        self.alpha = alpha
        # Positions
        self.bot_pos = self.place_entity()
        self.crew_pos = self.place_entity()
        self.alien_pos = self.place_entity(outside_range=self.k)

    def place_entity(self, outside_range=None):
        while True:
            pos = (random.randint(0, self.dimension - 1), random.randint(0, self.dimension - 1))
            if self.grid[pos] == 0:
                if outside_range and self.bot_pos:
                    if abs(pos[0] - self.bot_pos[0]) <= outside_range * 2 and abs(
                            pos[1] - self.bot_pos[1]) <= outside_range * 2:
                        continue
                return pos

    def move_bot(self):
        directions = [(0, -1), (-1, 0), (1, 0), (0, 1), (0, 0)]  # Up, Left, Right, Down, Stay
        best_move = None
        min_distance = float('inf')
        for direction in directions:
            new_pos = (self.bot_pos[0] + direction[0], self.bot_pos[1] + direction[1])
            if 0 <= new_pos[0] < self.dimension and 0 <= new_pos[1] < self.dimension:
                distance = abs(new_pos[0] - self.crew_pos[0]) + abs(new_pos[1] - self.crew_pos[1])
                if distance < min_distance:
                    min_distance = distance
                    best_move = new_pos
        self.bot_pos = best_move

    def move_alien(self):
        directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
        random.shuffle(directions)
        for direction in directions:
            new_pos = (self.alien_pos[0] + direction[0], self.alien_pos[1] + direction[1])
            if 0 <= new_pos[0] < self.dimension and 0 <= new_pos[1] < self.dimension:
                self.alien_pos = new_pos
                break

    def run(self):
        steps = 0
        while True:
            self.move_bot()
            self.move_alien()
            steps += 1

            # Check for game end conditions
            if self.bot_pos == self.crew_pos:
                print(f"Rescued the crew member in {steps} steps!")
                break
            elif self.bot_pos == self.alien_pos:
                print(f"Encountered the alien and was destroyed after {steps} steps.")
                break


# Run the simulation
game = SpaceRoombaGame(k=3, alpha=0.5)  # Example k and alpha values
game.run()
