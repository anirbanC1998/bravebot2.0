import numpy as np
import random
from environment import SpaceRoombaEnvironment

class Bot2(SpaceRoombaEnvironment):
    def run(self):
        steps = 0
        while True:
            self.update_prob_matrix_after_move()
            # Check for success before moving
            if self.bot_pos == self.crew_pos:
                print(f"Bot 2 rescued the crew member in {steps} steps!")
                break

            self.decide_move_based_on_utility()
            self.move_alien_randomly()

            print(f"Step {steps}:")
            self.print_grid()

            # Check for alien encounter after moving
            if self.bot_pos == self.alien_pos:
                print(f"Bot 2 was destroyed by the alien after {steps} steps.")
                break
            
            steps += 1

    def decide_move_based_on_utility(self):
        best_move = self.bot_pos  # Start with the current position as the default
        best_utility = float('-inf')
        
        for dx, dy in [(0, -1), (-1, 0), (1, 0), (0, 1), (0, 0)]:
            next_pos = (self.bot_pos[0] + dx, self.bot_pos[1] + dy)
            if 0 <= next_pos[0] < self.dimension and 0 <= next_pos[1] < self.dimension:
                utility = self.calculate_utility(next_pos)
                if utility > best_utility:
                    best_utility = utility
                    best_move = next_pos

        self.bot_pos = best_move
        self.update_grid()
        
    def calculate_utility(self, pos):
        # Example adjusted utility calculation:
        crew_prob = self.prob_matrix[pos]
        alien_distance = max(1, self.distance(pos, self.alien_pos))  # Avoid division by zero
        
        # Simplified utility: Increase reward for crew probability and distance from the alien
        utility = crew_prob * 100 - 10 / alien_distance
        
        # You might want to add a small randomness to break ties and encourage exploration
        utility += random.uniform(-0.1, 0.1)
        
        return utility

if __name__ == "__main__":
    bot = Bot2()
    bot.run()
