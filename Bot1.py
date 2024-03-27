import numpy as np
from environment import SpaceRoombaEnvironment

class Bot1(SpaceRoombaEnvironment):
    def run(self):
        steps = 0
        while True:
            self.update_prob_matrix_after_move()
            # Check for success before moving
            if self.bot_pos == self.crew_pos:
                print(f"Bot 1 rescued the crew member in {steps} steps!")
                break
            
            self.move_based_on_prob()
            self.move_alien_randomly()
            
            print(f"Step {steps}:")
            self.print_grid()

            # Check for alien encounter after moving
            if self.bot_pos == self.alien_pos:
                print(f"Bot 1 was destroyed by the alien after {steps} steps.")
                break

            steps += 1

    def move_based_on_prob(self):
        target_pos = np.unravel_index(self.prob_matrix.argmax(), self.prob_matrix.shape)
        dx = np.sign(target_pos[0] - self.bot_pos[0])
        dy = np.sign(target_pos[1] - self.bot_pos[1])
        next_pos = (self.bot_pos[0] + dx, self.bot_pos[1] + dy)
        if 0 <= next_pos[0] < self.dimension and 0 <= next_pos[1] < self.dimension:
            self.bot_pos = next_pos
        self.update_grid()

if __name__ == "__main__":
    bot = Bot1()
    bot.run()

