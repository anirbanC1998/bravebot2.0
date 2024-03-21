import numpy as np
from scipy.stats import poisson
from collections import deque
import random

#init roomba
class SpaceRoomba:
    def __init__(self, ship_layout, k, alpha):
        self.ship_layout = ship_layout
        self.k = k
        self.alpha = alpha
        self.bot_position = self.place_bot()
        self.crew_positions = self.place_crew()
        self.alien_positions = self.place_aliens()
        self.alien_detected = False
        self.moves = 0

#place roomba in 2d matrix
    def place_bot(self):
        while True:
            x, y = np.random.randint(len(self.ship_layout)), np.random.randint(len(self.ship_layout[0]))
            if self.ship_layout[x][y] == 0:
                return x, y

#place crew in matrix
    def place_crew(self):
        crew_positions = []
        for _ in range(2):
            while True:
                x, y = np.random.randint(len(self.ship_layout)), np.random.randint(len(self.ship_layout[0]))
                if self.ship_layout[x][y] == 0 and (x, y) != self.bot_position:
                    crew_positions.append((x, y))
                    break
        return crew_positions
    
#place aliens in matrix
    def place_aliens(self):
        alien_positions = []
        while len(alien_positions) < 1:
            x, y = np.random.randint(len(self.ship_layout)), np.random.randint(len(self.ship_layout[0]))
            if self.ship_layout[x][y] == 0 and self.is_outside_detection_square(x, y, self.bot_position):
                alien_positions.append((x, y))
        return alien_positions

#bool check for creating aliens to be outside bot radius
    def is_outside_detection_square(self, x, y, bot_position):
        bx, by = bot_position
        return abs(x - bx) > self.k or abs(y - by) > self.k

#bool check to sense aliens, manhatten distance
    def sense_aliens(self):
        bx, by = self.bot_position
        for ax, ay in self.alien_positions:
            if abs(ax - bx) <= self.k and abs(ay - by) <= self.k:
                self.alien_detected = True
                break

#beep for crew, random prob for now
    def sense_crew(self):
        bx, by = self.bot_position
        for cx, cy in self.crew_positions:
            d = abs(cx - bx) + abs(cy - by)
            p_detection = np.exp(-self.alpha * (d - 1))
            if np.random.uniform(0, 1) < p_detection:
                return True
        return False

# update bot position
    def update_bot_position(self, new_position):
        self.bot_position = new_position

#update alien position in matrix
    def update_alien_positions(self):
        new_alien_positions = []
        for ax, ay in self.alien_positions:
            possible_moves = [(ax+i, ay+j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i != 0 or j != 0) and self.ship_layout[ax+i][ay+j] == 0]
            if possible_moves:
                new_alien_positions.append(random.choice(possible_moves))
            else:
                new_alien_positions.append((ax, ay))
        self.alien_positions = new_alien_positions

#indicate whether bot reaches crew, or move towards crew, or flee, bool check
    def bot_action(self):
        if self.sense_crew():
            return "rescue"
        elif not self.alien_detected:
            return "move_towards_crew"
        else:
            return "flee"

#main conditional of the bot, ends when all crew is rescused 
    def bot_move(self):
        action = self.bot_action()
        if action == "rescue":
            return
        elif action == "move_towards_crew":
            self.move_towards_crew()
        elif action == "flee":
            self.flee()

#move towards crew, bot 1 strat
    def move_towards_crew(self):
        bx, by = self.bot_position
        possible_moves = [(bx+i, by+j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i != 0 or j != 0) and self.ship_layout[bx+i][by+j] == 0]
        max_prob = 0
        best_move = None
        for move in possible_moves:
            prob = self.calculate_crew_probability(move)
            if prob > max_prob:
                max_prob = prob
                best_move = move
        if best_move:
            self.update_bot_position(best_move)
            self.moves += 1

#main prob threshold to move towards crew
    def calculate_crew_probability(self, position):
        
        return -1

#flee from alien, need prob threshold to balance with crew
    def flee(self):
        bx, by = self.bot_position
        possible_moves = [(bx+i, by+j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i != 0 or j != 0) and self.ship_layout[bx+i][by+j] == 0]
        min_prob = 1
        best_move = None
        for move in possible_moves:
            prob = self.calculate_alien_probability(move)
            if prob < min_prob:
                min_prob = prob
                best_move = move
        if best_move:
            self.update_bot_position(best_move)
            self.moves += 1

#main prob threshold to flee from aliens
    def calculate_alien_probability(self, position):
       
        return -1

def main():
    # Define ship layout
    ship_layout = np.zeros((35, 35))

    # Define constants
    k = 3
    alpha = 0.5

    # Initialize SpaceRoomba
    space_roomba = SpaceRoomba(ship_layout, k, alpha)

    # Bot 1
    print("Bot 1:")
    while len(space_roomba.crew_positions) > 0:
        space_roomba.sense_aliens()
        space_roomba.bot_move()
        print("Bot position:", space_roomba.bot_position)
        print("Moves:", space_roomba.moves)

    # Bot 2 (Custom bot)
    print("\nBot 2:")
    while len(space_roomba.crew_positions) > 0:
        space_roomba.sense_aliens()
       # space_roomba.move_towards_crew()
        print("Bot position:", space_roomba.bot_position)
        print("Moves:", space_roomba.moves)

if __name__ == "__main__":
    main()
