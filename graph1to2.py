import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from Bot1 import Bot1
from Bot2 import Bot2

# Simulation function for a single bot
def simulate_bot(bot_class, alpha, k, num_trials):
    success_count = 0
    total_moves = 0

    for _ in range(num_trials):
        bot = bot_class(dimension = 35,alpha=alpha, k=k)
        success, moves = bot.run()  # Assume .run() returns success, moves, no need for crew saved, crew saved = success
        if success:
            success_count += 1
        total_moves += moves
        
    success_rate = success_count / num_trials
    avg_moves = total_moves / num_trials

    return success_rate, avg_moves

# Function to execute simulations across a range of alpha values using threading
def simulate_for_alpha_range(bot_class, k, alpha_range, num_trials):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(simulate_bot, bot_class, alpha, k, num_trials) for alpha in alpha_range]
        results = [f.result() for f in futures]
    return results  # List of tuples (success_rate, avg_moves, avg_crew_saved)

# Function to simulate and plot results for bots, considering multiple k values and alpha range
def simulate_and_plot(bot_classes, k_values, alpha_range, num_trials):
    for k in k_values:
        plt.figure(figsize=(15, 10))

        for bot_class in bot_classes:
            results = simulate_for_alpha_range(bot_class, k, alpha_range, num_trials)
            success_rates, avg_moves = zip(*results)  # Unpack results

            # Plotting
            plt.subplot(3, 1, 1)
            plt.plot(alpha_range, success_rates, label=f'{bot_class.__name__} k={k}')
            plt.xlabel('Alpha')
            plt.ylabel('Success Rate')
            plt.title('Success Rate vs Alpha')

            plt.subplot(3, 1, 2)
            plt.plot(alpha_range, avg_moves, label=f'{bot_class.__name__} k={k}')
            plt.xlabel('Alpha')
            plt.ylabel('Average Moves')
            plt.title('Average Moves vs Alpha')

        plt.legend()
        plt.tight_layout()
        plt.savefig('Bot12_performance.png', dpi=300)
        plt.show()

if __name__ == "__main__":
    bot_classes = [Bot1, Bot2]
    k_values = [1, 3 , 5]  # Example k values
    alpha_range = np.linspace(0.001, 0.1, 10)  # Example alpha range
    num_trials = 25 # Adjust based on your needs

    simulate_and_plot(bot_classes, k_values, alpha_range, num_trials)
