import matplotlib.pyplot as plt
import numpy as np


from Bot1 import Bot1
from Bot2 import Bot2

def simulate_bot1(alpha_range=np.linspace(0.001, 0.01, 10), num_trials = 50):
    success_rates = []
    average_moves = []
    for a in alpha_range:
        successes = 0
        total_moves = 0
        for _ in range(num_trials):
            bot = Bot1(dimension=35, alpha=a, k=1)
            result = bot.run()  
            if result[0]:
                successes += 1
            total_moves += result[1]
        success_rates.append(successes / num_trials)
        average_moves.append(total_moves / num_trials)
    return alpha_range, success_rates, average_moves

def simulate_bot2(alpha_range=np.linspace(0.001, 0.01, 10), num_trials = 50):
    success_rates = []
    average_moves = []
    for a in alpha_range:
        successes = 0
        total_moves = 0
        for _ in range(num_trials):
            bot = Bot2(dimension=35, alpha=a, k=1)
            result = bot.run() 
            if result[0]:
                successes += 1
            total_moves += result[1]
        success_rates.append(successes / num_trials)
        average_moves.append(total_moves / num_trials)
    return alpha_range, success_rates, average_moves


def plot_performance(alpha_range, success_rates, average_moves, label):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Success Rate', color=color)
    ax1.plot(alpha_range, success_rates, color=color, label=f'Success Rate - {label}')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Average Moves', color=color)  
    ax2.plot(alpha_range, average_moves, color=color, linestyle='--', label=f'Average Moves - {label}')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title(f'Performance of {label}')
    plt.show()

# Parameters
num_trials = 100
alpha_range = np.linspace(0.001, 0.01, 10)


# Simulate and plot for Bot1
alpha_range, success_rates, average_moves = simulate_bot1(alpha_range, num_trials)
plot_performance(alpha_range, success_rates, average_moves, 'Bot1')

# Simulate and plot for Bot2
alpha_range, success_rates, average_moves = simulate_bot2(alpha_range, num_trials)
plot_performance(alpha_range, success_rates, average_moves, 'Bot2')
