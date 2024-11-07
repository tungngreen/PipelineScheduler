import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted
from pandas import DataFrame

from final_figures import plot_over_time

def reward_plot(directory):
    directory = os.path.join(directory, 'fcpo_logs')
    agents = natsorted(os.listdir(directory))
    rewards = {}
    for agent in agents:
        rewards_dir = os.path.join(directory, agent, 'latest_log.csv')
        # the file has two different types of rows:
        # step,322,1,60,17.7,0,6,2
        # episodeEnd,28948,1,60,17.7,0.295
        # we are only interested in the rows with the episodeEnd
        rewards[agent] = []
        with open(rewards_dir, 'r') as f:
            for line in f:
                if 'episodeEnd' in line:
                    rewards[agent].append(float(line.split(',')[5]))

    rewards = DataFrame(rewards)
    rewards.plot()
    plt.title('Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()

def total_performance(directory):
    plot_over_time(os.path.join(directory, 'fcpo_results'), inlcude_people=False, missed=False, memory=False, stepless=False)
