import os
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted
from pandas import DataFrame

from final_figures import colors, plot_over_time
from run_log_analyzes import read_file
from database_conn import bucket_and_average

def reward_plot(directory):
    directory = os.path.join(directory, 'fcpo_logs')
    agents = natsorted(os.listdir(directory))
    rewards = {}
    for agent in agents:
        rewards[agent] = []
        instances = natsorted(os.listdir(os.path.join(directory, agent)))
        for instance in instances:
            logs = natsorted(os.listdir(os.path.join(directory, agent, instance)))
            index = 0;
            for log in logs:
                if 'latest_log' in log:
                    rewards_dir = os.path.join(directory, agent, instance, log)
                    # the file has two different types of rows:
                    # step,322,1,60,17.7,0,6,2
                    # episodeEnd,28948,1,60,17.7,0.295
                    # we are only interested in the rows with the episodeEnd
                    with open(rewards_dir, 'r') as f:
                        for line in f:
                            if 'step' in line:
                                if index >= len(rewards[agent]):
                                    rewards[agent].append(float(line.split(',')[4]))
                                else:
                                    # if (rewards[agent][index] < float(line.split(',')[5])):
                                    rewards[agent][index] += float(line.split(',')[4])
                                index += 1
        for i in range(len(rewards[agent])):
            rewards[agent][i] = (rewards[agent][i]) / len(instances)


    # plot the results
    plt.figure()
    for agent in agents:
        plt.plot(rewards[agent], label=agent)
    plt.legend()
    plt.title('Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()

def total_performance(directory):
    directory = os.path.join(directory, 'fcpo_results')
    include_people = True
    stepless = True
    fig, axs = plt.subplots(1, 1, figsize=(8, 3), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    ax1 = axs
    cars = []
    if include_people: people = []
    # if the csvs do not exist, create them
    if not os.path.exists(os.path.join(directory, 'df_cars.csv')) or not os.path.exists(
            os.path.join(directory, 'df_people.csv')):
        for f in natsorted(os.listdir(os.path.join(directory, 'OURS'))):
            c, p, _, _ = read_file(os.path.join(directory, 'OURS', f))
            if include_people: people.extend(p)
            cars.extend(c)
        df_cars = pd.DataFrame(cars, columns=['path', 'latency', 'timestamps'])
        df_cars.to_csv(os.path.join(directory, 'df_cars.csv'), index=False)
        if include_people:
            df_people = pd.DataFrame(people, columns=['path', 'latency', 'timestamps'])
            df_people.to_csv(os.path.join(directory, 'df_people.csv'), index=False)
    else:
        df_cars = pd.read_csv(os.path.join(directory, 'df_cars.csv'))
        if include_people: df_people = pd.read_csv(os.path.join(directory, 'df_people.csv'))
    lables = ['Traffic Throughput', 'Surveillance Throughput']
    for j, df in enumerate([df_cars, df_people] if include_people else [df_cars]):
        df['timestamps'] = df['timestamps'].apply(lambda x: int(x))
        df['latency'] = df['latency'].apply(lambda x: int(x))
        df['aligned_timestamp'] = (df['timestamps'] // 1e6).astype(int) * 1e6
        df['throughput'] = df.groupby('aligned_timestamp')['latency'].transform('size')
        df['aligned_timestamp'] = (df['aligned_timestamp'] - df['aligned_timestamp'].iloc[0]) / (60 * 1e6)
        df = df.groupby('aligned_timestamp').agg({'throughput': 'mean'}).reset_index()
        df = bucket_and_average(df, ['throughput'], num_buckets=780 if stepless else 80)
        # remove values before 10 and offset the time for the remaining values
        df = df[df['aligned_timestamp'] > 11]
        df['aligned_timestamp'] -= 11
        ax1.plot(df['aligned_timestamp'], df['throughput'], label=lables[j], color=colors[j], linewidth=3)
    ax1.set_title('Throughput over 8h', size=12)
    ax1.set_ylabel('Throughput (objects / s)', size=12)
    ax1.set_xlabel('Minutes Passed since Start (min)', size=12)
    ax1.set_xlim([0, 480])
    ax1.legend(loc='lower left', fontsize=12)
    plt.tight_layout()
    plt.show()
