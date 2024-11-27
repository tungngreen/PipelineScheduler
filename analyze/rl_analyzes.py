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
        if not os.path.isdir(os.path.join(directory, agent)): continue
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
                            if 'episodeEnd' in line:
                                if index >= len(rewards[agent]):
                                    rewards[agent].append(float(line.split(',')[5]))
                                else:
                                    # if (rewards[agent][index] < float(line.split(',')[5])):
                                    rewards[agent][index] += float(line.split(',')[5])
                                index += 1
        for i in range(len(rewards[agent])):
            rewards[agent][i] = (rewards[agent][i]) / len(instances)


    # plot the results
    plt.figure()
    for agent in agents:
        plt.plot(rewards[agent], label=agent)
    plt.legend(fontsize=12)
    plt.title('Rewards', size=12)
    plt.xlabel('Episodes', size=12)
    plt.ylabel('Reward', size=12)
    plt.ylim(0, 2)
    plt.show()

def total_performance(directory):
    directory = os.path.join(directory, 'fcpo_results')
    include_people = True
    stepless = True
    fig, axs = plt.subplots(1, 1, figsize=(8, 3), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    ax1 = axs
    cars = []
    dfs_cars = {}
    if include_people:
        people = []
        dfs_people = {}
    for d in natsorted(os.listdir(directory)):
        if not os.path.isdir(os.path.join(directory, d)): continue
        # if the csvs do not exist, create them
        if not os.path.exists(os.path.join(directory, d, 'df_cars.csv')) or not os.path.exists(
                os.path.join(directory, d, 'df_people.csv')):
                for f in natsorted(os.listdir(os.path.join(directory, d))):
                    c, p, _, _ = read_file(os.path.join(directory, d, f))
                    if include_people: people.extend(p)
                    cars.extend(c)
                df_cars = pd.DataFrame(cars, columns=['path', 'latency', 'timestamps'])
                df_cars.to_csv(os.path.join(directory, d, 'df_cars.csv'), index=False)
                dfs_cars[d] = df_cars
                if include_people:
                    df_people = pd.DataFrame(people, columns=['path', 'latency', 'timestamps'])
                    df_people.to_csv(os.path.join(directory, d, 'df_people.csv'), index=False)
                    dfs_people[d] = df_people
        else:
            df_cars = pd.read_csv(os.path.join(directory, d, 'df_cars.csv'))
            dfs_cars[d] = df_cars
            if include_people:
                df_people = pd.read_csv(os.path.join(directory, d, 'df_people.csv'))
                dfs_people[d] = df_people
    for j, d in enumerate(natsorted(os.listdir(directory))):
        dfs_cars[d]['timestamps'] = dfs_cars[d]['timestamps'].apply(lambda x: int(x))
        dfs_cars[d]['latency'] = dfs_cars[d]['latency'].apply(lambda x: int(x))
        dfs_cars[d]['aligned_timestamp'] = (dfs_cars[d]['timestamps'] // 1e6).astype(int) * 1e6
        dfs_cars[d]['throughput'] = dfs_cars[d].groupby('aligned_timestamp')['latency'].transform('size')
        dfs_cars[d]['aligned_timestamp'] = (dfs_cars[d]['aligned_timestamp'] - dfs_cars[d]['aligned_timestamp'].iloc[0]) / (60 * 1e6)
        dfs_cars[d] = dfs_cars[d].groupby('aligned_timestamp').agg({'throughput': 'mean'}).reset_index()
        dfs_cars[d] = bucket_and_average(dfs_cars[d], ['throughput'], num_buckets=780 if stepless else 80)
        # remove values before 10 and offset the time for the remaining values
        dfs_cars[d] = dfs_cars[d][dfs_cars[d]['aligned_timestamp'] > 11]
        dfs_cars[d]['aligned_timestamp'] -= 11
        ax1.plot(dfs_cars[d]['aligned_timestamp'], dfs_cars[d]['throughput'], label=d + ' Traffic', color=colors[j], linewidth=3)
        if include_people:
            if d == 'OURS': j = 2
            dfs_people[d]['timestamps'] = dfs_people[d]['timestamps'].apply(lambda x: int(x))
            dfs_people[d]['latency'] = dfs_people[d]['latency'].apply(lambda x: int(x))
            dfs_people[d]['aligned_timestamp'] = (dfs_people[d]['timestamps'] // 1e6).astype(int) * 1e6
            dfs_people[d]['throughput'] = dfs_people[d].groupby('aligned_timestamp')['latency'].transform('size')
            dfs_people[d]['aligned_timestamp'] = (dfs_people[d]['aligned_timestamp'] - dfs_people[d]['aligned_timestamp'].iloc[0]) / (60 * 1e6)
            dfs_people[d] = dfs_people[d].groupby('aligned_timestamp').agg({'throughput': 'mean'}).reset_index()
            dfs_people[d] = bucket_and_average(dfs_people[d], ['throughput'], num_buckets=780 if stepless else 80)
            # remove values before 10 and offset the time for the remaining values
            dfs_people[d] = dfs_people[d][dfs_people[d]['aligned_timestamp'] > 11]
            dfs_people[d]['aligned_timestamp'] -= 11
            ax1.plot(dfs_people[d]['aligned_timestamp'], dfs_people[d]['throughput'], label=d + ' Surveillance', color=colors[j], linewidth=3, linestyle='--')
    ax1.set_title('Throughput over 8h', size=12)
    ax1.set_ylabel('Throughput (objects / s)', size=12)
    ax1.set_xlabel('Minutes Passed since Start (min)', size=12)
    ax1.set_xlim([0, 480])
    ax1.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
