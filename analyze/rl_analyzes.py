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
    loss = {}
    for agent in agents:
        if not os.path.isdir(os.path.join(directory, agent)): continue
        rewards[agent] = []
        loss[agent] = []
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
                                    rewards[agent][index] += float(line.split(',')[5])
                                if index >= len(loss[agent]):
                                    loss[agent].append(float(line.split(',')[6]))
                                else:
                                    loss[agent][index] += float(line.split(',')[6])
                                index += 1
        for i in range(len(rewards[agent])):
            rewards[agent][i] = (rewards[agent][i]) / len(instances)
            loss[agent][i] = (loss[agent][i]) / len(instances)


    # plot the results
    plt.figure()
    for agent in agents:
        if not os.path.isdir(os.path.join(directory, agent)): continue
        #if agent == 'emotionnet' or agent == 'gender' or agent == 'age': continue
        plt.plot(rewards[agent], label=agent)
    plt.legend(fontsize=12, loc='lower center')
    plt.title('Rewards', size=12)
    plt.xlabel('Episodes', size=12)
    plt.ylabel('Reward', size=12)
    plt.xlim(0, 80)
    #plt.savefig('main-rewards.svg')
    plt.show()

    plt.figure()
    for agent in agents:
        if not os.path.isdir(os.path.join(directory, agent)): continue
        #if agent == 'emotionnet' or agent == 'gender' or agent == 'age': continue
        plt.plot(loss[agent], label=agent)
    plt.legend(fontsize=12)
    plt.title('Loss', size=12)
    plt.xlabel('Episodes', size=12)
    plt.ylabel('Loss', size=12)
    plt.xlim(0, 80)
    #plt.savefig('main-loss.svg')
    plt.show()

def overall_performance(directory):
    directory = os.path.join(directory, 'fcpo_results')
    fig, axs = plt.subplots(1, 1, figsize=(8, 3), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    ax1 = axs
    dfs_cars = {}
    dfs_people = {}
    for d in natsorted(os.listdir(directory)):
        if not os.path.isdir(os.path.join(directory, d)): continue
        # if the csvs do not exist, create them
        if not os.path.exists(os.path.join(directory, d, 'df_cars.csv')) or not os.path.exists(
                os.path.join(directory, d, 'df_people.csv')):
                cars = []
                people = []
                for f in natsorted(os.listdir(os.path.join(directory, d))):
                    c, p, _, _ = read_file(os.path.join(directory, d, f))
                    people.extend(p)
                    cars.extend(c)
                df_cars = pd.DataFrame(cars, columns=['path', 'latency', 'timestamps'])
                df_cars.to_csv(os.path.join(directory, d, 'df_cars.csv'), index=False)
                dfs_cars[d] = df_cars
                df_people = pd.DataFrame(people, columns=['path', 'latency', 'timestamps'])
                df_people.to_csv(os.path.join(directory, d, 'df_people.csv'), index=False)
                dfs_people[d] = df_people
        else:
            df_cars = pd.read_csv(os.path.join(directory, d, 'df_cars.csv'))
            dfs_cars[d] = df_cars
            df_people = pd.read_csv(os.path.join(directory, d, 'df_people.csv'))
            dfs_people[d] = df_people
    for j, d in enumerate(natsorted(os.listdir(directory))):
        dfs_cars[d]['timestamps'] = dfs_cars[d]['timestamps'].apply(lambda x: int(x))
        dfs_people[d]['timestamps'] = dfs_people[d]['timestamps'].apply(lambda x: int(x))
        dfs_cars[d]['latency'] = dfs_cars[d]['latency'].apply(lambda x: int(x))
        dfs_people[d]['latency'] = dfs_people[d]['latency'].apply(lambda x: int(x))
        dfs_cars[d]['aligned_timestamp'] = (dfs_cars[d]['timestamps'] // 1e6).astype(int) * 1e6
        dfs_people[d]['aligned_timestamp'] = (dfs_people[d]['timestamps'] // 1e6).astype(int) * 1e6
        dfs_combined = pd.concat([dfs_cars[d], dfs_people[d]])
        dfs_combined['throughput'] = dfs_combined.groupby('aligned_timestamp')['latency'].transform('size')
        dfs_combined['aligned_timestamp'] = (dfs_combined['aligned_timestamp'] - dfs_combined['aligned_timestamp'].iloc[0]) / (60 * 1e6)
        dfs_combined = dfs_combined.groupby('aligned_timestamp').agg({'throughput': 'mean'}).reset_index()
        dfs_combined = bucket_and_average(dfs_combined, ['throughput'], num_buckets=780)
        ax1.plot(dfs_combined['aligned_timestamp'], dfs_combined['throughput'], label=d, color=colors[j], linewidth=3)
    ax1.set_title('Throughput over 1h', size=12)
    ax1.set_ylabel('Throughput (objects / s)', size=12)
    ax1.set_xlabel('Minutes Passed since Start (min)', size=12)
    ax1.set_xlim([0, 70])
    ax1.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('total-throughput.svg')
    plt.show()

def perPipeline_performance(directory):
    directory = os.path.join(directory, 'fcpo_results')
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 3), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    labels = ['Traffic', 'Surveillance', 'Indoor']
    styles = ['-', '--', '--']
    for d in natsorted(os.listdir(directory)):
        if not os.path.isdir(os.path.join(directory, d)): continue
        for f in natsorted(os.listdir(os.path.join(directory, d))):
            if not f.endswith('.txt'): continue
            print(d + "/" + f)
            tmp = []
            c, p, _, _ = read_file(os.path.join(directory, d, f))
            i = 0
            j = 1
            if "indoor" in f:
                i = 2
                j = 3
                tmp.extend(p)
                tmp.extend(c)
                df = pd.DataFrame(tmp, columns=['path', 'latency', 'timestamps'])
            elif "people" in f:
                i = 1
                tmp.extend(p)
                tmp.extend(c)
                df = pd.DataFrame(tmp, columns=['path', 'latency', 'timestamps'])
            else:
                tmp.extend(c)
                tmp.extend(p)
                df = pd.DataFrame(tmp, columns=['path', 'latency', 'timestamps'])
            df['timestamps'] = df['timestamps'].apply(lambda x: int(x))
            df['latency'] = df['latency'].apply(lambda x: int(x))
            df['aligned_timestamp'] = (df['timestamps'] // 1e6).astype(int) * 1e6
            df['throughput'] = df.groupby('aligned_timestamp')['latency'].transform('size')
            df['aligned_timestamp'] = (df['aligned_timestamp'] - df['aligned_timestamp'].iloc[0]) / (60 * 1e6)
            df = df.groupby('aligned_timestamp').agg({'throughput': 'mean'}).reset_index()
            df = bucket_and_average(df, ['throughput'], num_buckets=780)
            if d == 'FCPO':
                j = 0
                if "indoor" in f:
                    j = 2
            ax1.plot(df['aligned_timestamp'], df['throughput'], label=d + ' ' + labels[i], color=colors[j],
                         linewidth=3, linestyle=styles[i])

    ax1.set_title('Throughput over 4h', size=12)
    ax1.set_ylabel('Throughput (objects / s)', size=12)
    ax1.set_xlabel('Minutes Passed since Start (min)', size=12)
    ax1.set_xlim([0, 260])
    ax1.legend(fontsize=12, loc='upper center')
    plt.tight_layout()
    plt.savefig('perPipeline-throughput.svg')
    plt.show()
