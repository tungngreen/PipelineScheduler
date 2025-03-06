import os
import json
import pickle
from math import sqrt
from random import randint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted
from pandas import DataFrame

from final_figures import colors, base_plot
from run_log_analyzes import read_file, get_total_objects
from database_conn import bucket_and_average, align_timestamps, avg_memory


def read_rl_logs(directory, agents, rewards, loss):
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


def reward_plot(base_directory):
    if not os.path.exists(os.path.join(base_directory, 'processed_logs', 'reward-loss.pkl')):
        fcpo = os.path.join(base_directory, 'fcpo_logs')
        fcpo_rewards = {}
        fcpo_loss = {}
        read_rl_logs(fcpo, natsorted(os.listdir(fcpo)), fcpo_rewards, fcpo_loss)

        fcpo_reduced = os.path.join(base_directory, 'fcpo_reduced_logs')
        fcpor_rewards = {}
        fcpor_loss = {}
        read_rl_logs(fcpo_reduced, natsorted(os.listdir(fcpo_reduced)), fcpor_rewards, fcpor_loss)

        fcpo_global = os.path.join(base_directory, 'fcpo_global_logs')
        fcpog_rewards = {}
        fcpog_loss = {}
        read_rl_logs(fcpo_global, natsorted(os.listdir(fcpo_global)), fcpog_rewards, fcpog_loss)

        fcpo_fed = os.path.join(base_directory, 'federated_experiment')
        fcpof_rewards = {}
        fcpof_loss = {}
        read_rl_logs(fcpo_fed, natsorted(os.listdir(fcpo_fed)), fcpof_rewards, fcpof_loss)

        bce = os.path.join(base_directory, 'bce_logs')
        bce_rewards = {}
        bce_loss = {}
        read_rl_logs(bce, natsorted(os.listdir(bce)), bce_rewards, bce_loss)

        # create average reward across all agents per step
        rewards = {}
        loss = {}
        for algo, data in {'FCPO': [fcpo_rewards, fcpo_loss], 'BCE': [bce_rewards, bce_loss], 'FCPO-reduced': [fcpor_rewards, fcpor_loss], 'without local optimization': [fcpog_rewards, fcpog_loss]}.items():
            rewards[algo] = []
            loss[algo] = []
            for agent in data[0]:
                for i in range(len(data[0][agent])):
                    if i >= len(rewards[algo]):
                        rewards[algo].append(data[0][agent][i])
                        loss[algo].append(data[1][agent][i])
                    else:
                        rewards[algo][i] += data[0][agent][i]
                        loss[algo][i] += data[1][agent][i]
            for i in range(len(rewards[algo])):
                rewards[algo][i] = rewards[algo][i] / len(data[0])
                loss[algo][i] = loss[algo][i] / len(data[0])
        # store all results in a pickle file
        with open(os.path.join(base_directory, 'processed_logs', 'reward-loss.pkl'), 'wb') as f:
            pickle.dump([rewards, loss, fcpo_rewards, fcpo_loss, bce_rewards, bce_loss, fcpof_rewards, fcpof_loss, fcpor_rewards, fcpor_loss, fcpog_rewards, fcpog_loss], f)
    else:
        with open(os.path.join(base_directory, 'processed_logs', 'reward-loss.pkl'), 'rb') as f:
            rewards, loss, fcpo_rewards, fcpo_loss, bce_rewards, bce_loss, fcpof_rewards, fcpof_loss, fcpor_rewards, fcpor_loss, fcpog_rewards, fcpog_loss = pickle.load(f)

    # plot the results
    for j, algo in enumerate(['FCPO', 'BCE']):
        fig, ax1 = plt.subplots(1, 1, figsize=(4, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        ax2 = ax1.twinx()
        ax1.plot(rewards[algo], label='reward', color=colors[j], linewidth=1)
        ax2.plot(loss[algo], label='loss', linestyle='--', color=colors[j], linewidth=1)
        fig.legend(fontsize=12)
        ax1.set_xlabel('Episodes', size=12)
        ax1.set_ylabel('Reward Avg.', size=12)
        ax2.set_ylabel('Loss Avg.', size=12)
        ax1.set_xlim(0, 100)
        plt.tight_layout()
        plt.savefig(f"{algo}-learning.svg")
        plt.show()

    fig1, ax1 = plt.subplots(1, 1, figsize=(2.5, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    fig2, ax2 = plt.subplots(1, 1, figsize=(2.5, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    styles = ['-', '--', 'x-']
    for j, algo in enumerate(['FCPO', 'FCPO-reduced', 'without local optimization']):
        ax1.plot(rewards[algo], styles[j], label=algo, color=colors[j], linewidth=1)
        ax2.plot(loss[algo], styles[j], label=algo, color=colors[j], linewidth=1)
    ax1.set_xlabel('Episodes', size=12)
    ax1.set_ylabel('Reward Avg.', size=12)
    ax2.set_xlabel('Episodes', size=12)
    ax2.set_ylabel('Loss Avg.', size=12)
    ax1.set_xlim(0, 50)
    ax2.set_xlim(0, 50)
    fig1.tight_layout()
    fig1.savefig("ablation-rewards.svg")
    fig1.show()
    fig2.tight_layout()
    fig2.savefig("ablation-loss.svg")
    fig2.show()

    for algo, data in {'FCPO-rewards': fcpo_rewards, 'BCE-rewards': bce_rewards, 'FCPO-loss': fcpo_loss,
                       'BCE-loss': bce_loss, 'federated-rewards': fcpof_rewards, 'federated-loss': fcpof_loss}.items():
        if 'federated' in algo:
            fig, ax1 = plt.subplots(1, 1, figsize=(4, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(4, 4), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        i = 0
        for agent, series in data.items():
            ax1.plot(series, label=agent, color=colors[i])
            i += 1
        ax1.legend(fontsize=12)
        ax1.set_xlabel('Episodes', size=12)
        if 'reward' in algo:
            ax1.set_ylabel('Reward', size=12)
        else:
            ax1.set_ylabel('Loss', size=12)
        if 'federated' in algo:
            ax1.set_xlim(0, 40)
        else:
            ax1.set_xlim(0, 100)
        plt.tight_layout()
        plt.savefig(f"{algo}.svg")
        plt.show()


def overall_performance_timeseries(directory, experiment):
    directory = os.path.join(directory, experiment)
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    styles = ['-', '--', 'x-']
    avg_throughput = {'traffic_throughput': {'total': {}}, 'traffic_goodput': {'total': {}}}
    systems = natsorted(os.listdir(directory))
    if 'main' in experiment:
        systems = ['FCPO', 'BCEdge', 'Distream']
    for j, d in enumerate(systems):
        if not os.path.isdir(os.path.join(directory, d)): continue
        if not os.path.exists(os.path.join(directory, d, 'df_cars.csv')) or not os.path.exists(
                os.path.join(directory, d, 'df_people.csv')):
                cars = []
                people = []
                for f in natsorted(os.listdir(os.path.join(directory, d))):
                    c, p, _, _ = read_file(os.path.join(directory, d, f))
                    people.extend(p)
                    cars.extend(c)
                df_cars = pd.DataFrame(cars, columns=['path', 'latency', 'timestamps'])
                df_people = pd.DataFrame(people, columns=['path', 'latency', 'timestamps'])
                df_cars['timestamps'] = df_cars['timestamps'].apply(lambda x: int(x))
                df_people['timestamps'] = df_people['timestamps'].apply(lambda x: int(x))
                df_cars['latency'] = df_cars['latency'].apply(lambda x: int(x) / 1000)
                df_people['latency'] = df_people['latency'].apply(lambda x: int(x) / 1000)
                df_cars['aligned_timestamp'] = (df_cars['timestamps'] // 1e6).astype(int) * 1e6
                df_people['aligned_timestamp'] = (df_people['timestamps'] // 1e6).astype(int) * 1e6
                df_cars.to_csv(os.path.join(directory, d, 'df_cars.csv'), index=False)
                df_people.to_csv(os.path.join(directory, d, 'df_people.csv'), index=False)
        if (not os.path.exists(os.path.join(directory, d, 'combined_df.csv')) or not os.path.exists(
                os.path.join(directory, d, 'avg_throughput.pkl'))):
            df_cars = pd.read_csv(os.path.join(directory, d, 'df_cars.csv'))
            df_people = pd.read_csv(os.path.join(directory, d, 'df_people.csv'))
            dfs_combined = pd.concat([df_cars, df_people])
            # calculate the goodput with ratio of the number of objects with latency < 250 ms
            ratio = dfs_combined[dfs_combined['latency'] < 250]['aligned_timestamp'].count() / dfs_combined['aligned_timestamp'].count()
            dfs_combined['throughput'] = dfs_combined.groupby('aligned_timestamp')['latency'].transform('size')
            dfs_combined['aligned_timestamp'] = (dfs_combined['aligned_timestamp'] - dfs_combined['aligned_timestamp'].iloc[0]) / (60 * 1e6)
            dfs_combined = dfs_combined.groupby('aligned_timestamp').agg({'throughput': 'mean', 'latency': 'mean'}).reset_index()
            dfs_combined = bucket_and_average(dfs_combined, ['throughput', 'latency'], num_buckets=780)
            dfs_combined['aligned_timestamp'] = dfs_combined['aligned_timestamp'] - dfs_combined['aligned_timestamp'].iloc[0]
            dfs_combined['avg_latency'] = dfs_combined['latency']
            avg_throughput['traffic_throughput']['total'][d] = dfs_combined['throughput'].mean()
            avg_throughput['traffic_goodput']['total'][d] = avg_throughput['traffic_throughput']['total'][d] * ratio
            total_traffic_people, total_traffic_cars, total_people_people, total_people_cars = get_total_objects(directory)
            avg_throughput['max_traffic_throughput'] = (
                        total_traffic_people + total_traffic_cars + total_people_people + total_people_cars) / 720
            dfs_combined.to_csv(os.path.join(directory, d, 'combined_df.csv'), index=False)
            with open(os.path.join(directory, d, 'avg_throughput.pkl'), 'wb') as f:
                pickle.dump(avg_throughput, f)
        else:
            with open(os.path.join(directory, d, 'avg_throughput.pkl'), 'rb') as f:
                avg_throughput = pickle.load(f)
            dfs_combined = pd.read_csv(os.path.join(directory, d, 'combined_df.csv'))
        ax1.plot(dfs_combined['aligned_timestamp'], dfs_combined['throughput'], styles[j], label=d,
                 color=colors[j], linewidth=1, markevery=0.2)
        ax2.plot(dfs_combined['aligned_timestamp'], dfs_combined['avg_latency'], styles[j], label=d,
                 color=colors[j], linewidth=1, markevery=0.2)

    if 'main' in experiment:
        experiment = 'total'
    else:
        experiment = 'ablation'
    #ax1.set_title('Throughput over 1h', size=12)
    ax1.set_ylabel('Throughput (objects / s)', size=12)
    ax1.set_xlabel('Minutes Passed since Start (min)', size=12)
    ax1.set_xticks(np.arange(0, 250, 50))
    ax1.set_xlim([0, 250])
    ax1.legend(fontsize=12)
    fig1.tight_layout()
    fig1.savefig(f"{experiment}-throughput.svg")
    fig1.show()
    #ax2.set_title('Average Latency over 1h', size=12)
    ax2.set_ylabel('Avg Latency (ms)', size=12)
    ax2.set_xlabel('Minutes Passed since Start (min)', size=12)
    ax2.set_xticks(np.arange(0, 250, 50))
    ax2.set_xlim([0, 250])
    ax2.set_ylim([0, 900])
    ax2.legend(fontsize=12)
    fig2.tight_layout()
    fig2.savefig(f"{experiment}-latency.svg")
    fig2.show()

    # plot the throughput and latency for each pipeline
    fig, ax1 = plt.subplots(1, 1, figsize=(3, 6), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    base_plot(avg_throughput, ax1, '', True, systems, False)
    plt.tight_layout()
    plt.savefig('total-throughput-comparison.svg')
    plt.show()


def perPipeline_performance(directory):
    directory = os.path.join(directory, 'fcpo_results')
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 3), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    labels = ['Traffic', 'Surveillance', 'Indoor']
    styles = ['-', '--', 'x-']
    for j, d in enumerate(natsorted(os.listdir(directory))):
        if not os.path.isdir(os.path.join(directory, d)): continue
        for f in natsorted(os.listdir(os.path.join(directory, d))):
            if not f.endswith('.txt'): continue
            print(d + "/" + f)
            tmp = []
            c, p, _, _ = read_file(os.path.join(directory, d, f))
            i = 0
            if "indoor" in f:
                i = 2
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
            ax1.plot(df['aligned_timestamp'], df['throughput'], styles[i], label=d + ' ' + labels[i], color=colors[j],
                         linewidth=1)

    ax1.set_title('Throughput over 4h', size=12)
    ax1.set_ylabel('Throughput (objects / s)', size=12)
    ax1.set_xlabel('Minutes Passed since Start (min)', size=12)
    ax1.set_xlim([0, 260])
    ax1.legend(fontsize=12, loc='upper center')
    plt.tight_layout()
    plt.savefig('perPipeline-throughput.svg')
    plt.show()


def limited_network_performance(directory):
    bandwidth_json = json.loads(open(os.path.join(directory, '..', 'jsons', 'bandwidth-limited.json')).read())
    # extract the bandwidth and timestamps from the json
    bandwidth = []
    bandwidth_timestamps = []
    for limit in bandwidth_json['bandwidth_limits']:
        bandwidth.append(limit['mbps'])
        bandwidth_timestamps.append(limit['time'] / 60 - 1.5) # remove system startup time
    directory = os.path.join(directory, 'fcpo_netw')
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    ax2 = ax1.twinx()
    styles = ['-', '--', 'x-']
    data = {}
    for j, d in enumerate(natsorted(os.listdir(directory), reverse=True)):
        if not os.path.exists(os.path.join(directory, 'df.csv')):
            data[d] = []
            for f in natsorted(os.listdir(os.path.join(directory, d))):
                if not f.endswith('.txt'): continue
                c, p, _, _ = read_file(os.path.join(directory, d, f))
                data[d].extend(c)
                data[d].extend(p)
            df = pd.DataFrame(data[d], columns=['path', 'latency', 'timestamps'])
            df['latency'] = df['latency'].apply(lambda x: int(x))
            df['timestamps'] = df['timestamps'].apply(lambda x: int(x))
            data[d] = df
            df = DataFrame()
            data[d]['aligned_timestamp'] = (data[d]['timestamps'] // 1e6).astype(int) * 1e6
            # remove elements with latency > 250 ms
            data[d] = data[d][data[d]['latency'] < 250000]
            df['throughput'] = data[d].groupby('aligned_timestamp')['latency'].transform('size')
            df['aligned_timestamp'] = (data[d]['aligned_timestamp'] - data[d]['aligned_timestamp'].iloc[0]) / (60 * 1e6)
            df = df.groupby('aligned_timestamp').agg({'throughput': 'mean'}).reset_index()
            df = bucket_and_average(df, ['throughput'], num_buckets=180)

            data[d] = align_timestamps([data[d]])
            data[d]['throughput'] = data[d].apply(lambda x: (x['timestamps'] / (x['latency'] / 1000) / 10000000000), axis=1)
            data[d] = bucket_and_average(data[d], ['latency', 'throughput'], num_buckets=180)

            df['throughput'] = (df['throughput'] + data[d]['throughput']) / 2
            # store df to csv
            df.to_csv(os.path.join(directory, d, 'df.csv'), index=False)
        else:
            df = pd.read_csv(os.path.join(directory, d, 'df.csv'))
        ax1.plot(df['aligned_timestamp'], df['throughput'], styles[j], label=d, color=colors[j], linewidth=1,
                 markevery=0.2)

    ax1.set_xlabel('Minutes Passed since Start (min)', size=12)
    ax1.set_ylabel('Throughput (objects / s)', size=12)
    ax1.set_xlim([0, 31.5])
    ax1.legend(fontsize=12)

    ax2.plot(bandwidth_timestamps, bandwidth, label='Bandwidth Limit', color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Bandwidth Limit (Mb/s)', size=12)
    ax2.set_ylim([0, 170])
    plt.tight_layout()
    plt.savefig('network-throughput.svg')
    plt.show()


def continual_learning_performance(base_directory):
    directory = os.path.join(base_directory, 'fcpo_continual')
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    styles = ['-', '--', 'x-']
    data = {}
    # add a grey vertical line every 5 minutes
    for i in range(1, 50):
        ax1.axvline(x=i * 5, color='grey', linestyle='-', linewidth=0.5)
    for j, d in enumerate(natsorted(os.listdir(directory))):
        if not os.path.exists(os.path.join(directory, d, 'df.csv')):
            data[d] = []
            for f in natsorted(os.listdir(os.path.join(directory, d))):
                c, p, _, _ = read_file(os.path.join(directory, d, f))
                data[d].extend(c)
                data[d].extend(p)
            df = pd.DataFrame(data[d], columns=['path', 'latency', 'timestamps'])
            df['latency'] = df['latency'].apply(lambda x: int(x))
            df['timestamps'] = df['timestamps'].apply(lambda x: int(x))
            data[d] = df
            df = DataFrame()
            data[d]['aligned_timestamp'] = (data[d]['timestamps'] // 1e6).astype(int) * 1e6
            # remove elements with latency > 250 ms
            data[d] = data[d][data[d]['latency'] < 250000]
            df['throughput'] = data[d].groupby('aligned_timestamp')['latency'].transform('size')
            df['aligned_timestamp'] = (data[d]['aligned_timestamp'] - data[d]['aligned_timestamp'].iloc[0]) / (60 * 1e6)
            df = df.groupby('aligned_timestamp').agg({'throughput': 'mean'}).reset_index()
            df = bucket_and_average(df, ['throughput'], num_buckets=180)

            data[d] = align_timestamps([data[d]])
            data[d]['throughput'] = data[d].apply(lambda x: (x['timestamps'] / (x['latency'] / 1000) / 10000000000), axis=1)
            data[d] = bucket_and_average(data[d], ['latency', 'throughput'], num_buckets=180)

            df['throughput'] = (df['throughput'] + data[d]['throughput']) / 2
            df.to_csv(os.path.join(directory, d, 'df.csv'), index=False)
        else:
            df = pd.read_csv(os.path.join(directory, d, 'df.csv'))
        ax1.plot(df['aligned_timestamp'], df['throughput'], styles[j], label=d, color=colors[j], linewidth=1,
                 markevery=0.2)

    ax1.set_xlabel('Minutes Passed since Start (min)', size=12)
    ax1.set_ylabel('Throughput (objects / s)', size=12)
    ax1.set_xlim([0, 49])
    ax1.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('continual-throughput.svg')
    plt.show()

    fcpo = os.path.join(base_directory, 'fcpo_continual_logs')
    fcpo_rewards = {}
    fcpo_loss = {}
    read_rl_logs(fcpo, natsorted(os.listdir(fcpo)), fcpo_rewards, fcpo_loss)
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    for series in fcpo_rewards.values():
        ax1.plot(series, color=colors[0])
    ax1.set_xlabel('Episodes', size=12)
    ax1.set_ylabel('Reward', size=12)
    ax1.set_xlim(0, 23)
    plt.tight_layout()
    plt.savefig(f"continual-rewards.svg")
    plt.show()


def system_overhead(directory):
    # Memory
    systems = ['fcpo', 'bce', 'ppp']
    labels = ['FCPO', 'BCE', 'Distream']
    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    if not os.path.exists(os.path.join(directory, '..', 'processed_logs', 'memory.pkl')):
        server_memory = avg_memory('full', systems, 0)
        edge_memory = avg_memory('full', systems, 1)
        with open(os.path.join(directory, '..', 'processed_logs', 'memory.pkl'), 'wb') as f:
            pickle.dump([server_memory, edge_memory], f)
    else:
        with open(os.path.join(directory, '..', 'processed_logs', 'memory.pkl'), 'rb') as f:
            server_memory, edge_memory = pickle.load(f)
    for i, system in enumerate(systems):
        ax1.bar(i, (server_memory[system][0] + server_memory[system][1]) / 1024, 0.7, label=labels[i], color=colors[i])
        ax2.bar(i, edge_memory[system][0] / (1024 * 1024), 0.7, label=labels[i], color=colors[i])
        # add black line at the top of the bar
        ax1.plot([i - 0.35, i + 0.35], [(server_memory['ppp'][0] + server_memory['ppp'][1]) / 1024,
                                         (server_memory['ppp'][0] + server_memory['ppp'][1]) / 1024], color='black')
        ax2.plot([i - 0.35, i + 0.35], [edge_memory['ppp'][0] / (1024 * 1024), edge_memory['ppp'][0] / (1024 * 1024)], color='black')
    ax1.set_ylabel('Mem Usage (GB)', size=12)
    ax1.set_xticks([])
    ax2.legend(fontsize=12, loc='lower center')
    ax2.set_ylabel('Mem Usage (GB)', size=12)
    ax2.set_xticks([])
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig('server-memory.svg')
    fig2.savefig('edge-memory.svg')
    fig1.show()
    fig2.show()

    # RL Latency
    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    x_labels = ['Server', 'AGX', 'NX', 'Nano', 'OnPrem']
    x = np.arange(len(x_labels))
    data = json.loads(open(os.path.join(directory, 'latency.json')).read())
    i = 0
    for system, values in data.items():
        j = 0
        for value in values['infer'].values():
            ax1.bar(j + i * 0.4, value, 0.4, label=system, color=colors[i])
            j += 1
        j = 0
        for value in values['update'].values():
            ax2.bar(j + i * 0.4, value, 0.4, label=system, color=colors[i])
            j += 1
        i += 1

    ax1.set_xticks(x + 0.2)
    ax1.set_xticklabels(x_labels, size=12)
    ax1.set_ylabel('Decision Latency (ms)  _', size=12)
    ax2.set_xticks(x + 0.2)
    ax2.set_xticklabels(x_labels, size=12)
    ax2.set_ylabel('Update Latency (ms)  ', size=12)
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig('inference-latency.svg')
    fig2.savefig('update-latency.svg')
    fig1.show()
    fig2.show()