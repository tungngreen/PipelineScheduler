import os
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from natsort import natsorted
from pandas import DataFrame

from final_figures import colors, base_plot, individual_figures
from objectcount import load_data
from run_log_analyzes import read_file, get_total_objects, analyze_single_experiment, get_bandwidths
from database_conn import bucket_and_average, align_timestamps, avg_memory, avg_edge_power

styles = ['-', '--', 'o-', 'x-']

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
        with open(os.path.join(base_directory, 'processed_logs', 'reward-loss.pkl'), 'wb') as f:
            pickle.dump([rewards, loss, fcpo_rewards, fcpo_loss, bce_rewards, bce_loss, fcpof_rewards, fcpof_loss, fcpor_rewards, fcpor_loss, fcpog_rewards, fcpog_loss], f)
    else:
        with open(os.path.join(base_directory, 'processed_logs', 'reward-loss.pkl'), 'rb') as f:
            rewards, loss, fcpo_rewards, fcpo_loss, bce_rewards, bce_loss, fcpof_rewards, fcpof_loss, fcpor_rewards, fcpor_loss, fcpog_rewards, fcpog_loss = pickle.load(f)

    # plot the results
    for j, algo in enumerate(['FCPO', 'BCE']):
        fig, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        ax2 = ax1.twinx()
        ax1.plot(rewards[algo], label='reward', color=colors[j*2], linewidth=1)
        ax2.plot(loss[algo], label='loss', linestyle='--', color=colors[j*2+1], linewidth=1)
        #fig.legend(fontsize=12, loc='upper center')
        ax1.set_ylabel(r'Reward $\bullet$  ', size=13, color=colors[j*2])
        ax1.set_ylim([0, 1])
        ax1.set_yticks([0, 1])

        ax2.set_ylabel(r'Loss $\circ$', size=13, color=colors[j*2+1])
        ax1.set_xlim(0, 100)
        ax1.set_xticks([0, 30, 90])
        ax1.set_xticklabels([0, 30, '90 Episodes'], size=12)
        plt.tight_layout()
        plt.savefig(f"{algo}-learning.pdf")
        plt.show()

    fig1, ax1 = plt.subplots(1, 1, figsize=(2.5, 2.7), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    fig2, ax2 = plt.subplots(1, 1, figsize=(2.5, 2.7), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    for j, algo in enumerate(['FCPO', 'FCPO-reduced', 'without local optimization']):
        ax1.plot(rewards[algo], styles[j], label=algo, color=colors[j], linewidth=1, markevery=0.2)
        ax2.plot(loss[algo], styles[j], label=algo, color=colors[j], linewidth=1, markevery=0.2)
    ax1.set_xlabel('Episodes', size=19)
    ax1.set_ylabel('Reward Avg.', size=19)
    ax2.set_xlabel('Episodes', size=19)
    ax2.set_ylabel('Loss Avg.', size=19)
    ax1.set_xlim(0, 50)
    ax2.set_xlim(0, 50)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    fig1.tight_layout()
    fig1.savefig("ablation-rewards.pdf")
    fig1.show()
    fig2.tight_layout()
    fig2.savefig("ablation-loss.pdf")
    fig2.show()

    for algo, data in {'FCPO-rewards': fcpo_rewards, 'BCE-rewards': bce_rewards, 'FCPO-loss': fcpo_loss,
                       'BCE-loss': bce_loss, 'federated-rewards': fcpof_rewards, 'federated-loss': fcpof_loss}.items():
        if 'federated' in algo:
            fig, ax1 = plt.subplots(1, 1, figsize=(4, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(4, 4), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        i = 0
        for agent, series in data.items():
            ax1.plot(series, label=agent, color=colors[i])
            i += 1
        ax1.legend(fontsize=17)
        ax1.tick_params(axis='x', labelsize=17)
        ax1.tick_params(axis='y', labelsize=17)
        if 'reward' in algo:
            ax1.set_ylabel('Avg. Reward   ', size=20)
        else:
            ax1.set_ylabel('Avg. Loss', size=20)
        if 'federated' in algo:
            ax1.set_xlim(0, 40)
            ax1.set_xticks([0, 15, 30])
            ax1.set_xticklabels([0, 15, '30 Episodes'])
        else:
            ax1.set_xlabel('Episodes', size=20)
            ax1.set_xlim(0, 100)
        plt.tight_layout()
        plt.savefig(f"{algo}.pdf")
        plt.show()


def overall_performance_timeseries(directory, experiment, xticks=None):
    directory = os.path.join(directory, experiment)
    systems = natsorted(os.listdir(directory))
    if 'main' in experiment:
        systems = ['FCPO', 'BCEdge', 'Distream', 'OctopInf']
        fig1, ax1 = plt.subplots(1, 1, figsize=(4.5, 3.4), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3.4), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    else:
        fig1, ax1 = plt.subplots(1, 1, figsize=(4.5, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        fig2, ax2 = plt.subplots(1, 1, figsize=(4.5, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    avg_throughput = {'traffic_throughput': {'total': {}}, 'traffic_goodput': {'total': {}}, 'people_throughput': {'total': {}}, 'people_goodput': {'total': {}}}
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
        if d == 'without local optimization':
            d = 'no local opt.'
        ax1.plot(dfs_combined['aligned_timestamp'], dfs_combined['throughput'], styles[j], label=d,
                 color=colors[j], linewidth=1, markevery=0.2)
        ax2.plot(dfs_combined['aligned_timestamp'], dfs_combined['avg_latency'], styles[j], label=d,
                 color=colors[j], linewidth=1, markevery=0.2)

    if 'main' in experiment:
        experiment = 'total'
        ax1.set_ylabel('Thrghpt (1000 obj / s)', size=16)
        ax2.set_ylabel('Avg Latency (100 ms)', size=16)
    else:
        experiment = 'ablation'
        ax1.set_ylabel('Thrghpt (1000 o/s)     ', size=16)
        ax2.set_ylabel('Latency (100 ms)   ', size=16)


    ax2.set_ylim([0, 850])
    for ax in [ax1, ax2]:
        ax.set_xscale('symlog', linthresh=50)
        ax.set_xticks([0, 10, 20, 50, 100, 200])
        ax.set_xticklabels([0, 10, 20, 50, 100, 200], size=14)
        ax.set_xlim([0, 250])
    ax1.set_xlabel('Minutes Passed since Start (min)', size=16)
    ax2.set_xlabel('Minutes Passed since Start (min)      ', size=16)
    ax1.legend(fontsize=14, loc="lower center")
    ax2.legend(fontsize=14, loc="upper right")
    ax1.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
    ax1.set_yticklabels([0, 1, 2, 3, 4, 5], size=14)
    ax2.set_yticks([0, 200, 400, 600, 800])
    ax2.set_yticklabels([0, 2, 4, 6, 8], size=14)
    fig1.tight_layout()
    fig1.savefig(f"{experiment}-throughput.pdf")
    fig1.show()
    fig2.tight_layout()
    fig2.savefig(f"{experiment}-latency.pdf")
    fig2.show()

    # plot the throughput and latency for each pipeline
    fig, ax1 = plt.subplots(1, 1, figsize=(2.5, 3.4), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    base_plot(avg_throughput, ax1, '', True, systems, False, xticks, 'Effect. Thrghpt (100 obj/s)    ')
    plt.tight_layout()
    plt.savefig(f"{experiment}-throughput-comparison.pdf")
    plt.show()


def perPipeline_performance(directory):
    directory = os.path.join(directory, 'fcpo_results')
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 3), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    labels = ['Traffic', 'Surveillance', 'Indoor']
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
                         linewidth=1, markevery=0.2)

    ax1.set_title('Throughput over 4h', size=12)
    ax1.set_ylabel('Throughput (obj / s)', size=12)
    ax1.set_xlabel('Minutes Passed since Start (min)', size=12)
    ax1.set_xlim([0, 260])
    ax1.legend(fontsize=12, loc='upper center')
    plt.tight_layout()
    plt.savefig('perPipeline-throughput.pdf')
    plt.show()


def warm_start_performance(directory):
    directory = os.path.join(directory, 'warm_start')
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    data = {}
    algorithms = ['fcpo', 'fcpo_cold', 'bce']
    labels = ['FCPO', 'FCPO (Cold Start)', 'BCE']
    dfs = {}
    for j, d in enumerate(algorithms):
        if not os.path.exists(os.path.join(directory, d, 'df.csv')):
            data[d] = []
            c, p, _, _ = read_file(os.path.join(directory, d, os.listdir(os.path.join(directory, d))[0]))
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

            data[d] = align_timestamps([data[d]])
            data[d]['throughput'] = data[d].apply(lambda x: (x['timestamps'] / (x['latency'] / 1000) / 10000000000), axis=1)
            data[d] = bucket_and_average(data[d], ['latency', 'throughput'], num_buckets=180)

            df['throughput'] = (df['throughput'] + data[d]['throughput']) / 2
            df.to_csv(os.path.join(directory, d, 'df.csv'), index=False)
        else:
            df = pd.read_csv(os.path.join(directory, d, 'df.csv'))
        dfs[d] = df
        ax1.plot(df['aligned_timestamp'], df['throughput'], styles[j], label=labels[j], color=colors[j],
                 linewidth=1, markevery=0.2)

    ax1.set_xlabel('Minutes Passed since Start (min)', size=15)
    ax1.set_xlim([0, 6])
    ax1.tick_params(axis='x', labelsize=14)
    ax1.set_ylabel('Throughput (obj / s)     ', size=15)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('warm-start.pdf')
    plt.show()


def limited_network_performance(directory):
    bandwidth_json = json.loads(open(os.path.join(directory, 'bandwidth-limited.json')).read())
    # extract the bandwidth and timestamps from the json
    bandwidth = []
    bandwidth_timestamps = []
    for limit in bandwidth_json['bandwidth_limits']:
        bandwidth.append(limit['mbps'])
        bandwidth_timestamps.append(limit['time'] / 60 - 1.5) # remove system startup time
    directory = os.path.join(directory, 'fcpo_netw')
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    ax2 = ax1.twinx()
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
            df.to_csv(os.path.join(directory, d, 'df.csv'), index=False)
        else:
            df = pd.read_csv(os.path.join(directory, d, 'df.csv'))
        ax1.plot(df['aligned_timestamp'], df['throughput'], styles[j], label=d, color=colors[j], linewidth=1,
                 markevery=0.2)

    ax1.set_xlabel('Minutes Passed since Start (min)', size=12)
    ax1.set_ylabel('Throughput (o/s)', size=12)
    ax1.set_xlim([0, 31.5])
    ax1.legend(fontsize=12)

    ax2.plot(bandwidth_timestamps, bandwidth, label='Bandwidth Limit', color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Bandwidth (Mb/s)', size=12)
    ax2.set_ylim([0, 170])
    plt.tight_layout()
    plt.savefig('network-throughput.pdf')
    plt.show()


def continual_learning_performance(base_directory):
    directory = os.path.join(base_directory, 'fcpo_continual')
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    data = {}
    # add a grey vertical line every 5 minutes
    for i in range(1, 50):
        ax1.axvline(x=i * 5, color='grey', linestyle='-', linewidth=0.5)

    for j, d in enumerate(natsorted(os.listdir(directory))):
        if not os.path.isdir(os.path.join(directory, d)): continue
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
        if d == 'FCPO':
            workload_index, workload = load_data(os.path.join(directory, 'trafficchanges.csv'))
            # 3 data sources in experiment
            ax1.plot(workload_index, [w * 3 for w in workload], label='Workload', color='red', linestyle='--', linewidth=1)
        ax1.plot(df['aligned_timestamp'], df['throughput'], styles[j], label=d, color=colors[j], linewidth=1, markevery=0.2)

    ax1.set_xlabel('Minutes Passed since Start (min)', size=15)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.set_ylabel('Throughput (obj / s)   ', size=15)
    ax1.set_ylim([650, 1500])
    ax1.set_yticks([800, 1000, 1200, 1400])
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_xlim([0, 49])
    ax1.legend(fontsize=15, loc='lower center')
    plt.tight_layout()
    plt.savefig('continual-throughput.pdf')
    plt.show()

def reduced_slo(base_directory):
    base_directory = os.path.join(base_directory, 'reduced_slo')
    fig1, ax1 = plt.subplots(1, 1, figsize=(5.5, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    algorithm_names = ['fcpo', 'bce', 'dis', 'ppp']
    label_map = {'fcpo': 'FCPO', 'bce': 'BCE', 'dis': 'Dis', 'ppp': 'OInf'}
    slos = []
    data = {}
    for i, slo in enumerate(natsorted(os.listdir(base_directory))):
        directory = os.path.join(base_directory, slo)
        if not os.path.exists(os.path.join(directory, 'data.pkl')):
            data[slo] = analyze_single_experiment(directory, natsorted(os.listdir(directory)), 1, int(slo))
            with open(os.path.join(directory, 'data.pkl'), 'wb') as f:
                pickle.dump(data[slo], f)
        else:
            with open(os.path.join(directory, 'data.pkl'), 'rb') as f:
                data[slo] = pickle.load(f)
        slos.append(slo)
    slos = natsorted(slos, reverse=True)
    xs = np.arange(len(slos))
    for j, a in enumerate(algorithm_names):
        ax1.bar(xs + j * 0.2,
                [int(data[s]['traffic_throughput']['total'][a]) + int(data[s]['people_throughput']['total'][a]) for s in slos],
                0.2, alpha=0.5, color=colors[j], hatch='//', edgecolor='white')
        ax1.bar(xs + j * 0.2,
                [int(data[s]['traffic_goodput']['total'][a]) + int(data[s]['people_goodput']['total'][a]) for s in slos],
                0.2, label=label_map[a], color=colors[j], edgecolor='white', linewidth=0.5)
    ax1.axhline(y=data[slos[0]]['max_traffic_throughput'] + data[slos[0]]['max_people_throughput'], color='red', linestyle='--',
               linewidth=2, xmin=0.05, xmax=0.95)
    striped_patch = mpatches.Patch(facecolor='grey', alpha=0.5, hatch='//', edgecolor='white', label='Thrpt')
    solid_patch = mpatches.Patch(facecolor='grey', label='Effect. Thrpt')
    line_patch = Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Workload')
    mpl.rcParams['hatch.linewidth'] = 2
    ax2 = ax1.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=[striped_patch, solid_patch, line_patch], loc='lower left', fontsize=13, frameon=True)

    ax1.set_ylabel('(100 Objects / s)   ', size=14)
    ax1.set_yticks([0, 400, 800, 1200])
    ax1.set_yticklabels([0, 4, 8, 12], size=14)
    ax1.set_xticks(xs + 0.5 * 0.2 * (len(algorithm_names) - 1))
    ax1.set_xticklabels([str(s) + 'ms' for s in slos], size=14)
    ax1.legend(fontsize=13, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.savefig('reduced-slo.pdf')
    plt.show()


def system_overhead(directory):
    # Memory
    systems = ['fcpo', 'bce', 'ppp']
    labels = ['FCPO', 'BCE', 'OInf']
    fig1, ax1 = plt.subplots(1, 1, figsize=(2, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    fig2, ax2 = plt.subplots(1, 1, figsize=(2, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    if not os.path.exists(os.path.join(directory, '..', 'processed_logs', 'memory.pkl')):
        server_memory = avg_memory('nful', systems, 0)
        edge_memory = avg_memory('nful', systems, 1)
        with open(os.path.join(directory, '..', 'processed_logs', 'memory.pkl'), 'wb') as f:
            pickle.dump([server_memory, edge_memory], f)
    else:
        with open(os.path.join(directory, '..', 'processed_logs', 'memory.pkl'), 'rb') as f:
            server_memory, edge_memory = pickle.load(f)
    for i, system in enumerate(systems):
        ax1.bar(i, (server_memory[system][0] + server_memory[system][1]) / 1024, 0.7, label=labels[i], color=colors[i])
        ax2.bar(i, edge_memory[system][0] / (1024 * 1024), 0.7, label=labels[i], color=colors[i])
        ax1.plot([i - 0.35, i + 0.35], [(server_memory['ppp'][0] + server_memory['ppp'][1]) / 1024,
                                         (server_memory['ppp'][0] + server_memory['ppp'][1]) / 1024], color='black')
        ax2.plot([i - 0.35, i + 0.35], [edge_memory['ppp'][0] / (1024 * 1024), edge_memory['ppp'][0] / (1024 * 1024)], color='black')
    ax1.legend(fontsize=14, loc='lower center')
    ax1.set_ylabel('Mem Usage (GB)', size=15)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_xticks([])
    ax2.legend(fontsize=14, loc='lower center')
    ax2.set_ylabel('Mem Usage (GB)', size=15)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.set_xticks([])
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig('server-memory.pdf')
    fig2.savefig('edge-memory.pdf')
    fig1.show()
    fig2.show()

    # Power consumption
    fig1, ax1 = plt.subplots(1, 1, figsize=(2, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    if not os.path.exists(os.path.join(directory, '..', 'processed_logs', 'power.pkl')):
        power = avg_edge_power('nful', systems)
        with open(os.path.join(directory, '..', 'processed_logs', 'power.pkl'), 'wb') as f:
            pickle.dump(power, f)
    else:
        with open(os.path.join(directory, '..', 'processed_logs', 'power.pkl'), 'rb') as f:
            power = pickle.load(f)
    for i, system in enumerate(systems):
        if system == 'ppp':
            continue
        ax1.bar(i, (power[system][0] - power['ppp'][0]), 0.7, label=labels[i], color=colors[i])
    ax1.legend(fontsize=13, loc='upper right', bbox_to_anchor=(1, 0.8))
    ax1.set_ylabel('Avg Additional Power\nUsage (100 mW)', size=15)
    ax1.set_yticks([0, 100, 200])
    ax1.set_yticklabels([0, 1, 2])
    ax1.set_xticks([])
    fig1.tight_layout()
    fig1.savefig('power-consumption.pdf')
    fig1.show()

    # RL Latency
    fig1, ax1 = plt.subplots(1, 1, figsize=(3.5, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    fig2, ax2 = plt.subplots(1, 1, figsize=(3.5, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
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

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend([handles[0], handles[5]], [labels[0], labels[5]], fontsize=13, loc='upper right')
    ax1.set_xticks(x + 0.2)
    ax1.set_xticklabels(x_labels, size=15, rotation=25)
    ax1.set_ylabel('Decision Time (ms)       ', size=15)
    ax1.tick_params(axis='y', labelsize=14)
    ax2.legend([handles[0], handles[5]], [labels[0], labels[5]], fontsize=13, loc='upper left')
    ax2.set_xticks(x + 0.2)
    ax2.set_xticklabels(x_labels, size=15, rotation=25)
    ax2.set_ylabel('Train Time (100ms)      ', size=15)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.set_yticks([0, 500, 1000])
    ax2.set_yticklabels([0, 5, 10])
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig('inference-latency.pdf')
    fig2.savefig('update-latency.pdf')
    fig1.show()
    fig2.show()

    #FL Latency
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, sharey=True, figsize=(2, 2.3), gridspec_kw={'height_ratios': [1], 'width_ratios': [2,3]})
    FineTuning_latencies = []
    FL_latencies = []
    with open(os.path.join(directory, 'latest_log_21_20-59-1.csv'), 'r') as f:
        for line in f:
            if 'federatedAggregation' in line:
                FineTuning_latencies.append(int(line.split(',')[1]) / 1000000) # convert to seconds
                FL_latencies.append(int(line.split(',')[2]) / 1000000) # convert to seconds
    ax1a.text(0.5, 0.5, 'Fine-Tune  ', ha='center', va='bottom', transform=ax1a.transAxes, fontsize=15)
    ax1a.boxplot(FineTuning_latencies, positions=[0], vert=False, widths=0.8, patch_artist=True, boxprops=dict(facecolor=colors[0]),
                medianprops=dict(color='black'), showfliers=True)
    ax1b.text(0.5, 0.35, 'FL Round-Trip', ha='center', va='bottom', transform=ax1b.transAxes, fontsize=15)
    ax1b.boxplot(FL_latencies, positions=[1], vert=False, widths=0.8, patch_artist=True, boxprops=dict(facecolor=colors[0]),
                medianprops=dict(color='black'), showfliers=True)

    ax1a.set_yticks([])
    ax1a.set_xlim(0, 0.25)
    ax1b.set_xlim(3.5, 12.5)
    ax1a.set_xticks([0, 0.1, 0.2])
    ax1b.set_xticks([4, 6, 9, 12])
    ax1a.set_xticklabels([0, '.1', '.2'], size=15)
    ax1b.set_xticklabels([4, 6, 9, '12s'], size=15)

    ax1a.spines['right'].set_visible(False)
    ax1b.spines['left'].set_visible(False)
    d = .015  # size of diagonal lines
    kwargs = dict(transform=ax1a.transAxes, color='k', clip_on=False)
    ax1a.plot([1 - d, 1 + d], [-d, +d], **kwargs)
    ax1a.plot([1 - d, 1 + d], [1 - d, 1 + d], **kwargs)

    kwargs.update(transform=ax1b.transAxes)
    ax1b.plot([-d, +d], [-d, +d], **kwargs)
    ax1b.plot([-d, +d], [1 - d, 1 + d], **kwargs)
    fig1.subplots_adjust(wspace=0.1)
    fig1.savefig('FL-latency.pdf')
    fig1.show()


def hyperparameter_sensitivity(base_directory):
    base_directory = os.path.join(base_directory, 'hyperparameters')
    for i, exp in enumerate(natsorted(os.listdir(base_directory))):
        directory = os.path.join(base_directory, exp)
        params = os.listdir(directory)
        if not os.path.exists(os.path.join(base_directory, '..', 'processed_logs', f"{exp}.pkl")):
            x_vals = sorted(set(float(p.split('-')[0].replace(',', '.')) for p in params))
            y_vals = sorted(set(float(p.split('-')[1].replace(',', '.')) for p in params))
            X, Y = np.meshgrid(x_vals, y_vals)
            data = analyze_single_experiment(directory, natsorted(params), 1, 250)
            z = []
            for x, y in zip(X.flatten(), Y.flatten()):
                param_str = f"{x:.1f}-{y:.1f}".replace('.', ',')
                try:
                    z.append(data['traffic_throughput']['total'][param_str] + data['people_throughput']['total'][param_str])
                except KeyError:
                    z.append(0)
            Z = np.array(z).reshape(X.shape)
            x_flat = X.flatten()
            y_flat = Y.flatten()
            z_flat = Z.flatten()
            with open(os.path.join(base_directory, '..', 'processed_logs', f"{exp}.pkl"), 'wb') as f:
                pickle.dump([X, x_flat, Y, y_flat, Z, z_flat], f)
        else:
            with open(os.path.join(base_directory, '..', 'processed_logs', f"{exp}.pkl"), 'rb') as f:
                X, x_flat, Y, y_flat, Z, z_flat = pickle.load(f)

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection='3d')
        colors = cm.viridis_r((z_flat - z_flat.min()) / (z_flat.max() - z_flat.min()))
        sc = ax.scatter(x_flat, y_flat, z_flat, c=colors, s=40)

        # Connect neighbor points (grid lines)
        rows, cols = X.shape
        for i in range(rows):
            for j in range(cols):
                if j < cols - 1:
                    ax.plot([X[i, j], X[i, j + 1]], [Y[i, j], Y[i, j + 1]], [Z[i, j], Z[i, j + 1]],
                        color='gray', linewidth=0.5)
                if i < rows - 1:
                    ax.plot([X[i, j], X[i + 1, j]], [Y[i, j], Y[i + 1, j]], [Z[i, j], Z[i + 1, j]],
                        color='gray', linewidth=0.5)

        ax.set_xlabel(exp.split('_')[0])
        ax.set_ylabel(exp.split('_')[1])
        cbar = plt.colorbar(sc, ax=ax, pad=0, shrink=0.5)
        cbar.ax.invert_yaxis()
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(['high', 'avg', 'low'])
        cbar.set_label('Throughput')
        ax.set_zticks([])
        ax.set_yticks(y_flat)
        ax.set_xticks(x_flat)
        plt.tight_layout()
        plt.savefig(f"{exp}.pdf")
        plt.show()

def perDeviceAnalysis(base_directory):
    fig, ax = plt.subplots(4, 3, figsize=(7.5, 5))

    axs = [ax[0, 0], ax[0,1], ax[0,2], ax[1,0], ax[1,1], ax[1,2], ax[2,0], ax[2,1], ax[2,2], ax[3,0], ax[3,1], ax[3,2]]
    videos = ['traffic1.mp4', 'traffic2.mp4', 'traffic3.mp4', 'traffic4.mp4', 'traffic5.mp4', 'traffic6.mp4', 'traffic7.mp4',
              'traffic100.mp4', 'traffic20.mp4', 'traffic30.mp4', 'traffic40.mp4', 'traffic50.mp4', 'traffic60.mp4', 'traffic70.mp4',
              'people1.mp4', 'people2.mp4', 'people3.mp4', 'people4.mp4', 'indoor1.mp4', 'indoor2.mp4']
    bandwidth_files = ['bandwidth_limits1.json', 'bandwidth_limits2.json', 'bandwidth_limits3.json',
                       'bandwidth_limits4.json', 'bandwidth_limits5.json', 'bandwidth_limits6.json',
                       'bandwidth_limits7.json', 'bandwidth_limits8.json', 'bandwidth_limits9.json',
                       'bandwidth_limits10.json', 'bandwidth_limits11.json', 'bandwidth_limits12.json']
    bandwidths = get_bandwidths(base_directory)
    bandwidths_timestamps = {}
    for b in bandwidths:
        bandwidths_timestamps[b] = bandwidths[b][1]
        bandwidths[b] = bandwidths[b][0]
    for i in range(12):
        detailed_data = []
        individual_figures(axs[i], os.path.join(base_directory, 'workload'), detailed_data, videos[i], 0.5,
                       bandwidths_timestamps[bandwidth_files], bandwidths[bandwidth_files], 0, 0)

    plt.subplots_adjust(wspace=0.004, hspace=0)
    plt.tight_layout(pad=0.04)
    plt.savefig('individual-patterns.pdf')
    plt.show()
