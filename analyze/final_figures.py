import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted
from pandas import DataFrame

from run_log_analyzes import analyze_single_experiment, read_file
from database_conn import avg_memory, full_memory, align_timestamps, bucket_and_average, arrival_query

bar_width = 0.2
x_labels = ['traffic', 'surveillance']
x = np.arange(len(x_labels))
algorithm_names = ['OURS', 'dis', 'jlf', 'rim']
colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#F0E442']
label_map = {'ppp': 'OctopInf', 'OURS': 'OctopInf', 'dis': 'Distream', 'jlf': 'Jellyfish', 'rim': 'Rim'}


def base_plot(data, ax, title):
    for j, a in enumerate(algorithm_names):
        ax.bar(x + j * bar_width, [data['traffic_throughput']['total'][a], data['people_throughput']['total'][a]], bar_width, alpha=0.5, color=colors[j], hatch='//', edgecolor='white')
        traffic_intime = data['traffic_goodput']['total'][a]
        people_intime = data['people_goodput']['total'][a]
        ax.bar(x + j * bar_width, [traffic_intime, people_intime], bar_width, label=label_map[a], color=colors[j], edgecolor='white', linewidth=0.5)

        ax.text(x[0] + j * bar_width, traffic_intime, f'{traffic_intime:.0f}', ha='center', va='bottom', size=12)
        ax.text(x[1] + j * bar_width, people_intime, f'{people_intime:.0f}', ha='center', va='bottom', size=12)

    ax.axhline(y=data['max_traffic_throughput'], color='red', linestyle='--', linewidth=2, xmin=0.05, xmax=0.45)
    ax.axhline(y=data['max_people_throughput'], color='red', linestyle='--', linewidth=2, xmin=0.55, xmax=0.95)

    ax.set_title(title, size=12)
    ax.set_ylabel('Throughput (objects / s)', size=12)
    if data['max_traffic_throughput'] < 1500:
        yticks = np.arange(0, int(data['max_traffic_throughput']), 250).tolist()
    else:
        yticks = np.arange(0, int(data['max_traffic_throughput']), 500).tolist()
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, size=12)
    ax.set_xticks(x + 0.5 * bar_width * (len(algorithm_names) - 1))
    ax.set_xticklabels(x_labels, size=12)
    ax.legend(fontsize=12)


def memory_plot(experiment, ax):
    memory = avg_memory(experiment)
    for j, a in enumerate(algorithm_names):
        if a == 'OURS':
            a = 'ppp'
        ax.bar(j, memory[a][0] / 1024, bar_width + 0.5, color=colors[j])
    ax.set_title('b) Avg GPU Memory', size=12)
    ax.set_ylabel('Mem Usage (GB)', size=12)
    ax.set_xticks([])
    ax.set_yticks([0, 20, 40, 60, 80])


def create_figures(args, dirs):
    dirs = [d for d in dirs if "old" not in d]
    figs = args.figs.split(',')
    total = {}
    if 'full' in figs or 'thir' in figs or 'slo' in figs or 'abla' in figs:
        for d in dirs:
            if d != 'full' and (d == 'bndw' or d == 'long' or len([f for f in figs if f in d]) == 0):
                continue
            filepath = os.path.join(args.directory, d)
            if 'lslo' in d:
                total[d] = analyze_single_experiment(filepath, natsorted(os.listdir(filepath)), args.num_results, 150)
            elif 'mslo' in d:
                total[d] = analyze_single_experiment(filepath, natsorted(os.listdir(filepath)), args.num_results, 100)
            else:
                total[d] = analyze_single_experiment(filepath, natsorted(os.listdir(filepath)), args.num_results)

    ########################################################################################################################
    ##### Full Experiment Plot #############################################################################################
    ########################################################################################################################

    if 'full' in figs:
        fig, axs = plt.subplots(2, 2, figsize=(8, 5), gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [1, 3]})
        axs[0, 1].remove()
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)  # Create a new subplot spanning the first row
        ax2 = axs[1, 0]
        ax3 = axs[1, 1]
        base_plot(total['full'], ax1, 'a) Throughput Comparisons in General Settings')
        memory_plot('full', ax2)
        for j, a in enumerate(algorithm_names):
            ax3.boxplot([total['full']['traffic_latency']['total'][a]], positions=[j], vert=False, widths=0.9,
                        patch_artist=True, boxprops=dict(facecolor=colors[j]), medianprops=dict(color='black'), showfliers=False)
            max_value = max(total['full']['traffic_latency']['total'][a])
            # ax3.text(560, j+0.1, f'{max_value:.0f}', ha='center', va='bottom', size=12, color=colors[j])
            # ax3.plot(560, j, 'o', color=colors[j])
        ax3.set_xlabel('End-to-End Latency (ms)', size=12)
        ax3.set_xlim([0, 600])
        # ax3.set_xticks([0, 150, 300, 450, 560])
        ax3.set_xticks([0, 150, 300, 450, 600])
        # ax3.set_xticklabels([0, 150, 300, 450, 'max'], size=12)
        ax3.set_xticklabels([0, 150, 300, 450, 600], size=12)
        ax3.set_yticks([])
        ax3.set_title('c) End-to-end Latency Distribution of Traffic Pipeline', size=12)
        plt.subplots_adjust(wspace=0.1)
        plt.tight_layout()
        plt.show()

    ########################################################################################################################
    ##### 30FPS Experiment Plot ############################################################################################
    ########################################################################################################################

    if 'thir' in figs:
        fig, axs = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'height_ratios': [1], 'width_ratios': [3, 1]})
        ax1 = axs[0]
        ax2 = axs[1]
        base_plot(total['thir'], ax1, 'a) Throughput Comparisons under Higher Workloads')
        memory_plot('thir', ax2)
        plt.tight_layout()
        plt.show()

    ########################################################################################################################
    ##### Limited Bandwidth Experiment Plot ################################################################################
    ########################################################################################################################

    if 'bndw' in figs:
        fig, axs = plt.subplots(1, 1, figsize=(8, 3))
        ax1 = axs
        ax2 = ax1.twinx()
        data = {}
        keys = []
        for d in natsorted(os.listdir(os.path.join(args.directory, 'bndw'))):
            data[d] = []
            keys.append(d)
            for f in natsorted(os.listdir(os.path.join(args.directory, 'bndw', d))):
                c, p, _, _ = read_file(os.path.join(args.directory, 'bndw', d, f))
                data[d].extend(c)
                data[d].extend(p)
        for d in keys:
            df = pd.DataFrame(data[d], columns=['path', 'latency', 'timestamps'])
            df['latency'] = df['latency'].apply(lambda x: int(x))
            df['timestamps'] = df['timestamps'].apply(lambda x: int(x))
            data[d] = df
        for j, d in enumerate(keys):
            df = DataFrame()

            data[d]['aligned_timestamp'] = (data[d]['timestamps'] // 1e6).astype(int) * 1e6
            df['throughput'] = data[d].groupby('aligned_timestamp')['latency'].transform('size')
            df['aligned_timestamp'] = (data[d]['aligned_timestamp'] - data[d]['aligned_timestamp'].iloc[0]) / (60 * 1e6)
            df = df.groupby('aligned_timestamp').agg({'throughput': 'mean'}).reset_index()
            df = bucket_and_average(df, ['throughput'], num_buckets=180)

            data[d] = align_timestamps([data[d]])
            data[d]['throughput'] = data[d].apply(lambda x: (x['timestamps'] / (x['latency'] / 1000) / 10000000000), axis=1)
            data[d] = bucket_and_average(data[d], ['latency', 'throughput'], num_buckets=180)

            df['throughput'] = (df['throughput'] + data[d]['throughput']) / 2
            ax1.plot(df['aligned_timestamp'], df['throughput'], label=label_map[d], color=colors[j], linewidth=3)

        ax1.set_title('Effective Throughput Comparisons of Both Pipelines under Fluctuating Bandwidths', size=12)
        ax1.set_xlabel('Minutes Passed since Start (min)', size=12)
        ax1.set_ylabel('Throughput (objects / s)', size=12)
        ax1.set_xlim([0, 31.5])
        ax1.legend(loc='lower right', fontsize=12)

        bandwidth = [50, 50, 25, 25, 10, 10, 35, 35, 20, 20, 30, 30]
        bandwidth_timestamps = [0, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 32]
        ax2.plot(bandwidth_timestamps, bandwidth, label='Bandwidth Limit', color='black', linestyle='--', linewidth=3)
        ax2.set_yticks([0, 10, 20, 30, 40, 50])
        ax2.set_yticklabels([0, 10, 20, 30, 40, 50], size=12)
        ax2.set_ylabel('Bandwidth Limit (Mb/s)', size=12)
        plt.tight_layout()
        plt.show()

    ########################################################################################################################
    ##### Ablation Experiment Plot #########################################################################################
    ########################################################################################################################

    if 'abla' in figs:
        fig, axs = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'height_ratios': [1], 'width_ratios': [3, 2]})
        ax1 = axs[0]
        ax2 = axs[1]
        versions = ['OURS', 'dis', 'jlf', 'ppsb', 'ppos', 'ppwt']
        label_names = {'OURS': 'OctopInf', 'jlf': 'Jellyfish', 'dis': 'Distream', 'ppsb': 'Static Batching', 'ppos': 'Server Only', 'ppwt': 'Server Only w/o Coral'}
        for j, a in enumerate(versions):
            val = int(total['abla']['traffic_goodput']['total'][a]) + int(total['abla']['people_goodput']['total'][a])
            ax1.bar(j, [val], label=label_names[a], color=colors[j])
            ax1.text(j, val, f'{val:.0f}', ha='center', va='bottom', size=12)
        ax1.set_title('a) Effective Thrpt of Both Pipelines in Different Configurations', size=12)
        ax1.set_ylabel('Throughput (objects / s)', size=12)
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        ax1.legend(fontsize=12, loc='lower left')
        dec = 0
        for j, a in enumerate(versions):
            if a == 'jlf' or a == 'dis':
                dec += 1
                continue
            ax2.boxplot([total['abla']['traffic_latency']['total'][a]], positions=[j-dec], vert=True, widths=0.9,
                        patch_artist=True, boxprops=dict(facecolor=colors[j]), medianprops=dict(color='black'), showfliers=False)
        ax2.set_ylabel(' (ms)', size=12)
        ax2.set_ylim([0, 180])
        ax2.set_yticks([0, 50, 100, 150])
        ax2.set_yticklabels([0, 50, 100, 150], size=12)
        ax2.set_xticks([])
        ax2.set_title('b) End-to-End Latency Dist', size=12)
        plt.tight_layout()
        plt.show()

    ########################################################################################################################
    ##### Reduced SLO Experiment Plot ######################################################################################
    ########################################################################################################################

    if 'slo' in figs:
        fig, axs = plt.subplots(1, 1, figsize=(8, 3), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        ax1 = axs
        slos = ['full', 'lslo', 'mslo']
        slo_labels = {'full': 'Full SLO', 'lslo': 'SLO reduced by 50ms', 'mslo': 'SLO reduced by 100ms'}
        xs = np.arange(len(slos))
        for j, a in enumerate(algorithm_names):
            val = {}
            for s in slos:
                val[s] = int(total[s]['traffic_throughput']['total'][a]) + int(total[s]['people_throughput']['total'][a])
            ax1.bar(xs + j * bar_width, [val['full'], val['lslo'], val['mslo']], bar_width, alpha=0.5, color=colors[j], hatch='//', edgecolor='white')
            for s in slos:
                val[s] = int(total[s]['traffic_goodput']['total'][a]) + int(total[s]['people_goodput']['total'][a])
            ax1.bar(xs + j * bar_width, [val['full'], val['lslo'], val['mslo']], bar_width, label=label_map[a], color=colors[j], edgecolor='white', linewidth=0.5)
            ax1.text(xs[0] + j * bar_width, val['full'], f'{val["full"]:.0f}', ha='center', va='bottom', size=12)
            ax1.text(xs[1] + j * bar_width, val['lslo'], f'{val["lslo"]:.0f}', ha='center', va='bottom', size=12)
            ax1.text(xs[2] + j * bar_width, val['mslo'], f'{val["mslo"]:.0f}', ha='center', va='bottom', size=12)
        ax1.set_title('Throughput Comparisons of Both Pipelines under Stricter SLOs', size=12)
        ax1.set_ylabel('Throughput (objects / s)', size=12)
        ax1.set_xticks(xs + 0.5 * bar_width * (len(algorithm_names) - 1))
        ax1.set_xticklabels([slo_labels[s] for s in slos], size=12)
        ax1.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    ########################################################################################################################
    ##### Long Runtime Experiment Plot #####################################################################################
    ########################################################################################################################

    if 'long' in figs:
        fig, axs = plt.subplots(1, 1, figsize=(8, 3), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        ax1 = axs
        # ax2 = ax1.twinx()
        cars = []
        people = []
        # if the csvs do not exist, create them
        if not os.path.exists(os.path.join(args.directory, 'long', 'df_cars.csv')) or not os.path.exists(os.path.join(args.directory, 'long', 'df_people.csv')):
            for f in natsorted(os.listdir(os.path.join(args.directory, 'long', 'OURS'))):
                c, p, _, _ = read_file(os.path.join(args.directory, 'long', 'OURS', f))
                cars.extend(c)
                people.extend(p)
            df_cars = pd.DataFrame(cars, columns=['path', 'latency', 'timestamps'])
            df_cars.to_csv(os.path.join(args.directory, 'long', 'df_cars.csv'), index=False)
            df_people = pd.DataFrame(people, columns=['path', 'latency', 'timestamps'])
            df_people.to_csv(os.path.join(args.directory, 'long', 'df_people.csv'), index=False)
        else:
            df_cars = pd.read_csv(os.path.join(args.directory, 'long', 'df_cars.csv'))
            df_people = pd.read_csv(os.path.join(args.directory, 'long', 'df_people.csv'))
        lables = ['Traffic Throughput', 'Surveillance Throughput']
        for j, df in enumerate([df_cars, df_people]):
            df['timestamps'] = df['timestamps'].apply(lambda x: int(x))
            df['latency'] = df['latency'].apply(lambda x: int(x))
            df['aligned_timestamp'] = (df['timestamps'] // 1e6).astype(int) * 1e6
            df['throughput'] = df.groupby('aligned_timestamp')['latency'].transform('size')
            df['aligned_timestamp'] = (df['aligned_timestamp'] - df['aligned_timestamp'].iloc[0]) / (60 * 1e6)
            df = df.groupby('aligned_timestamp').agg({'throughput': 'mean'}).reset_index()
            df = bucket_and_average(df, ['throughput'], num_buckets=780)
            ax1.plot(df['aligned_timestamp'], df['throughput'], label=lables[j], color=colors[j], linewidth=3)
        # ax1.plot(0, 0, label='Memory Usage', color='black', linestyle='--', linewidth=2)
        if not os.path.exists(os.path.join(args.directory, 'long', 'missed.csv')):
            missed = bucket_and_average(arrival_query('long_ppp'), ['queue_drops', 'late_requests'], num_buckets=780)
            missed.to_csv(os.path.join(args.directory, 'long', 'missed.csv'), index=False)
        else:
            missed = pd.read_csv(os.path.join(args.directory, 'long', 'missed.csv'))
        ax1.plot(missed['aligned_timestamp'], missed['queue_drops'], label='Dropped in Queues', color=colors[2], linewidth=2)
        ax1.plot(missed['aligned_timestamp'], missed['late_requests'], label='Late Requests', color=colors[5], linewidth=2)
        ax1.set_title('Throughput and Missed Requests over 13h', size=12)
        ax1.set_xlabel('Minutes Passed since Start (min)', size=12)
        ax1.set_xlim([0, 780])
        ax1.set_ylabel('Throughput / Missed (objects / s)', size=12)
        ax1.legend(loc='lower left', fontsize=12)
        # if not os.path.exists(os.path.join(args.directory, 'long', 'memory.csv')):
        #     memory = full_memory('long_ppp', 780)
        #     memory.to_csv(os.path.join(args.directory, 'long', 'memory.csv'), index=False)
        # else:
        #     memory = pd.read_csv(os.path.join(args.directory, 'long', 'memory.csv'))
        # ax2.plot(memory['bucket'], memory['total_gpu_mem'] / 1024, label='Memory Usage', color='black', linestyle='--', linewidth=3)
        # ax2.set_ylabel('Memory Usage (GB)', size=12)
        plt.tight_layout()
        plt.savefig('long_runtime.svg')

