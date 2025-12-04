import os
import json
import pickle
import itertools

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter

from natsort import natsorted
from pandas import DataFrame

from run_log_analyzes import read_file, get_total_objects, analyze_single_experiment, get_bandwidths
from database_connection import bucket_and_average, align_timestamps, avg_memory, full_edge_power
from analyze_object_counts import load_data

colors = ['#365d8d', '#e6ab02', '#471164', '#5dc863', '#21908c']
long_colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#F0E442', '#D55E00']
bar_width = 0.2
x_labels = ['traffic', 'surveillance']
x = np.arange(len(x_labels))
algorithm_names = ['OURS', 'dis', 'jlf', 'rim']
styles = ['-', '--', 'o-', 'x-']
markers = ['', 'x', 'o', '*', '//', '\\\\']  # plain, triangle, circle, star, stripes left and right
label_map = {'ppp': 'OctopInf', 'OURS': 'OctopInf', 'dis': 'Distream', 'jlf': 'Jellyfish', 'rim': 'Rim'}


def base_plot(data, ax, title, sum_throughput=False, labels=algorithm_names, use_label_map=True, xticks=None, y_label='Objects / s'):
    # error handling for different data formats
    if xticks is None:
        xticks = ['OInf', 'Dis', 'Jlf', 'Rim']
    for x in ['traffic_throughput', 'people_throughput', 'traffic_goodput', 'people_goodput']:
        if x not in data:
            data[x] = {'total': {a: 0 for a in labels}}
    for x in ['max_traffic_throughput', 'max_people_throughput']:
        if x not in data:
            data[x] = 0

    for j, a in enumerate(labels):
        if sum_throughput:
            ax.bar(j * bar_width,
                   [data['traffic_throughput']['total'][a] + data['people_throughput']['total'][a]],
                   bar_width, alpha=0.5, color=colors[j], hatch='//', edgecolor='white')
        else:
            ax.bar(x + j * bar_width,
                   [data['traffic_throughput']['total'][a], data['people_throughput']['total'][a]],
                   bar_width, alpha=0.5, color=colors[j], hatch='//', edgecolor='white')
        traffic_intime = data['traffic_goodput']['total'][a]
        people_intime = data['people_goodput']['total'][a]
        if use_label_map:
            if sum_throughput:
                ax.bar(j * bar_width, [traffic_intime + people_intime], bar_width, label=label_map[a], color=colors[j],
                       edgecolor='white', linewidth=0.5)
                ax.text(j * bar_width, traffic_intime + people_intime, f'{traffic_intime + people_intime:.0f}', ha='center',
                        va='bottom', size=12)
            else:
                ax.bar(x + j * bar_width, [traffic_intime, people_intime], bar_width, label=label_map[a], color=colors[j],
                       hatch=markers[j], edgecolor='white', linewidth=0.5)
                ax.text(x[0] + j * bar_width, traffic_intime, f'{traffic_intime:.0f}', ha='center', va='bottom', size=10)
                ax.text(x[1] + j * bar_width, people_intime, f'{people_intime:.0f}', ha='center', va='bottom', size=10)
        else:
            if sum_throughput:
                ax.bar(j * bar_width, [traffic_intime + people_intime], bar_width, label=a, color=colors[j],
                       edgecolor='white', linewidth=0.5)
                ax.text(j * bar_width, traffic_intime + people_intime, f'{traffic_intime + people_intime:.0f}', ha='center',
                        va='bottom', size=12)
            else:
                ax.bar(x + j * bar_width, [traffic_intime, people_intime], bar_width, label=a, color=colors[j],
                       hatch=markers[j], edgecolor='white', linewidth=0.5)
                ax.text(x[0] + j * bar_width, traffic_intime, f'{traffic_intime:.0f}', ha='center', va='bottom', size=10)
                ax.text(x[1] + j * bar_width, people_intime, f'{people_intime:.0f}', ha='center', va='bottom', size=10)

        if a == 'OURS' or a == 'FCPO':
            striped_patch = mpatches.Patch(facecolor='grey', alpha=0.5, hatch='//', edgecolor='white', label='Thrpt')
            solid_patch = mpatches.Patch(facecolor='grey', label='Effect. Thrpt')
            line_patch = Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Workload')
            mpl.rcParams['hatch.linewidth'] = 2

            if sum_throughput:
                ax.legend(handles=[striped_patch, solid_patch, line_patch], loc='lower left', fontsize=12, frameon=True, bbox_to_anchor = (0, -0.06))
            else:
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.legend(handles=[striped_patch, solid_patch, line_patch], loc='upper center', fontsize=10, frameon=True)

    if sum_throughput:
        ax.axhline(y=data['max_traffic_throughput'] + data['max_people_throughput'], color='red', linestyle='--',
                   linewidth=2, xmin=0.05, xmax=0.95)
        ax.set_xticks([i * 0.2 for i in range(len(xticks))])
        ax.set_xticklabels(xticks, size=12, rotation=15)
        yticks = np.arange(0, int(data['max_traffic_throughput'] + data['max_people_throughput']), 500).tolist()
        ax.set_yticks(yticks)
        if use_label_map:
            ax.set_yticklabels([int(y / 100) for y in yticks], size=10)
        else:
            ax.set_yticklabels([int(y / 100) for y in yticks], size=11)
    else:
        ax.axhline(y=data['max_traffic_throughput'], color='red', linestyle='--', linewidth=2, xmin=0.05, xmax=0.45)
        ax.axhline(y=data['max_people_throughput'], color='red', linestyle='--', linewidth=2, xmin=0.55, xmax=0.95)
        ax.set_xticks(x + 0.5 * bar_width * (len(algorithm_names) - 1))
        ax.set_xticklabels(x_labels, size=10)
        if data['max_traffic_throughput'] < 1500:
            yticks = np.arange(0, int(data['max_traffic_throughput']), 250).tolist()
        else:
            yticks = np.arange(0, int(data['max_traffic_throughput']), 500).tolist()
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, size=10)

    ax.set_title(title, size=12)
    ax.set_ylabel(y_label, size=12)

    if not sum_throughput:
        ax.legend(fontsize=12)


def memory_plot(experiment, ax, first_plot):
    memory = avg_memory(experiment)
    for j, a in enumerate(algorithm_names):
        if a == 'OURS':
            a = 'ppp'
        ax.bar(j, memory[a][0] / 1024, bar_width + 0.5, label=label_map[a], color=colors[j], edgecolor='white', linewidth=0.5)
    if first_plot:
        ax.set_title('c) Avg GPU Mem (GB)', size=12)

    else:
        ax.set_title('b) Avg GPU Mem (GB)', size=12)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['OInf', 'Dis', 'Jlf', 'Rim'], size=10)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels([0, 20, 40, 60, 80], size=10)


def prepare_timeseries(df, with_latency=True):
    df = df[df['latency'] < 200000]
    output_df = DataFrame()
    output_df['throughput'] = df.groupby('aligned_timestamp')['latency'].transform('size')
    df = df.sort_values(by='aligned_timestamp')
    output_df['aligned_timestamp'] = (df['aligned_timestamp'] - df['aligned_timestamp'].iloc[0]) / (60 * 1e6)
    output_df = output_df.groupby('aligned_timestamp').agg({'throughput': 'mean'}).reset_index()
    output_df = bucket_and_average(output_df, ['throughput'], num_buckets=180)

    if with_latency:
        df = align_timestamps([df])
        df['throughput'] = df.apply(lambda x: (x['timestamps'] / (x['latency'] / 1000) / 10000000000), axis=1)
        df = bucket_and_average(df, ['latency', 'throughput'], num_buckets=180)
        output_df['throughput'] = (output_df['throughput'] + df['throughput']) / 2

    # when the last element in the dataframe has aligned timestamp smaller than 30
    if output_df['aligned_timestamp'].iloc[-1] < 30:
        # calculate difference between last element and 30, then add that many elements to the beginning of the Dataframe with throughput 0
        diff = 30 - output_df['aligned_timestamp'].iloc[-1]
        missed_start = {'aligned_timestamp': [i for i in range(0, int(diff))], 'latency':
            [0 for _ in range(0, int(diff))], 'throughput': [0 for _ in range(0, int(diff))]}
        output_df['aligned_timestamp'] = output_df['aligned_timestamp'] + diff
        output_df = pd.concat([DataFrame(missed_start), output_df], ignore_index=True)
    return output_df


def closest_index(array, value):
    pos = np.searchsorted(array, value)
    if pos == 0:
        return 0
    if pos == len(array):
        return len(array) - 1
    before = pos - 1
    after = pos
    if abs(array[after] - value) < abs(array[before] - value):
        return after
    else:
        return before


def fcpoMainFigure(base_directory):
    cases = ['centralized', 'distributed']
    case_labels = ['i-Centralized', 'ii-Distributed']
    scenarios = ['camp', 'fcty']
    scenario_labels = ['Urban Surveillance', 'Smart Factory']
    latencies = ['100', '200', '250']
    systems = ['fcpo', 'base', 'rule', 'bce']
    system_labels = ['FCPO', 'Base', 'Rule', 'BCEdge']

    # create 2 barplots, one for each case
    total = {}
    for case in cases:
        fig, axs = plt.subplots(1, 1, figsize=(6.5, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        directory = os.path.join(base_directory, case)

        data = {}
        for lat in latencies:
            data[lat] = {}
        for exp in scenarios:
            if not os.path.exists(os.path.join(directory, exp, f'processed_logs.pkl')):
                for lat in latencies:
                    dir = os.path.join(directory, exp, lat)
                    if exp == 'fcty':
                        data[lat][exp] = analyze_single_experiment(dir, systems, 1, int(lat), True, 2160)
                    elif exp == 'camp':
                        data[lat][exp] = analyze_single_experiment(dir, systems, 1, int(lat), True, 7200)
                with open(os.path.join(directory, exp, f'processed_logs.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(os.path.join(directory, exp, f'processed_logs.pkl'), 'rb') as f:
                    data = pickle.load(f)

        total[case] = data
        xticks = []
        xtick_labels = []
        for k, exp in enumerate(scenarios):
            for j, lat in enumerate(latencies):
                for i, sys in enumerate(systems):
                    throughput_rate = data[lat][exp]['throughput'][sys] / data[lat][exp]['max_throughput'] * 100
                    goodput_rate = data[lat][exp]['goodput'][sys] / data[lat][exp]['max_throughput'] * 100

                    bar_id = i + len(systems) * (j + k * len(latencies))
                    group_id = bar_id // len(systems)
                    in_group_index = bar_id % len(systems)
                    pattern_width = len(systems) * bar_width
                    cumulative_gap = sum(0.4 if (g % len(latencies)) == 2 else 0.1 for g in range(group_id))
                    x_base = group_id * pattern_width + cumulative_gap + in_group_index * bar_width

                    if group_id % 3 == 1 and in_group_index == 0:
                        xticks.append(x_base + (len(systems) * bar_width) / 2 - bar_width / 2)
                        xtick_labels.append(f'{lat}ms\n{scenario_labels[k]}')
                    elif in_group_index == 0:
                        xticks.append(x_base + (len(systems) * bar_width) / 2 - bar_width / 2)
                        xtick_labels.append(f'{lat}ms')
                    axs.bar(x_base, throughput_rate,
                            bar_width, alpha=0.5, color=colors[i], hatch='//', edgecolor='white')
                    axs.bar(x_base, goodput_rate,
                            bar_width, label=system_labels[i] if k == 0 and j == 0 else "",
                            color=colors[i], hatch=markers[i], edgecolor='white', linewidth=0.5, hatch_linewidth=0.5)
                    axs.text(x_base, goodput_rate, f'{goodput_rate:.0f}',
                             ha='center', va='bottom', size=12)

        if case == 'distributed':
            striped_patch = mpatches.Patch(facecolor='grey', alpha=0.5, hatch='//', edgecolor='white', label='Throughput')
            solid_patch = mpatches.Patch(facecolor='grey', label='Goodput')
            mpl.rcParams['hatch.linewidth'] = 2
            ax2 = axs.twinx()
            ax2.set_yticks([])
            ax2.legend(handles=[striped_patch, solid_patch], loc='lower center', fontsize=12, frameon=True, bbox_to_anchor=(0.85, -0.12), ncol=1)
        axs.set_xticks(xticks)
        axs.set_xticklabels(xtick_labels, size=13)
        axs.set_yticks([0, 50, 100])
        axs.set_yticklabels([0, 50, 100], size=13, rotation=90)
        axs.set_ylabel('Success Rate (%)', size=13)
        if case == 'centralized':
            axs.legend(fontsize=13, ncol=4)
        plt.tight_layout()
        plt.savefig(f'fcpo-{case}-throughput-comparison.pdf')
        plt.show()


    fig, axs = plt.subplots(1, 1, figsize=(6.5, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    box_width = 0.6
    xticks = []
    xtick_labels = []
    for k, case in enumerate(cases):
        for j, lat in enumerate(latencies):
            for i, sys in enumerate(systems):
                data = []
                for exp in scenarios:
                    total[case][lat][exp]['full_latencies'][sys] = [max(7, x) for x in total[case][lat][exp]['full_latencies'][sys]]
                    data += total[case][lat][exp]['full_latencies'][sys]

                box_id = i + len(systems) * (j + k * len(latencies))
                group_id = box_id // len(systems)
                in_group_index = box_id % len(systems)
                pattern_width = len(systems) * box_width
                cumulative_gap = sum(0.4 if (g % len(latencies)) == 2 else 0.1 for g in range(group_id))
                x_base = group_id * pattern_width + cumulative_gap + in_group_index * box_width
                if in_group_index == 0:
                    xticks.append(x_base + (len(systems) * box_width) / 2 - box_width / 2)
                    if group_id % len(latencies) == 1:
                        xtick_labels.append(f'{lat}ms\n{case_labels[k]}')
                    else:
                        xtick_labels.append(f'{lat}ms')

                axs.boxplot(
                    [data], positions=[x_base], widths=box_width,
                    patch_artist=True, label=system_labels[i] if k == 0 and j == 0 else "",
                    boxprops=dict(facecolor=colors[i], hatch=markers[i], edgecolor='black', linewidth=0.5),
                    medianprops=dict(color='black'),
                    flierprops=dict(marker='o', markersize=3, alpha=0.6)
                )

    axs.set_xticks(xticks)
    axs.set_xticklabels(xtick_labels, size=13)
    axs.set_ylabel('Latency (ms)', size=13)
    axs.set_yscale('log')
    axs.set_ylim(5, 1000)
    axs.legend(fontsize=13, ncol=4, loc='lower center')
    plt.tight_layout()
    plt.savefig(f'fcpo-latency-boxplot.png', dpi=900)
    plt.show()


def individual_figures(ax, directory, data, key, offset, bandwidth_timestamps, bandwidth, row, col):
    if "num_missed_requests" not in data['OURS']:
        data['OURS']["num_missed_requests"] = []
    ax_right = ax.twinx()
    workload = os.path.join(directory, key.replace('.mp4', '.csv'))
    workload_index, workload = load_data(workload)
    for i, wi in enumerate(workload_index):
        idx = min(range(len(data['OURS'][key]['aligned_timestamp'])),
                  key=lambda i: abs(data['OURS'][key]['aligned_timestamp'][i] - wi))
        if i > len(data['OURS']["num_missed_requests"]) - 1:
            data['OURS']["num_missed_requests"].append(0)
        if workload[i] < data['OURS'][key]['throughput'][idx]:
            workload[i] = data['OURS'][key]['throughput'][idx]
        elif workload[i] > data['OURS'][key]['throughput'][idx]:
            data['OURS']["num_missed_requests"][i] += workload[i] - data['OURS'][key]['throughput'][idx]
    ax.plot(workload_index, workload, label='Workload', color='red', linestyle='--', linewidth=1)
    ax.fill_between(workload_index, workload, color='red', alpha=0.2)
    #  find gaps in aligned_timestamps that are larger than 1 minute, then add a new row to the dataframe with throughput 0
    differences = [data['OURS'][key]['aligned_timestamp'][i + 1] - data['OURS'][key]['aligned_timestamp'][i] for i in range(len(data['OURS'][key]['aligned_timestamp']) - 1)]
    for i, diff in enumerate(differences):
        if diff > 1:
            new_row = {'aligned_timestamp': data['OURS'][key]['aligned_timestamp'][i] + 1, 'throughput': 0, 'latency': 0}
            data['OURS'][key] = pd.concat([data['OURS'][key], pd.DataFrame([new_row])], ignore_index=True)
    data['OURS'][key] = data['OURS'][key].sort_values(by='aligned_timestamp')

    ax.plot(data['OURS'][key]['aligned_timestamp'], data['OURS'][key]['throughput'], label=label_map['OURS'], color=colors[0], linewidth=1)
    ax.set_ylim([0, 650])
    bandwidth_timestamps = [b + offset for b in bandwidth_timestamps]
    ax_right.plot(bandwidth_timestamps, bandwidth, label='Bandwidth', color='black', linestyle='dotted', linewidth=1)
    ax_right.set_xlim([0, 30])
    ax_right.set_ylim([0, 300])
    ax_right.set_yticks([0, 100, 200])
    ax_right.set_yticklabels([0, 1, 2], size=10)

    title = key.replace('.mp4', '')
    title = title[0:-1] + ' ' + title[-1:]
    ax.set_title(title, size=12)

    if row != 2:
        ax.set_xticks([])
    else:
        ax.set_xticks([0, 10, 20, 30])
        ax.set_xticklabels([0, 10, 20, 30], size=10)
        if col == 1:
            ax.set_xlabel('Minutes Passed since Start', size=12)

    if col == 0:
        ax.set_yticks([0, 300, 600])
        ax.set_yticklabels([0, 3, 6], size=10)

        ax_right.set_yticks([])

        if row == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend([handles[1]], [labels[1]], fontsize=10, loc='upper center')
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_fontsize(10)
            ax.yaxis.get_offset_text().set_position((-0.01, -1))
        if row == 1:
            ax.set_ylabel('Objects / s', size=12)

    if col == 1:
        ax_right.set_yticks([])
        ax.set_yticks([])
        if row == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend([handles[0]], [labels[0]], fontsize=10, loc='upper center')

    if col == 2:
        ax_right.set_yticks([0, 100, 200, 300])
        ax_right.set_yticklabels([0, 1, 2, 3], size=10)
        ax.set_yticks([])
        if row == 0:
            handles, labels = ax_right.get_legend_handles_labels()
            ax_right.legend([handles[0]], [labels[0]], fontsize=10, loc='upper center')
        if row == 1:
            ax_right.set_ylabel('Bandwidth (Mb/s)', size=12)

    if col == 2 and row == 0:
        ax_right.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax_right.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_right.yaxis.get_offset_text().set_fontsize(10)
        ax_right.yaxis.get_offset_text().set_position((1, -1))
    return data


def overall_performance_timeseries(directory, experiment, xticks=None):
    directory = os.path.join(directory, experiment)
    systems = natsorted(os.listdir(directory))
    if 'main' in experiment:
        systems = ['FCPO', 'Tutti', 'OctopInf', 'BCEdge']
        fig1, ax1 = plt.subplots(1, 1, figsize=(4, 2.75), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        fig2, ax2 = plt.subplots(1, 1, figsize=(4, 2.75), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    else:
        fig1, ax1 = plt.subplots(1, 1, figsize=(4.5, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        fig2, ax2 = plt.subplots(1, 1, figsize=(4.5, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    avg_throughput = {'traffic_throughput': {'total': {}}, 'traffic_goodput': {'total': {}}, 'people_throughput': {'total': {}}, 'people_goodput': {'total': {}}}
    total_data = {}
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
        total_data[d] = dfs_combined
        if d == 'without local optimization':
            d = 'no local opt.'
        ax1.plot(dfs_combined['aligned_timestamp'], dfs_combined['throughput'], styles[j], label=d,
                 color=colors[j], linewidth=1, markevery=0.2)
        ax2.plot(dfs_combined['aligned_timestamp'], dfs_combined['avg_latency'], styles[j], label=d,
                 color=colors[j], linewidth=1, markevery=0.2)

    if 'main' in experiment:
        experiment = 'total'
        ax1.set_ylabel('Throughput (100 obj/s)       ', size=14)
        ax2.set_ylabel('Avg Latency (100 ms)    ', size=14)
    else:
        experiment = 'ablation'
        ax1.set_ylabel('Thrghpt (100 obj / s)       ', size=12)
        ax2.set_ylabel('Latency (100 ms)   ', size=12)


    ax2.set_ylim([0, 1050])
    for ax in [ax1, ax2]:
        ax.set_xlabel('Minutes Passed since Start (min)', size=14)
        ax.set_xscale('symlog', linthresh=40)
        ax.set_xticks([0, 5, 10, 15, 30, 45, 60])
        ax.set_xticklabels([0, 5, 10, 15, 30, 45, 60], size=14)
        ax.set_xlim([0, 60])
    ax1.legend(handles=ax1.get_legend_handles_labels()[0][-2:], labels=ax1.get_legend_handles_labels()[1][-2:], fontsize=14, loc="lower center", ncol=2)
    ax2.legend(handles=ax2.get_legend_handles_labels()[0][:2], labels=ax2.get_legend_handles_labels()[1][:2], fontsize=14, loc="upper center", ncol=2)
    ax1.set_yticks([0, 200, 400, 600, 800, 1000, 1200])
    ax1.set_yticklabels([0, 2, 4, 6, 8, 10, 12], size=12)
    ax2.set_yticks([0, 200, 400, 600, 800, 1000, 1200])
    ax2.set_yticklabels([0, 2, 4, 6, 8, 10, 12], size=12)
    fig1.tight_layout()
    fig1.savefig(f"{experiment}-throughput.pdf")
    fig1.show()
    fig2.tight_layout()
    fig2.savefig(f"{experiment}-latency.pdf")
    fig2.show()

    fig, axs = plt.subplots(2, 2, figsize=(8, 4), gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
    for j, d in enumerate(systems):
        if d not in total_data:
            continue
        dfs_combined = total_data[d]
        k = j // 2
        j = j % 2
        ax = axs[j][k]
        ax_right = ax.twinx()
        ax.plot(dfs_combined['aligned_timestamp'], dfs_combined['throughput'], styles[0], label=f'Throughput',
                color=colors[0], linewidth=1)
        ax_right.plot(dfs_combined['aligned_timestamp'], dfs_combined['avg_latency'], styles[1],
                      color=colors[1], label=f'Latency', linewidth=1, linestyle=(0, (10, 5)))
        if k == 0:
            ax.set_ylabel(' ', size=14)
            ax.set_yticks([0, 200, 400, 600, 800, 1000, 1200])
            ax.set_yticklabels([0, 2, 4, 6, 8, 10, 12], size=12)
            ax_right.set_yticks([])
        if k == 1:
            ax_right.set_ylabel(' ', size=14)
            ax_right.set_yticks([0, 200, 400, 600, 800, 1000])
            ax_right.set_yticklabels([0, 2, 4, 6, 8, 10], size=14)
            ax.set_yticks([])
        if j == 1:
            ax.set_xlabel(' ', size=14)
        ax.set_xticks([0, 5, 15, 30, 45, 60])
        ax.set_xticklabels([0, 5, 15, 30, 45, 60], size=14)
        ax.set_title(d, size=15)
        ax.set_xlim([0, 60])
        ax.set_ylim([0, 1250])
        ax_right.set_ylim([0, 1050])
    fig.legend(handles=ax.get_legend_handles_labels()[0] + ax_right.get_legend_handles_labels()[0],
               labels=ax.get_legend_handles_labels()[1] + ax_right.get_legend_handles_labels()[1],
               loc='center', fontsize=14, bbox_to_anchor=(0.5, 0.715), ncol=2)
    fig.text(0.02, 0.5, 'Total Throughput (100 obj/s)', va='center', rotation='vertical', fontsize=14)
    fig.text(0.96, 0.5, 'Average Latency (100 ms)', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.04, 'Minutes Passed since Start (min)', ha='center', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{experiment}-individual-timeseries.pdf")
    plt.show()

    # plot the throughput and latency for each pipeline
    fig, ax1 = plt.subplots(1, 1, figsize=(2.5, 3), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    base_plot(avg_throughput, ax1, '', True, systems, False, xticks, 'Effective Throughput (100 obj/s)      ')
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
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    data = {}
    algorithms = ['fcpo', 'bce', 'fcpo_no_fl', 'fcpo_cold']
    labels = ['FCPO', 'BCE', 'FCPO w/o FL', 'FCPO (Cold Start)']
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
        ax1.plot(dfs[d]['aligned_timestamp'], dfs[d]['throughput'], styles[j], label=labels[j], color=colors[j],
                 linewidth=1, markevery=0.2)

    ax1.set_xlim([0, 6.5])
    ax1.set_xticklabels([0, 1, 2, 3, 4, 5, '6 min'], size=11)
    ax1.set_ylabel('Throughput (obj / s)     ', size=12)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.legend(fontsize=11, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.savefig('warm-start.pdf')
    plt.show()


def continual_learning_performance(base_directory):
    directory = os.path.join(base_directory, 'fcpo_continual')
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
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
            workload_index, workload = load_data(os.path.join(directory, 'workload.csv'))
            # 3 data sources in experiment
            ax1.plot(workload_index, [w * 3 for w in workload], label='Workload', color='black', linestyle='-.', linewidth=1)
        ax1.plot(df['aligned_timestamp'], df['throughput'], styles[j], label=d, color=colors[j], linewidth=1, markevery=0.2)

    ax1.set_xticks([2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5])
    ax1.set_xticklabels([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], size=10)
    ax1.set_xlabel('Cluster ID for network traces and source videos', size=11)
    ax1.set_ylabel('Throughput (obj / s)   ', size=11)
    ax1.set_ylim([400, 1400])
    ax1.set_yticks([400, 600, 800, 1000, 1200, 1400])
    ax1.tick_params(axis='y', labelsize=10)
    ax1.set_xlim([0, 49])
    ax1.legend(fontsize=10, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.savefig('continual-throughput.pdf')
    plt.show()


def reduced_slo(base_directory):
    base_directory = os.path.join(base_directory, 'reduced_slo')
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    algorithm_names = ['fcpo', 'tuti', 'ppp', 'bce']
    label_map = {'fcpo': 'FCPO', 'bce': 'BCE', 'tuti': 'Tutti', 'ppp': 'OInf'}
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
                0.2, label=label_map[a], color=colors[j], hatch=markers[j], edgecolor='white', linewidth=0.5, hatch_linewidth=0.5)
    ax1.axhline(y=data[slos[0]]['max_traffic_throughput'] + data[slos[0]]['max_people_throughput'], color='black', linestyle='-.',
               linewidth=2, xmin=0.05, xmax=0.95)
    striped_patch = mpatches.Patch(facecolor='grey', alpha=0.5, hatch='//', edgecolor='white', label='Thrpt')
    solid_patch = mpatches.Patch(facecolor='grey', label='Effect. Thrpt')
    line_patch = Line2D([0], [0], color='black', linestyle='-.', linewidth=2, label='Workload')
    mpl.rcParams['hatch.linewidth'] = 2
    ax2 = ax1.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=[striped_patch, solid_patch, line_patch], loc='lower left', fontsize=10, frameon=True, bbox_to_anchor=(0, -0.1), ncol=3)

    ax1.set_ylabel('Throughput (100 obj/s)    ', size=11)
    ax1.set_yticks([0, 200, 400, 600, 800, 1000])
    ax1.set_yticklabels([0, 2, 4, 6, 8, 10], size=11)
    ax1.set_xticks(xs + 0.5 * 0.2 * (len(algorithm_names) - 1))
    ax1.set_xticklabels([str(s) + 'ms' for s in slos], size=11)
    ax1.legend(fontsize=10, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.savefig('reduced-slo.pdf')
    plt.show()


def system_overhead(directory):
    # Memory
    systems = ['fcpo', 'bce', 'ppp']
    labels = ['FCPO', 'BCE', 'Rule']
    fig1, ax1 = plt.subplots(1, 1, figsize=(2.75, 2.1), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    fig2, ax2 = plt.subplots(1, 1, figsize=(2.75, 2.1), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    fig3, ax3 = plt.subplots(1, 1, figsize=(2.75, 2.1), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
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
        ax3.bar(i, (server_memory[system][0] + server_memory[system][1] + (edge_memory[system][0] / 1024)) / 1024, 0.7, label=labels[i], color=colors[i])
        ax1.plot([-.4, 2.4], [(server_memory['ppp'][0] + server_memory['ppp'][1]) / 1024,
                                         (server_memory['ppp'][0] + server_memory['ppp'][1]) / 1024], color='black')
        ax2.plot([-.4, 2.4], [edge_memory['ppp'][0] / (1024 * 1024), edge_memory['ppp'][0] / (1024 * 1024)], color='black')
        ax3.plot([-.4, 2.4], [(server_memory['ppp'][0] + server_memory['ppp'][1] + (edge_memory['ppp'][0] / 1024)) / 1024,
                                         (server_memory['ppp'][0] + server_memory['ppp'][1] + (edge_memory['ppp'][0] / 1024)) / 1024], color='black')
    ax1.legend(fontsize=16, loc='lower center')
    ax1.set_ylabel('Memory (10 GB) ', size=17)
    ax1.set_yticks([60, 80, 100, 120])
    ax1.set_yticklabels([6, 8, 10, 12], size=16)
    ax1.set_ylim([45, 135])
    ax1.set_xticks([])
    ax2.legend(fontsize=16, loc='lower center')
    ax2.set_ylabel('Memory (GB)', size=17)
    ax2.set_yticks([2, 4])
    ax2.set_yticklabels([2, 4], size=16)
    ax2.set_ylim([1.5, 6])
    ax2.set_xticks([])
    ax3.legend(fontsize=16, loc='lower center')
    ax3.set_ylabel('Memory (10 GB)', size=17)
    ax3.set_yticks([60, 80, 100, 120])
    ax3.set_yticklabels([6, 8, 10, 12], size=16)
    ax3.set_ylim([45, 135])
    ax3.set_xticks([])
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig1.savefig('server-memory.pdf')
    fig2.savefig('edge-memory.pdf')
    fig3.savefig('centralized-memory.pdf')
    fig1.show()
    fig2.show()
    fig3.show()

    # Power consumption
    fig1, ax1 = plt.subplots(1, 1, figsize=(6.5, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    energy_labels = ['FCPO', 'BCE', 'Rule']
    if not os.path.exists(os.path.join(directory, '..', 'processed_logs', 'power.pkl')):
        power = full_edge_power('nful', systems)
        for system in systems:
            start_time = power[system]['timestamps'][0]
            power[system]['timestamps'] = [(t - start_time) / 60000000 for t in power[system]['timestamps']]
        with open(os.path.join(directory, '..', 'processed_logs', 'power.pkl'), 'wb') as f:
            pickle.dump(power, f)
    else:
        with open(os.path.join(directory, '..', 'processed_logs', 'power.pkl'), 'rb') as f:
            power = pickle.load(f)

    for i, system in enumerate(systems):
        sys_arr = np.array(power[system]['gpu_0_power_consumption'])
        ppp_arr = np.array(power['ppp']['gpu_0_power_consumption'])
        min_len = min(len(sys_arr), len(ppp_arr))
        diff = sys_arr[:min_len] - ppp_arr[:min_len]
        # for the baseline itself, plot zeros
        if system == 'ppp':
            diff = np.zeros(min_len)
            label = f"{energy_labels[i]} (baseline)"
        else:
            label = energy_labels[i]
        ax1.plot(diff, styles[i], label=label, color=colors[i], linewidth=1, markevery=0.1)
    ax1.legend(fontsize=15, loc='upper right', bbox_to_anchor=(1, 0.9))
    ax1.set_ylabel('Power Use (mW)', size=15)
    ax1.set_yticks([0, 200, 400])
    ax1.set_yticklabels([0, 200, 400], size=15)
    ax1.set_xlim(1000, 2900)
    ax1.set_xticks([1000, 1600, 2200, 2800])
    ax1.set_xticklabels([0, 10, 20, '30min'], size=15)
    fig1.tight_layout()
    fig1.savefig('power-consumption.pdf')
    fig1.show()

    # RL Latency
    fig1, ax1 = plt.subplots(1, 1, figsize=(3.5, 2.2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    fig2, ax2 = plt.subplots(1, 1, figsize=(3.5, 2.2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    x_labels = ['Server', 'AGX', 'NX', 'Nano']
    x = np.arange(len(x_labels))
    data = json.loads(open(os.path.join(directory, 'latency.json')).read())
    i = 0
    for system, values in data.items():
        j = 0
        for value in values['infer'].values():
            if j < len(x_labels):
                ax1.bar(j + i * 0.4, value, 0.4, label=system, color=colors[i])
                j += 1
        j = 0
        for value in values['update'].values():
            if j < len(x_labels):
                ax2.bar(j + i * 0.4, value, 0.4, label=system, color=colors[i])
                j += 1
        i += 1

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend([handles[0], handles[5]], [labels[0], labels[5]], fontsize=15, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))
    ax1.set_xticks(x + 0.2)
    ax1.set_xticklabels(x_labels, size=15, rotation=15)
    ax1.set_ylabel('Latency (ms)', size=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.legend([handles[0], handles[5]], [labels[0], labels[5]], fontsize=15, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))
    ax2.set_xticks(x + 0.2)
    ax2.set_xticklabels(x_labels, size=15, rotation=15)
    ax2.set_ylabel('Latency (100ms)    ', size=15)
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
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, sharey=True, figsize=(2.5, 2.2), gridspec_kw={'height_ratios': [1], 'width_ratios': [2,3]})
    FineTuning_latencies = []
    FL_latencies = []
    with open(os.path.join(directory, 'latest_log_21_20-59-1.csv'), 'r') as f:
        for line in f:
            if 'federatedAggregation' in line:
                FineTuning_latencies.append(int(line.split(',')[1]) / 1000000) # convert to seconds
                FL_latencies.append(int(line.split(',')[2]) / 1000000) # convert to seconds
    ax1b.text(-0.02, 0.02, 'Fine-Tune  ', ha='center', va='bottom', transform=ax1b.transAxes, fontsize=17)
    ax1a.boxplot(FineTuning_latencies, positions=[0], vert=False, widths=0.8, patch_artist=True, boxprops=dict(facecolor=colors[0]),
                medianprops=dict(color='black'), showfliers=True)
    ax1b.text(0.43, 0.37, 'FL Round-Trip', ha='center', va='bottom', transform=ax1b.transAxes, fontsize=17)
    ax1b.boxplot(FL_latencies, positions=[1], vert=False, widths=0.8, patch_artist=True, boxprops=dict(facecolor=colors[0]),
                medianprops=dict(color='black'), showfliers=True)

    ax1a.set_yticks([])
    ax1a.set_xlim(0, 0.25)
    ax1b.set_xlim(3.5, 13)
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
    fig1.subplots_adjust(wspace=0.1, bottom=0.22)
    fig1.savefig('FL-latency.pdf')
    fig1.show()


def rl_param_plot(base_directory, exp):
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
    colors = cm.viridis((z_flat - z_flat.min()) / (z_flat.max() - z_flat.min()))
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
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(['low', 'high'])
    cbar.set_label('Throughput')
    ax.set_zticks([])
    ax.set_yticks(y_flat)
    ax.set_xticks(x_flat)
    plt.tight_layout()
    plt.savefig(f"{exp}.pdf")
    plt.show()


def reward_param_plot(directory):
    params = os.listdir(directory)
    cols = ["theta", "sigma", "phi", "rho"]
    if not os.path.exists(os.path.join(directory, 'processed_logs.csv')):
        data = []
        exp_data = analyze_single_experiment(directory, natsorted(params), 1, 250)
        for param in params:
            folder_path = os.path.join(directory, param)
            # Ensure it's a directory
            if os.path.isdir(folder_path):
                # Split folder name into hyperparameters
                try:
                    theta, sigma, phi, rho = map(float, param.split("-"))
                except ValueError:
                    print(f"Skipping folder {param}, invalid name format")
                    continue


                data.append({
                    "theta": int(theta),
                    "sigma": int(sigma),
                    "phi": int(phi),
                    "rho": int(rho),
                    "throughput": exp_data['traffic_throughput']['total'][param] + exp_data['people_throughput']['total'][param]
                })

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(directory, 'processed_logs.csv'), index=False)
    else:
        df = pd.read_csv(os.path.join(directory, 'processed_logs.csv'))

    hyperparams = ["theta", "sigma", "phi", "rho"]
    pairs = list(itertools.combinations(hyperparams, 2))

    fig, axes = plt.subplots(2, 3, figsize=(8, 5.5))
    axes = axes.flatten()
    for i, (xparam, yparam) in enumerate(pairs):
        ax = axes[i]
        pivot_table = df.pivot_table(
            index=yparam,
            columns=xparam,
            values="throughput",
            aggfunc="mean"
        )
        # draw heatmaps without colorbars; colorbar will be in a separate figure
        sns.heatmap(
            pivot_table,
            ax=ax,
            cmap="viridis",
            cbar=False
        )
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.set_xlabel(xparam, fontsize=14)
        ax.set_ylabel(yparam, fontsize=14)
    plt.tight_layout()
    plt.savefig('reward-hyperparameters-heatmaps.pdf')
    plt.show()

    # separate vertical colorbar figure (very narrow, tall)
    vmin = df["throughput"].min()
    vmax = df["throughput"].max()
    fig_cb = plt.figure(figsize=(0.5, 5.5))
    ax_cb = fig_cb.add_axes([0.05, 0.1, 0.4, 0.85])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # for matplotlib < 3.1 compatibility
    cbar = fig_cb.colorbar(sm, cax=ax_cb, orientation='vertical')
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(['min', 'max'], rotation=90, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    fig_cb.savefig('reward-hyperparameters-colorbar.pdf')
    fig_cb.show()

    #parallel axis plot
    df = df.sort_values(by='throughput', ascending=True).reset_index(drop=True)
    perf = df["throughput"].values
    norm_perf = (perf - perf.min()) / (perf.max() - perf.min())
    colors = cm.viridis(norm_perf)

    fig, ax = plt.subplots(figsize=(8, 3))
    x = list(range(len(cols)))
    # Draw each line as a segment
    for i in range(len(df)):
        y = df.iloc[i].values[:len(cols)]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=[colors[i]], linewidth=1.5, alpha=0.8)
        ax.add_collection(lc)

    # Set axis ticks and labels
    ax.set_xlim([0, len(cols) - 1])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=20)
    ax.set_ylim([0, 16])
    ax.set_yticks(range(0, 17, 2))
    ax.set_yticklabels(range(0, 17, 2), fontsize=14)
    ax.set_ylabel("Hyperparameter Value", fontsize=14)
    ax.set_title("Parallel Coordinates Plot for Reward Hyperparameters", fontsize=12)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=perf.min(), vmax=perf.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Effective Throughput (obj / s)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    ax.tick_params(axis='both', labelsize=12)
    plt.savefig('reward-hyperparameters-parallel-coordinates.pdf')
    plt.show()


def hyperparameter_sensitivity(base_directory):
    base_directory = os.path.join(base_directory, 'hyperparameters')
    for i, exp in enumerate(natsorted(os.listdir(base_directory))):
        if '_' not in exp:
            reward_param_plot(os.path.join(base_directory, exp))
        else:
            rl_param_plot(base_directory, exp)


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


