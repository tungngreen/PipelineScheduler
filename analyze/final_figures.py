import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from natsort import natsorted
from pandas import DataFrame

from run_log_analyzes import analyze_single_experiment, read_file, get_bandwidths
from database_conn import avg_memory, full_memory, align_timestamps, bucket_and_average, arrival_query
from objectcount import load_data

bar_width = 0.2
x_labels = ['traffic', 'surveillance']
x = np.arange(len(x_labels))
algorithm_names = ['OURS', 'dis', 'jlf', 'rim']
colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#F0E442', '#D55E00']
markers = ['', 'x', 'o', '*', '//', '\\\\']  # plain, triangle, circle, star, stripes left and right
label_map = {'ppp': 'OctopInf', 'OURS': 'OctopInf', 'dis': 'Distream', 'jlf': 'Jellyfish', 'rim': 'Rim'}


def base_plot(data, ax, title, sum_throughput=False):
    for j, a in enumerate(algorithm_names):
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
        if sum_throughput:
            ax.bar(j * bar_width, [traffic_intime + people_intime], bar_width, label=label_map[a], color=colors[j],
                   edgecolor='white', linewidth=0.5)
            ax.text(j * bar_width, traffic_intime + people_intime, f'{traffic_intime + people_intime:.0f}', ha='center',
                    va='bottom', size=10)
        else:
            ax.bar(x + j * bar_width, [traffic_intime, people_intime], bar_width, label=label_map[a], color=colors[j],
                   hatch=markers[j], edgecolor='white', linewidth=0.5)
            ax.text(x[0] + j * bar_width, traffic_intime, f'{traffic_intime:.0f}', ha='center', va='bottom', size=10)
            ax.text(x[1] + j * bar_width, people_intime, f'{people_intime:.0f}', ha='center', va='bottom', size=10)

        if a == 'OURS':
            striped_patch = mpatches.Patch(facecolor='grey', alpha=0.5, hatch='//', edgecolor='white', label='Thrpt')
            solid_patch = mpatches.Patch(facecolor='grey', label='Effect. Thrpt')
            line_patch = Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Workload')
            mpl.rcParams['hatch.linewidth'] = 2

            if sum_throughput:
                ax.legend(handles=[striped_patch, solid_patch, line_patch], loc='lower left', fontsize=10, frameon=True)
            else:
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.legend(handles=[striped_patch, solid_patch, line_patch], loc='upper center', fontsize=10, frameon=True)

    if sum_throughput:
        ax.axhline(y=data['max_traffic_throughput'] + data['max_people_throughput'], color='red', linestyle='--',
                   linewidth=2, xmin=0.05, xmax=0.95)
        ax.set_xticks([0, 0.2, 0.4, 0.6])
        ax.set_xticklabels(['OInf', 'Dis', 'Jlf', 'Rim'], size=10)
        ax.set_yticks([0, 500, 1000, 1500])
        ax.set_yticklabels([0, 5, 10, 15], size=10)
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
    ax.set_ylabel('Objects / s', size=12)

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
    if 'full' in figs:
        bandwidths = get_bandwidths(args.directory)
        bandwidths_timestamps = {}
        for b in bandwidths:
            bandwidths_timestamps[b] = bandwidths[b][1]
            bandwidths[b] = bandwidths[b][0]

    ########################################################################################################################
    ##### Full Experiment Plot #############################################################################################
    ########################################################################################################################

    if 'full' in figs:

        data, detailed_data = {}, {}
        keys = []
        for d in natsorted(os.listdir(os.path.join(args.directory, 'bndw'))):
            data[d] = []
            keys.append(d)
            for f in natsorted(os.listdir(os.path.join(args.directory, 'bndw', d))):
                if f.endswith('.csv'):
                    continue
                c, p, _, _ = read_file(os.path.join(args.directory, 'bndw', d, f))
                data[d].extend(c)
                data[d].extend(p)
        total_workloads_index, total_workloads = [], []
        for j, d in enumerate(keys):
            df = pd.DataFrame(data[d], columns=['path', 'latency', 'timestamps'])
            df['latency'] = df['latency'].apply(lambda x: int(x))
            df['timestamps'] = df['timestamps'].apply(lambda x: int(x))
            df['aligned_timestamp'] = (df['timestamps'] // 1e6).astype(int) * 1e6
            detailed_data[d] = {}

            # if directory does not contain csv files, create them
            if not os.path.exists(os.path.join(args.directory, 'bndw', d, 'traffic1.mp4.csv')):
                for i, row in df.iterrows():
                    if row['path'] not in detailed_data[d]:
                        detailed_data[d][row['path']] = []
                    detailed_data[d][row['path']].append(row)
                for source in detailed_data[d]:
                    if not os.path.exists(os.path.join(args.directory, 'bndw', d, f'{source}.csv')):
                        detailed_data[d][source] = prepare_timeseries(pd.DataFrame(detailed_data[d][source]), False)
                        detailed_data[d][source].to_csv(os.path.join(args.directory, 'bndw', d, f'{source}.csv'), index=False)
                    else:
                        detailed_data[d][source] = pd.read_csv(os.path.join(args.directory, 'bndw', d, f'{source}.csv'))
            else:
                for source in ['traffic1.mp4', 'traffic2.mp4', 'traffic3.mp4', 'traffic4.mp4', 'traffic5.mp4',
                               'traffic6.mp4', 'people1.mp4', 'people2.mp4', 'people3.mp4']:
                    detailed_data[d][source] = pd.read_csv(os.path.join(args.directory, 'bndw', d, f'{source}.csv'))
                    if d == 'OURS':
                        workload_path = os.path.join(args.directory, 'workload', source.replace('.mp4', '.csv'))
                        workload_index, workload = load_data(workload_path)
                        for i, wi in enumerate(workload_index):
                            if (len(total_workloads) <= i):
                                total_workloads.append(0)
                            total_workloads[i] += workload[i]
                            if (i >= len(total_workloads_index)):
                                total_workloads_index.append(wi)

            data[d] = prepare_timeseries(df, False)

    ########################################################################################################################
    ##### Individual Patterns Plot #########################################################################################
    ########################################################################################################################

        fig, axs = plt.subplots(3, 3, figsize=(7.5, 7.5/2))

        detailed_data = individual_figures(axs[0,0], os.path.join(args.directory, 'workload'), detailed_data, 'traffic1.mp4', 0.5,
                           bandwidths_timestamps['bandwidth_limits5.json'], bandwidths['bandwidth_limits5.json'], 0, 0)
        detailed_data = individual_figures(axs[0,1], os.path.join(args.directory, 'workload'), detailed_data, 'traffic2.mp4', 0,
                           bandwidths_timestamps['bandwidth_limits4.json'], bandwidths['bandwidth_limits4.json'], 0, 1)
        detailed_data = individual_figures(axs[0,2], os.path.join(args.directory, 'workload'), detailed_data, 'traffic3.mp4', -0.4,
                           bandwidths_timestamps['bandwidth_limits6.json'], bandwidths['bandwidth_limits6.json'], 0, 2)
        detailed_data = individual_figures(axs[1,0], os.path.join(args.directory, 'workload'), detailed_data, 'traffic4.mp4', 0,
                           bandwidths_timestamps['bandwidth_limits3.json'], bandwidths['bandwidth_limits3.json'], 1, 0)
        detailed_data = individual_figures(axs[1,1], os.path.join(args.directory, 'workload'), detailed_data, 'traffic5.mp4', -0.8,
                           bandwidths_timestamps['bandwidth_limits1.json'], bandwidths['bandwidth_limits1.json'], 1, 1)
        detailed_data = individual_figures(axs[1,2], os.path.join(args.directory, 'workload'), detailed_data, 'traffic6.mp4', -0.5,
                           bandwidths_timestamps['bandwidth_limits7.json'], bandwidths['bandwidth_limits7.json'], 1, 2)
        detailed_data = individual_figures(axs[2,0], os.path.join(args.directory, 'workload'), detailed_data, 'people1.mp4', 0,
                           bandwidths_timestamps['bandwidth_limits2.json'], bandwidths['bandwidth_limits2.json'], 2, 0)
        detailed_data = individual_figures(axs[2,1], os.path.join(args.directory, 'workload'), detailed_data, 'people2.mp4', -0.5,
                           bandwidths_timestamps['bandwidth_limits8.json'], bandwidths['bandwidth_limits8.json'], 2, 1)
        detailed_data = individual_figures(axs[2,2], os.path.join(args.directory, 'workload'), detailed_data, 'people3.mp4', 0.7,
                           bandwidths_timestamps['bandwidth_limits9.json'], bandwidths['bandwidth_limits9.json'], 2, 2)

        plt.subplots_adjust(wspace=0.004, hspace=0)
        plt.tight_layout(pad=0.04)
        plt.savefig('individual-patterns.pdf')
        plt.show()

    ########################################################################################################################
    ##### Main Experiment Plot #############################################################################################
    ########################################################################################################################

        fig, axs = plt.subplots(2, 3, figsize=(8, 4.5),
                                gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [6, 3.5, 3.5]})
        axs[1, 1].remove()
        axs[1, 2].remove()
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        ax1 = axs[0, 0]
        ax2 = axs[0, 1]
        ax3 = axs[0, 2]
        ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        base_plot(total['full'], ax1, 'a) Throughput', True)
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(2, 2))
        ax1.yaxis.get_offset_text().set_fontsize(10)
        ax1.yaxis.get_offset_text().set_position((-0.02, -10))
        ax1.margins(x=0.1)
        for j, a in enumerate(algorithm_names):
            ax2.boxplot([total['full']['traffic_latency']['total'][a]], positions=[j], vert=False, widths=0.8,
                        patch_artist=True, boxprops=dict(facecolor=colors[j]), medianprops=dict(color='black'),
                        showfliers=False)
        ax2.set_xlim([0, 650])
        ax2.set_xticks([0, 150, 300, 450, 600])
        ax2.set_xticklabels([0, 150, 300, 450, '600ms'], size=10)
        ax2.set_yticklabels(['OInf', 'Dis', 'Jlf', 'Rim'], size=10)
        ax2.set_ylabel('', size=1)
        ax2.set_title('b) Latency Distribution', size=12)
        memory_plot('full', ax3, True)

        for j, d in enumerate(keys):
            ax4.plot(data[d]['aligned_timestamp'], data[d]['throughput'],
                     label=label_map[d], color=colors[j],
                     marker=markers[j % len(markers)],  # Use markers in a cycle
                     linewidth=2, markersize=6, markevery=3)

        for i, wi in enumerate(total_workloads_index):
            differences = [abs(t - wi) for t in data['OURS']['aligned_timestamp']]
            idx = differences.index(min(differences))
            if (total_workloads[i] < data['OURS']['throughput'][idx]):
                total_workloads[i] = data['OURS']['throughput'][idx] + detailed_data['OURS']['num_missed_requests'][i]
        ax4.plot(total_workloads_index, total_workloads, label='Workload',
                 color='red', linestyle='--', linewidth=2)

        ax4.set_ylim([0, 2500])
        ax4.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax4.ticklabel_format(style='sci', axis='y', scilimits=(2, 2))
        ax4.yaxis.get_offset_text().set_fontsize(10)
        ax4.yaxis.get_offset_text().set_position((-0.02, 5))
        ax4.set_title('d) Effective Throughput Comparison over time', size=12)
        ax4.set_xlabel('Minutes Passed since Start', size=12)
        ax4.set_ylabel('Objects / s', size=12)
        ax4.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax4.set_xticklabels([0, 5, 10, 15, 20, 25, 30], size=10)
        ax4.set_xlim([0, 30])
        ax4.legend(loc='upper right', ncol=2, columnspacing=1.0, handletextpad=0.5, handlelength=2, fontsize=10,
                   frameon=True)

        plt.subplots_adjust(wspace=0.0, hspace=0)
        plt.tight_layout(pad=0.4)
        plt.savefig('full-experiment.pdf')
        plt.show()


    ########################################################################################################################
    ##### 30FPS Experiment Plot ############################################################################################
    ########################################################################################################################

    if 'thir' in figs:
        fig, axs = plt.subplots(1, 2, figsize=(8, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [3, 1]})
        ax1 = axs[0]
        ax2 = axs[1]
        base_plot(total['thir'], ax1, 'a) Throughput Comparisons under Higher Workloads')
        memory_plot('full', ax2, False)
        plt.tight_layout()
        plt.savefig('30-fps-experiment.pdf')
        plt.show()

    ########################################################################################################################
    ##### Ablation Experiment Plot #########################################################################################
    ########################################################################################################################

    if 'abla' in figs:
        fig, axs = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'height_ratios': [1], 'width_ratios': [4, 2]})
        ax1 = axs[0]
        ax2 = axs[1]
        versions = ['OURS', 'dis', 'jlf', 'ppwt', 'ppsb', 'ppos']
        label_names = {'OURS': 'OctopInf', 'jlf': 'Jellyfish', 'dis': 'Distream', 'ppsb': 'Static Batch', 'ppos': 'Server Only', 'pposwt': 'Server Only w/o Coral', 'ppwt': 'w/o Coral'}
        for j, a in enumerate(versions):
            val = int(total['abla']['traffic_goodput']['total'][a]) + int(total['abla']['people_goodput']['total'][a])
            ax1.bar(j, [val], label=label_names[a], color=colors[j], hatch=markers[j], edgecolor='white', linewidth=0.5)
            ax1.text(j, val, f'{val:.0f}', ha='center', va='bottom', size=10)
        ax1.set_title('a) Effective Thrpt of Both Pipelines in Different Configurations', size=12)
        ax1.set_ylabel('Objects / s', size=12)
        ax1.set_ylim([0, 1750])
        ax1.set_yticks([0, 500, 1000, 1500])
        ax1.set_yticklabels([0, 500, 1000, 1500], size=10)
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.1), fontsize=10, ncol=3)
        dec = 0
        for j, a in enumerate(versions):
            if a == 'jlf' or a == 'dis':
                dec += 1
                continue
            box = ax2.boxplot([total['abla']['traffic_latency']['total'][a]], positions=[j-dec], vert=True, widths=0.9,
                        patch_artist=True, boxprops=dict(facecolor=colors[j]),
                        medianprops=dict(color='black'), showfliers=False)
            for item in box['boxes']:
                item.set(hatch=markers[j], edgecolor='white', linewidth=0.5)
            ax2.boxplot([total['abla']['traffic_latency']['total'][a]], positions=[j - dec], vert=True,
                                  widths=0.9, patch_artist=True, boxprops=dict(facecolor='none'),
                                  medianprops=dict(color='black'), showfliers=False)
        ax2.set_ylim([0, 230])
        ax2.set_yticks([0, 50, 100, 150, 200])
        ax2.set_yticklabels([0, 50, 100, 150, '200\nms'], size=10)
        ax2.set_xticks([])
        ax2.set_title('b) End-to-End Latency', size=12)
        plt.tight_layout()
        plt.savefig('ablation-experiment.pdf')
        plt.show()

    ########################################################################################################################
    ##### Reduced SLO Experiment Plot ######################################################################################
    ########################################################################################################################

    if 'slo' in figs:
        fig, axs = plt.subplots(1, 1, figsize=(8, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        ax1 = axs
        slos = ['full', 'lslo', 'mslo']
        slo_labels = {'full': 'Full SLO', 'lslo': 'SLO reduced by 50ms', 'mslo': 'SLO reduced by 100ms'}
        xs = np.arange(len(slos))
        for j, a in enumerate(algorithm_names):
            val = {}
            for s in slos:
                val[s] = int(total[s]['traffic_throughput']['total'][a]) + int(total[s]['people_throughput']['total'][a])
            ax1.bar(xs + j * bar_width, [val['full'], val['lslo'], val['mslo']], bar_width, alpha=0.5,
                    color=colors[j], hatch='//', edgecolor='white')
            for s in slos:
                val[s] = int(total[s]['traffic_goodput']['total'][a]) + int(total[s]['people_goodput']['total'][a])
            ax1.bar(xs + j * bar_width, [val['full'], val['lslo'], val['mslo']], bar_width, label=label_map[a],
                    color=colors[j], hatch=markers[j], edgecolor='white', linewidth=0.5)
            ax1.text(xs[0] + j * bar_width, val['full'], f'{val["full"]:.0f}', ha='center', va='bottom', size=10)
            ax1.text(xs[1] + j * bar_width, val['lslo'], f'{val["lslo"]:.0f}', ha='center', va='bottom', size=10)
            ax1.text(xs[2] + j * bar_width, val['mslo'], f'{val["mslo"]:.0f}', ha='center', va='bottom', size=10)
        ax1.set_ylabel('Objects / s', size=12)
        ax1.set_xticks(xs + 0.5 * bar_width * (len(algorithm_names) - 1))
        ax1.set_xticklabels([slo_labels[s] for s in slos], size=10)
        ax1.legend(fontsize=10)

        ax2 = ax1.twinx()
        ax2.set_yticks([])
        striped_patch = mpatches.Patch(facecolor='grey', alpha=0.5, hatch='//', edgecolor='white', label='Thrpt')
        solid_patch = mpatches.Patch(facecolor='grey', label='Effect. Thrpt')
        line_patch = Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Workload')
        mpl.rcParams['hatch.linewidth'] = 2
        ax2.legend(handles=[striped_patch, solid_patch, line_patch], loc='lower left', fontsize=10, frameon=True, ncol=3)
        plt.tight_layout()
        plt.savefig('slo-experiment.pdf')
        plt.show()

    ########################################################################################################################
    ##### Long Runtime Experiment Plot #####################################################################################
    ########################################################################################################################

    if 'long' in figs:
        fig, axs = plt.subplots(1, 1, figsize=(8, 2.5), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        ax1 = axs
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
        labels = ['Traffic Throughput', 'Surveillance Throughput']

        for j, df in enumerate([df_cars, df_people]):
            df['timestamps'] = df['timestamps'].apply(lambda x: int(x))
            df['latency'] = df['latency'].apply(lambda x: int(x))
            df['aligned_timestamp'] = (df['timestamps'] // 1e6).astype(int) * 1e6
            df['throughput'] = df.groupby('aligned_timestamp')['latency'].transform('size')
            df['aligned_timestamp'] = (df['aligned_timestamp'] - df['aligned_timestamp'].iloc[0]) / (60 * 1e6)
            df = df.groupby('aligned_timestamp').agg({'throughput': 'mean'}).reset_index()
            df = bucket_and_average(df, ['throughput'], num_buckets=780)
            if j == 0:
                df_cars = df
            elif j == 1:
                df_people = df

        if not os.path.exists(os.path.join(args.directory, 'long', 'workloads.csv')):
            workload_index, traffic_workload, people_workload = [], [], []
            for f in natsorted(os.listdir(os.path.join(args.directory, 'workload_13h'))):
                wi, single_workload = load_data(os.path.join(args.directory, 'workload_13h', f), 1800, 0)
                if len(workload_index) == 0:
                    workload_index = wi
                    traffic_workload = np.array(single_workload) if 'traffic' in f else np.zeros(len(wi))
                    people_workload = np.array(single_workload) if 'people' in f else np.zeros(len(wi))
                else:
                    workload = np.array(single_workload)
                    if 'traffic' in f:
                        traffic_workload = np.add(traffic_workload, workload)
                    else:
                        people_workload = np.add(people_workload, workload)

            # Vectorized closest index calculation
            aligned_cars = np.array(df_cars['aligned_timestamp'])
            aligned_people = np.array(df_people['aligned_timestamp'])
            throughput_cars = np.array(df_cars['throughput'])
            throughput_people = np.array(df_people['throughput'])

            for i, wi in enumerate(workload_index):
                idx_cars = closest_index(aligned_cars, wi)
                idx_people = closest_index(aligned_people, wi)

                throughput_cars_value = throughput_cars[idx_cars] if idx_cars < len(throughput_cars) else 0
                if traffic_workload[i] < throughput_cars_value:
                    traffic_workload[i] = throughput_cars_value
                else:
                    traffic_workload[i] -= (traffic_workload[i] - throughput_cars_value) / 2
                throughput_people_value = throughput_people[idx_people] if idx_people < len(throughput_people) else 0
                if people_workload[i] < throughput_people_value:
                    people_workload[i] = throughput_people_value
                else:
                    people_workload[i] -= (people_workload[i] - throughput_people_value) / 2

            # Save workloads to a CSV file
            workloads = pd.DataFrame({
                'timestamp': workload_index,
                'traffic_workload': traffic_workload,
                'people_workload': people_workload,
            })
            workloads.to_csv(os.path.join(args.directory, 'long', 'workloads.csv'), index=False)
        else:
            workloads = pd.read_csv(os.path.join(args.directory, 'long', 'workloads.csv'))
        ax1.plot(workloads['timestamp'], workloads['traffic_workload'], label='Traffic Workload', color='red',
                 linestyle='--', linewidth=2, marker='2', markevery=10)
        ax1.plot(workloads['timestamp'], workloads['people_workload'] + 50, label='Surveillance Workload', color='darkred',
                 linestyle='--', linewidth=2, marker='^', markevery=10)
        ax1.plot(workloads['timestamp'], workloads['traffic_workload'], label='Traffic Workload', color='red',
                 linewidth=1)
        ax1.plot(workloads['timestamp'], workloads['people_workload'] + 50, label='Surveillance Workload',
                 color='darkred', linewidth=1)

        ax1.plot(df_cars['aligned_timestamp'], df_cars['throughput'], label=labels[0], color=colors[0], linewidth=1,
                 marker='1', markevery=10)
        ax1.plot(df_people['aligned_timestamp'], df_people['throughput'], label=labels[1], color=colors[1],
                 linewidth=1, marker='v', markevery=10)
        #if not os.path.exists(os.path.join(args.directory, 'long', 'missed.csv')):
        #    missed = bucket_and_average(arrival_query('long_ppp'), ['queue_drops', 'late_requests'], num_buckets=780)
        #    missed.to_csv(os.path.join(args.directory, 'long', 'missed.csv'), index=False)
        #else:
        #    missed = pd.read_csv(os.path.join(args.directory, 'long', 'missed.csv'))
        #ax1.plot(missed['aligned_timestamp'], missed['queue_drops'], label='Dropped in Queues', color=colors[2], linewidth=2)
        #ax1.plot(missed['aligned_timestamp'], missed['late_requests'] / 2, label='Late Requests', color=colors[5], linewidth=2)
        ax1.set_xlim([0, 780])
        ax1.set_xticks([0, 100, 200, 300, 400, 500, 600, 700])
        ax1.set_xticklabels([0, 100, 200, 300, 400, 500, 600, '700min'], size=10)
        ax1.set_ylabel('Objects / s', size=12)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend([handles[0], handles[4], handles[1], handles[5]], [labels[0], labels[4], labels[1], labels[5]], loc='lower left', fontsize=10)
        plt.tight_layout()
        plt.savefig('long-experiment.pdf')
        plt.show()

