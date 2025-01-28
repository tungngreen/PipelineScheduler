import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

from objectcount import objectcount
from run_log_analyzes import full_analysis
from final_figures import create_figures


def batch(files):
    results = {}
    for file in files:
        lines = []
        with open(os.path.join(args.directory, file), 'r') as f:
            for line in f:
                data = line.split(']|')
                path = data[0].split('][')
                timestamps = data[1].split('|')[0]
                diffs = data[1].split('|')[1]
                if "yolo" in data[0]:
                    lines.append([path[0].split('|')[2], timestamps.split(','), diffs.replace('\n', '').split(',')])

        results[file] = {}
        total = 0
        for line in lines:
            total += 1
            if line[0] not in results[file]:
                results[file][line[0]] = [[], [], [], []]
            results[file][line[0]][1].append(int(line[1][0]))  # creation time
            results[file][line[0]][2].append(int(line[1][1]))  # end time
            results[file][line[0]][3].append(int(line[2][0]))  # processing times
        min_i = sys.maxsize
        max_i = 0
        for key in results[file]:
            results[file][key][0] = total
            results[file][key][1] = np.min(results[file][key][1])
            if results[file][key][1] < min_i:
                min_i = results[file][key][1]
            results[file][key][2] = np.max(results[file][key][2])
            if results[file][key][2] > max_i:
                max_i = results[file][key][2]
            results[file][key][3] = np.mean(results[file][key][3])  # average inference time
        print(file, total)
        results[file]["0"] = total / ((max_i - min_i) / 1000000)  # throughput per second

    # inference time plot
    batch_plot(results, 3, 'Average Inference Time at Different Batch Sizes and 60 FPS',
               'Avg. Inference Time (μs)')

    # throughput plot
    batch_plot(results, -1, 'Throughput at Different Batch Sizes and 60 FPS', 'Throughput (Objects per Second)')

    # number of frames plot
    batch_plot(results, 0, 'Number of Frames at Different Batch Sizes and 60 FPS', 'Number of Objects')


def batch_plot(data, i, title, ylabel):
    x = []
    y = []
    for file in data:
        x.append(file)
        z = []
        if i > -1:
            for key in data[file]:
                if key != "0":
                    z.append(data[file][key][i])
        else:
            z.append(data[file]["0"])
        y.append(np.median(z))
    plt.bar(x, y, color='green')
    plt.xlabel('Batch Size of Models (except Yolov5)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tick_params(axis='both', which='major')
    plt.show()


def profiles(files):
    results = {}
    for file in files:
        lines = []
        with open(os.path.join(args.directory, file), 'r') as f:
            for line in f:
                data = line.split('|')
                if data[0].split(',')[1] != "1" or "WARMUP" in data[0].split(',')[3]:
                    continue
                lines.append([data[0].split(',')[0], data[1].split(','), data[2].replace('\n', '').split(',')])

        results[file] = {}
        for line in lines:
            if line[0] not in results[file]:
                results[file][line[0]] = [[], [], [], [], [], []]
            results[file][line[0]][0].append(int(line[2][0]))  # generator time
            results[file][line[0]][1].append(int(line[2][1]))  # queuing latency
            results[file][line[0]][2].append(int(line[2][2]))  # preprocessing
            results[file][line[0]][3].append(int(line[2][3]))  # inference
            results[file][line[0]][4].append(int(line[2][4]))  # postprocessing
            results[file][line[0]][5].append(int(line[2][5]))  # sink receive
        for key in results[file]:
            results[file][key][0] = np.mean(results[file][key][0])
            results[file][key][1] = np.mean(results[file][key][1])
            results[file][key][2] = np.mean(results[file][key][2])
            results[file][key][3] = np.mean(results[file][key][3])
            results[file][key][4] = np.mean(results[file][key][4])
            results[file][key][5] = np.mean(results[file][key][5])

    for file in results:
        z = []
        for key in results[file]:
            z.append(results[file][key][3])
        print(file, int(np.round(np.mean(z))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='')
    parser.add_argument('--mode', type=str, default='')
    parser.add_argument('--figs', type=str, default='full,thir,slo,abla,long')
    parser.add_argument('--num_results', type=int, default=3)
    args = parser.parse_args()

    if args.mode == '' or args.directory == '':
        print('Please provide the directory and the mode.')
        sys.exit()
    files = natsorted(os.listdir(args.directory))
    if args.mode == 'batch':
        batch(files)
    elif args.mode == 'profiles':
        profiles(files)
    elif args.mode == 'objectcount':
        objectcount(args, files)
    elif args.mode == 'full':
        full_analysis(args, files)
    elif args.mode == 'final':
        create_figures(args, files)
