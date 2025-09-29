import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

def get_total_objects(dir, path='full_run_total.json'):
    traffic_people, traffic_cars, people_people, people_cars = 0, 0, 0, 0
    try:
        with open(os.path.join(dir, path), 'r') as file:
            meta = json.load(file)
    except FileNotFoundError:
        print("Error: Json file not found.")
        return 1, 1, 1, 1
    for key in meta:
        if "traffic" in key:
            traffic_people += meta[key]["people"]
            traffic_cars += meta[key]["cars"]
        else:
            people_people += meta[key]["people"]
            people_cars += meta[key]["cars"]
    if 'thir' in dir:
        f = 1
    else:
        f = 0.5
    return traffic_people * 0.5 * f, traffic_cars * 2 * f, people_people * 2.5 * f, people_cars * 0


def bar_plot(dirs, cars, people, total, title, xlabel, ylabel, ax):
    bar_width = 0.2
    x_labels = ['Cars', 'People', 'Total']
    x = np.arange(len(x_labels))
    for i, d in enumerate(dirs):
        bar = ax.bar(x - bar_width, [cars[d], people[d], total[d]], color='C' + str(i), width=bar_width,
                     label=d)
        x = x + bar_width

        # Add the total value of each bar on top of it
        for rect in bar:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([r + 0.5 * bar_width for r in range(len(x_labels))])
    ax.set_xticklabels(x_labels)
    ax.legend()


def read_file(filepath, first = None, last = None):
    people = []
    cars = []
    with (open(filepath, 'r') as f):
        for line in f:
            data = line.split(']|')
            path = data[0].split('][')
            i = len(data) - 1
            timestamps = data[i].split('|')[0]
            if first:
                if int(timestamps.split(',')[0]) < first:
                    first = int(timestamps.split(',')[0])
            if last:
                if int(timestamps.split(',')[1]) > last:
                    last = int(timestamps.split(',')[1])
            diffs = data[i].split('|')[1]
            if "retina" in data[0] or "movenet" in data[0]:
                people.append(
                    [path[0].split('|')[1], diffs.replace('\n', '').split(',')[0], timestamps.split(',')[0]])  # store source, latency and arrival timestamp
            else:
                cars.append(
                    [path[0].split('|')[1], diffs.replace('\n', '').split(',')[0], timestamps.split(',')[0]])
    return cars, people, first, last


def analyze_single_experiment(base_dir, dirs, num_results = 3, latency_target = 200, combined = False, running_seconds = 1800):
    latency_target *= 1000
    people_latency_target = latency_target + 100000
    first, last = {}, {}

    dirs_copy = dirs.copy()
    for d in dirs_copy:
        if not os.path.isdir(os.path.join(base_dir, d)):
            dirs.remove(d)

    if combined:
        cars, people = {}, {}
        for d in dirs:
            filepath = os.path.join(base_dir, d)
            if not os.path.isdir(filepath):
                continue
            first[d], last[d] = sys.maxsize, 0
            cars[d], people[d] = [], []
            for file in natsorted(os.listdir(filepath)):
                if not 'txt' in file:
                    continue
                c, p, first[d], last[d] = read_file(os.path.join(filepath, file), first[d], last[d])
                cars[d].extend(c)
                people[d].extend(p)

        total = 0
        for d in dirs:
            total += min(100 * (len(cars[d]) + len(people[d])) / num_results, 100.0)

        throughput = {}
        for d in dirs:
            throughput[d] = (len(cars[d]) + len(people[d])) / num_results / running_seconds
        goodput = {}
        for d in dirs:
            c = len([x for x in cars[d] if int(x[1]) < latency_target])
            p = len([x for x in people[d] if int(x[1]) < people_latency_target])
            goodput[d] = (c + p) / num_results / running_seconds
        avg_latency = {}
        for d in dirs:
            avg_latency[d] = sum([int(x[1]) / 1000 for x in cars[d]] + [int(x[1]) / 1000 for x in people[d]]) / (len(cars[d]) + len(people[d])) if (len(cars[d]) + len(people[d])) > 0 else 1

        return {'throughput': throughput,
                'goodput': goodput,
                'latency': avg_latency,
                'max_throughput': (total) / running_seconds}
    else:
        traffic_people, traffic_cars, people_people, people_cars = {}, {}, {}, {}
        total_traffic_people, total_traffic_cars, total_people_people, total_people_cars = get_total_objects(base_dir)

        for d in dirs:
            filepath = os.path.join(base_dir, d)
            if not os.path.isdir(filepath):
                continue
            first[d], last[d] = sys.maxsize, 0
            traffic_people[d], traffic_cars[d], people_people[d], people_cars[d] = [], [], [], []
            for file in natsorted(os.listdir(filepath)):
                if not 'txt' in file:
                    continue
                cars, people, first[d], last[d] = read_file(os.path.join(filepath, file), first[d], last[d])
                if "traffic" in file:
                    traffic_people[d].extend(people)
                    traffic_cars[d].extend(cars)
                else:
                    people_people[d].extend(people)
                    people_cars[d].extend(cars)

        cars = {}
        people = {}
        total = {}
        for d in dirs:
            cars[d] = min(100 * len(traffic_cars[d]) / num_results / total_traffic_cars, 100.0)
            people[d] = min(100 * len(traffic_people[d]) / num_results / total_traffic_people, 100.0)
            total[d] = min(100 * (len(traffic_cars[d]) + len(traffic_people[d])) / num_results / (total_traffic_cars + total_traffic_people), 100.0)
        traffic_total_arr_percentage = {'cars': cars.copy(),'people': people.copy(),'total': total.copy()}

        for d in dirs:
            c = len([x for x in traffic_cars[d] if int(x[1]) < latency_target]) / num_results
            p = len([x for x in traffic_people[d] if int(x[1]) < latency_target]) / num_results
            cars[d] = min(100 * c / total_traffic_cars, 100.0)
            people[d] = min(100 * p / total_traffic_people, 100.0)
            total[d] = min(100 * (c + p) / (total_traffic_people + total_traffic_cars), 100.0)
        traffic_intime_arr_percentage = {'cars': cars.copy(),'people': people.copy(),'total': total.copy()}

        for d in cars:
            cars[d] = len(traffic_cars[d]) / num_results / running_seconds
            people[d] = len(traffic_people[d]) / num_results / running_seconds
            total[d] = (len(traffic_cars[d]) + len(traffic_people[d])) / num_results / running_seconds
        traffic_throughput = {'cars': cars.copy(),'people': people.copy(),'total': total.copy()}

        for d in dirs:
            c = len([x for x in traffic_cars[d] if int(x[1]) < latency_target])
            p = len([x for x in traffic_people[d] if int(x[1]) < latency_target])
            cars[d] = c / num_results / running_seconds
            people[d] = p / num_results / running_seconds
            total[d] = (c + p) / num_results / running_seconds
        traffic_goodput = {'cars': cars.copy(),'people': people.copy(),'total': total.copy()}

        for d in dirs:
            cars[d] = [int(x[1]) / 1000 for x in traffic_cars[d]]
            people[d] = [int(x[1]) / 1000 for x in traffic_people[d]]
            total[d] = cars[d] + people[d]
        traffic_latency = {'cars': cars.copy(),'people': people.copy(),'total': total.copy()}

        for d in dirs:
            cars[d] = 0
            people[d] = min(100 * len(people_people[d]) / num_results / total_people_people, 100.0)
            total[d] = min(100 * (len(people_cars[d]) + len(people_people[d])) / num_results / total_people_people, 100.0)
        people_total_arr_percentage = {'cars': cars.copy(),'people': people.copy(),'total': total.copy()}

        for d in dirs:
            c = len([x for x in people_cars[d] if int(x[1]) < people_latency_target]) / num_results
            p = len([x for x in people_people[d] if int(x[1]) < people_latency_target]) / num_results
            cars[d] = 0
            people[d] = min(100 * p / total_people_people, 100.0)
            total[d] = min(100 * (c + p) / total_people_people, 100.0)
        people_intime_arr_percentage = {'cars': cars.copy(),'people': people.copy(),'total': total.copy()}

        for d in dirs:
            cars[d] = 0
            people[d] = len(people_people[d]) / num_results / running_seconds
            total[d] = len(people_people[d]) / num_results / running_seconds
        people_throughput = {'cars': cars.copy(),'people': people.copy(),'total': total.copy()}

        for d in dirs:
            cars[d] = 0
            p = len([x for x in people_people[d] if int(x[1]) < people_latency_target])
            people[d] = p / num_results / running_seconds
            total[d] = p / num_results / running_seconds
        people_goodput = {'cars': cars.copy(),'people': people.copy(),'total': total.copy()}

        for d in dirs:
            cars[d] = [0 for _ in people_cars[d]]
            people[d] = [int(x[1]) / 1000 for x in people_people[d]]
            total[d] = cars[d] + people[d]
        people_latency = {'cars': cars.copy(),'people': people.copy(),'total': total.copy()}

        return {'traffic_total': traffic_total_arr_percentage,
                'traffic_intime': traffic_intime_arr_percentage,
                'traffic_throughput': traffic_throughput,
                'traffic_goodput': traffic_goodput,
                'traffic_latency': traffic_latency,
                'people_total': people_total_arr_percentage,
                'people_intime': people_intime_arr_percentage,
                'people_throughput': people_throughput,
                'people_goodput': people_goodput,
                'people_latency': people_latency,
                'max_traffic_throughput': (total_traffic_people + total_traffic_cars) / running_seconds,
                'max_people_throughput': total_people_people / running_seconds}


def full_analysis(args, dirs):
    data = analyze_single_experiment(args.directory, dirs, args.num_results)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    bar_plot(dirs, data['traffic_total']['cars'], data['traffic_total']['people'], data['traffic_total']['total'], 'Arrival Percentage of Traffic by Type', 'Object Type', 'Percentage', axs[0])
    bar_plot(dirs, data['traffic_intime']['cars'], data['traffic_intime']['people'], data['traffic_intime']['total'], 'Arrival Percentage of Traffic by Type (Latency < 200ms)', 'Object Type', 'Percentage', axs[1])
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    bar_plot(dirs, data['people_total']['cars'], data['people_total']['people'], data['people_total']['total'], 'Arrival Percentage of Surveillance by Type', 'Object Type', 'Percentage', axs[0])
    bar_plot(dirs, data['people_intime']['cars'], data['people_intime']['people'], data['people_intime']['total'], 'Arrival Percentage of Surveillance by Type (Latency < 300ms)', 'Object Type', 'Percentage', axs[1])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    bar_width = 0.2
    x_labels = ['Cars', 'People', 'Total']
    x = np.arange(len(x_labels))
    for i, d in enumerate(dirs):
        ax.boxplot([data['traffic_latency']['cars'][d], data['traffic_latency']['people'][d], data['traffic_latency']['total'][d]], positions=x - bar_width + i * bar_width, widths=bar_width,
                   patch_artist=True, boxprops=dict(facecolor='C' + str(i)), medianprops=dict(color='black'),
                   vert=False)

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Category')
    ax.set_xlim([0, 1000])
    ax.set_title('Latency Distribution of Traffic Objects')
    ax.set_yticks([r + 0.5 * bar_width for r in x])
    ax.set_yticklabels(x_labels)
    ax.legend([plt.Line2D((0, 1), (0, 0), color='C' + str(i), marker='o', linestyle='') for i in range(len(dirs))],
              dirs)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    x_labels = ['Total']
    x = np.arange(len(x_labels))
    for i, d in enumerate(dirs):
        ax.boxplot([data['people_latency']['people'][d]], positions=x - bar_width + i * bar_width, widths=bar_width,
                   patch_artist=True, boxprops=dict(facecolor='C' + str(i)), medianprops=dict(color='black'),
                   vert=False)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Category')
    ax.set_xlim([0, 4500])
    ax.set_title('Latency Distribution of Surveillance Objects')
    ax.set_yticks([r + 0.5 * bar_width for r in x])
    ax.set_yticklabels(['Total'])
    ax.legend([plt.Line2D((0, 1), (0, 0), color='C' + str(i), marker='o', linestyle='') for i in range(len(dirs))],
              dirs)
    plt.show()


def get_bandwidths(base_dir):
    bandwidths = {}
    base_dir = os.path.join(base_dir, 'bandwidths')
    max_length = 0
    for file in os.listdir(base_dir):
        data = json.loads(open(os.path.join(base_dir, file)).read())
        bandwidth = []
        timestamps = []
        for limit in data['bandwidth_limits']:
            bandwidth.append(limit['mbps'])
            timestamps.append(limit['time'] / 60 - 1) # convert to minutes and remove system startup time
        bandwidths[file] = (bandwidth, timestamps)
        if max_length < len(bandwidth):
            max_length = len(bandwidth)
    bandwidth = []
    timestamps = []
    # get average bandwidth over all files at each timestamp
    for i in range(max_length):
        avg, count, timestamp = 0, 0, 0
        for key in bandwidths:
            try:
                if count == 0:
                    timestamp = bandwidths[key][1][i]
                avg += bandwidths[key][0][i]
                count += 1
            except IndexError:
                pass
        bandwidth.append(round(avg / count, 2))
        timestamps.append(timestamp)

    bandwidths['overall'] = (bandwidth, timestamps)
    return bandwidths


def logSystemMetrics(base_directory):
    results = {}
    experiments = ['fcty', 'camp']
    for exp in experiments:
        directory = os.path.join(base_directory, exp)
        if not os.path.exists(os.path.join(directory, 'processed_logs.pkl')):
            systems = natsorted(os.listdir(directory))
            if exp == 'fcty':
                results[exp] = analyze_single_experiment(directory, systems, 1, 200, True, 2160)
            elif exp == 'camp':
                results[exp] = analyze_single_experiment(directory, systems, 1, 250, True, 7200)
            else:
                results[exp] = analyze_single_experiment(directory, systems, 1, 250, True)
            with open(os.path.join(directory, 'processed_logs.pkl'), 'wb') as f:
                pickle.dump(results[exp], f)
        else:
            with open(os.path.join(directory, 'processed_logs.pkl'), 'rb') as f:
                results[exp] = pickle.load(f)

    with open('systemMetrics.txt', 'wb') as f:
        for exp in experiments:
            f.write(f"Experiment: {exp}\n".encode())
            for key, value in results[exp].items():
                f.write(f"{key}: {value}\n".encode())
            f.write("\n".encode())