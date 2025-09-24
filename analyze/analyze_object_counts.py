import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_data(file, bucket_size = 120, length = 15.0):
    data = pd.read_csv(file)

    frame_index = data['Frame_Index'].values
    persons = data['person'].values
    cars = data['car'].values

    num_buckets = len(frame_index) // bucket_size
    buckets = np.arange(num_buckets)
    combined = []

    for i in buckets:
        start = i * bucket_size
        end = start + bucket_size

        # Extract the data for the current bucket and calculate the average
        if 'traffic' in file:
            for j in range(start, end): # because we only expect half of the persons to face the camera
                persons[j] = persons[j] / 2
        combined.append(np.sum(persons[start:end] + cars[start:end]) / (bucket_size / 30))

    if length > 0.0:
        buckets = [float(i) / length for i in buckets]
    return buckets[:len(combined)], combined


def moving_average_sub(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def moving_average(a, n=3):
    ret = moving_average_sub(a, n)
    for i in range(n - 1):
        # do not just repeat static value, but calculate the average of the last n - i values
        x = 0
        for j in range(10):
            x += ret[-j-1]
        ret = np.append(ret, (moving_average_sub(a[-(n - i):], n - i) + x) / 11)
    return ret

def read_csv_files(args, files, bucket_size=250, plot=False):
    total = {}
    for file in files:
        data = pd.read_csv(str(os.path.join(args.directory, file)))

        frame_index = data['Frame_Index'].values
        persons = data['person'].values
        cars = data['car'].values

        num_buckets = len(frame_index) // bucket_size
        buckets = np.arange(num_buckets)
        avg_persons, min_persons, max_persons = [], [], []
        avg_cars, min_cars, max_cars = [], [], []

        for i in buckets:
            start = i * bucket_size
            end = start + bucket_size
            # Extract the data for the current bucket and calculate the average
            avg_persons.append(np.mean(persons[start:end]))
            min_persons.append(np.min(persons[start:end]))
            max_persons.append(np.max(persons[start:end]))
            avg_cars.append(np.mean(cars[start:end]))
            min_cars.append(np.min(cars[start:end]))
            max_cars.append(np.max(cars[start:end]))

        # Apply moving average
        avg_persons_smooth = moving_average(np.array(avg_persons), n=250)
        avg_cars_smooth = moving_average(np.array(avg_cars), n=250)
        total[file] = [avg_persons_smooth, avg_cars_smooth, min_persons, max_persons, min_cars, max_cars, buckets]

        if plot:
            plt.figure(num=file, figsize=(10, 5))
            plt.plot(buckets[:len(avg_persons_smooth)], avg_persons_smooth, label='Person', color='orange', linewidth=4)
            plt.fill_between(buckets, min_persons, max_persons, color='orange', alpha=0.2)
            plt.plot(buckets[:len(avg_cars_smooth)], avg_cars_smooth, label='Car', color='blue', linewidth=4)
            plt.fill_between(buckets, min_cars, max_cars, color='blue', alpha=0.2)
            if file != 'traffic2.csv':
                plt.ylabel('Number of Objects', fontsize=25)
            plt.grid(True)
            plt.tick_params(axis='both', which='major', labelsize=25)
            x = len(buckets) / 4
            plt.xticks([0, x, 2*x, 3*x, 4*x], ['9:00', '12:25', '15:30', '18:45', '22:00'])

            filename_wo_extension = os.path.splitext(file)[0]
            parent_directory = os.path.dirname(args.directory)
            plt.savefig(os.path.join(parent_directory, filename_wo_extension) + '_' + bucket_size + '.pdf', format='pdf')

    return total

def objectcount(args, files):
    total = read_csv_files(args, files)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    buckets = np.arange(len(total['traffic1.csv'][0]))
    ax.plot(buckets, total['traffic1.csv'][1], label='§1-Vehicle', color='blue', linewidth=3)
    ax.plot(buckets, total['traffic1.csv'][0], label='§1-Person', color='blue', linewidth=3, linestyle='dotted')
    ax.plot(buckets, total['traffic2.csv'][1], label='§2-Vehicle', color='green', linewidth=3)
    ax.plot(buckets, total['traffic2.csv'][0], label='§2-Person', color='green', linewidth=3, linestyle='dotted')
    plt.ylabel('Object Count    ', fontsize=30)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=25)
    x = len(buckets) / 4
    plt.xticks([0, x, 2 * x, 3 * x, 4 * x], ['9:00', '12:25', '15:30', '18:45', '22:00'])
    plt.xlabel('Local Time of Footage', fontsize=30)
    plt.legend(loc='lower right', fontsize=25)
    plt.tight_layout()
    plt.savefig('content_dynamics.pdf', format='pdf')

    total = read_csv_files(args, files, 50)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    buckets = total['traffic1.csv'][6]
    ax.plot(buckets[:len(total['traffic1.csv'][0])], total['traffic1.csv'][0], label='§1-Person', color='orange',
            linewidth=3, linestyle='dotted')
    ax.fill_between(buckets, total['traffic1.csv'][2], total['traffic1.csv'][3], color='orange', alpha=0.25,
                    linestyle='dotted')
    ax.plot(buckets[:len(total['traffic1.csv'][1])], total['traffic1.csv'][1], label='§1-Vehicle', color='blue',
            linewidth=3)
    ax.fill_between(buckets, total['traffic1.csv'][4], total['traffic1.csv'][5], color='blue', alpha=0.25)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=25)
    x = len(buckets) / 26
    plt.xticks([3 * x, x * 3.33, x * 3.66, 4 * x], ['10:30', '10:40', '10:50', '11:00'])
    plt.xlim(3 * x - 100, 4 * x + 100)
    plt.ylim(0, 12)
    plt.xlabel('Local Time of Footage', fontsize=30)
    plt.ylabel('Object Count    ', fontsize=30)
    plt.legend(fontsize=25, loc='upper left')
    plt.tight_layout()
    plt.savefig('content_dynamics_detail.pdf', format='pdf')