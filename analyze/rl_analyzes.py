import os
import pickle

import matplotlib.pyplot as plt
from natsort import natsorted
from final_figures import (colors, long_colors, styles)

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
        cheis = os.path.join(base_directory, 'cheis_logs')
        cheis_rewards = {}
        cheis_loss = {}
        read_rl_logs(cheis, natsorted(os.listdir(cheis)), cheis_rewards, cheis_loss)

        cheis_reduced = os.path.join(base_directory, 'cheis_reduced_logs')
        cheisr_rewards = {}
        cheisr_loss = {}
        read_rl_logs(cheis_reduced, natsorted(os.listdir(cheis_reduced)), cheisr_rewards, cheisr_loss)

        cheis_global = os.path.join(base_directory, 'cheis_global_logs')
        cheisg_rewards = {}
        cheisg_loss = {}
        read_rl_logs(cheis_global, natsorted(os.listdir(cheis_global)), cheisg_rewards, cheisg_loss)

        cheis_fed = os.path.join(base_directory, 'federated_experiment')
        cheisf_rewards = {}
        cheisf_loss = {}
        read_rl_logs(cheis_fed, natsorted(os.listdir(cheis_fed)), cheisf_rewards, cheisf_loss)

        cheis_w_tutti = os.path.join(base_directory, 'cheis_logs_with_tutti')
        cheiswt_rewards = {}
        cheiswt_loss = {}
        read_rl_logs(cheis_w_tutti, natsorted(os.listdir(cheis_w_tutti)), cheiswt_rewards, cheiswt_loss)

        bce = os.path.join(base_directory, 'bce_logs')
        bce_rewards = {}
        bce_loss = {}
        read_rl_logs(bce, natsorted(os.listdir(bce)), bce_rewards, bce_loss)

        bce_w_tutti = os.path.join(base_directory, 'bce_logs_with_tutti')
        bcewt_rewards = {}
        bcewt_loss = {}
        read_rl_logs(bce_w_tutti, natsorted(os.listdir(bce_w_tutti)), bcewt_rewards, bcewt_loss)

        rewards = {}
        loss = {}
        for algo, data in {'CHEIS': [cheis_rewards, cheis_loss], 'BCE': [bce_rewards, bce_loss],
                           'FCPO-tutti': [cheiswt_rewards, cheiswt_loss], 'BCE-tutti': [bcewt_rewards, bcewt_loss],
                           'CHEIS-reduced': [cheisr_rewards, cheisr_loss], 'without local optimization': [cheisg_rewards, cheisg_loss]}.items():
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
                pickle.dump([rewards, loss, cheis_rewards, cheis_loss, bce_rewards, bce_loss, cheisf_rewards, cheisf_loss, cheisr_rewards, cheisr_loss, cheisg_rewards, cheisg_loss], f)
    else:
        with open(os.path.join(base_directory, 'processed_logs', 'reward-loss.pkl'), 'rb') as f:
            rewards, loss, cheis_rewards, cheis_loss, bce_rewards, bce_loss, cheisf_rewards, cheisf_loss, cheisr_rewards, cheisr_loss, cheisg_rewards, cheisg_loss = pickle.load(f)

    # plot the results
    learning_algos=['CHEIS', 'BCE', 'CHEIS-reduced', 'FCPO-tutti', 'BCE-tutti', 'without local optimization']
    for j, algo in enumerate(learning_algos):
        if j % 3 == 1:
            fig1, ax1 = plt.subplots(1, 1, figsize=(2, 1.3), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        elif j % 3 == 0:
            fig1, ax1 = plt.subplots(1, 1, figsize=(2.2, 1.3), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        else:
            fig1, ax1 = plt.subplots(1, 1, figsize=(2.2, 1.3), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        ax2 = ax1.twinx()

        ax1.plot(rewards[algo], label='reward', color=colors[0], linewidth=1)
        ax2.plot(loss[algo], label='loss', color=colors[0], linestyle=(0, (8, 8)), linewidth=1)

        ax1.set_xlim(0, 80)
        ax1.set_xticks([0, 80])
        ax1.set_xticklabels([0, 80], size=13)

        if j % 3 == 0:
            ax1.set_ylabel(r'Reward  ', size=13)
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels([0, 1], size=13)
        else:
            ax1.set_yticks([])

        if j % 3 == 2:
            ax2.set_ylabel(r'Loss', size=13)
            ax2.set_yticks([0, max(loss[algo])])
            ax2.set_yticklabels([0, 1], size=13)
        else:
            ax2.set_yticks([])

        ax1.set_ylim([0, 1])
        ax2.set_ylim([0, max(loss[algo])])

        fig1.tight_layout()
        fig1.savefig(f"{algo}-learning.pdf")
        fig1.show()
    exit()

    fig1, ax1 = plt.subplots(1, 1, figsize=(2, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    fig2, ax2 = plt.subplots(1, 1, figsize=(2, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    for j, algo in enumerate(['CHEIS', 'CHEIS-reduced', 'without local optimization']):
        ax1.plot(rewards[algo], styles[j], label=algo, color=colors[j], linewidth=1, markevery=0.2)
        ax2.plot(loss[algo], styles[j], label=algo, color=colors[j], linewidth=1, markevery=0.2)
    ax1.set_xlabel('Episodes', size=12)
    ax1.set_ylabel('Reward Avg.', size=12)
    ax2.set_xlabel('Episodes', size=12)
    ax2.set_ylabel('Loss Avg.', size=12)
    ax1.set_xlim(0, 50)
    ax2.set_xlim(0, 50)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    fig1.tight_layout()
    fig1.savefig("ablation-rewards.pdf")
    fig1.show()
    fig2.tight_layout()
    fig2.savefig("ablation-loss.pdf")
    fig2.show()

    for algo, data in {'CHEIS-rewards': cheis_rewards, 'BCE-rewards': bce_rewards, 'CHEIS-loss': cheis_loss,
                       'BCE-loss': bce_loss, 'federated-rewards': cheisf_rewards, 'federated-loss': cheisf_loss}.items():
        if 'federated' in algo:
            fig, ax1 = plt.subplots(1, 1, figsize=(4, 2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(4, 4), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
        i = 0
        for agent, series in data.items():
            if 'fed' in algo:
                ax1.plot(series, label=agent, color=colors[i])
            else:
                ax1.plot(series, label=agent, color=long_colors[i])
            i += 1
        ax1.legend(fontsize=14, loc='center right')
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        if 'reward' in algo:
            ax1.set_ylabel('Avg. Reward   ', size=14)
        else:
            ax1.set_ylabel('Avg. Loss', size=14)
        if 'federated' in algo:
            ax1.set_xlim(0, 40)
            ax1.set_xticks([0, 15, 30])
            ax1.set_xticklabels([0, 15, '30 Episodes'], size=14)
        else:
            ax1.set_xlabel('Episodes', size=12)
            ax1.set_xlim(0, 100)
        plt.tight_layout()
        plt.savefig(f"{algo}.pdf")
        plt.show()
