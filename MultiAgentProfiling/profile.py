# Library Imports
import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd


# Import Data


def open_log(experiment, agentnumber):
    with open(os.getcwd() + '/MultiAgentProfiling/data/' + experiment + '/agent' + str(agentnumber) + '_meanlog',
              'rb') as handle:
        _meanlog = pickle.load(handle)
        return _meanlog


def plot_and_save(experiments):
    for experiment in experiments:
        # open mean logs of all three agents
        _meanlog1 = open_log(experiment, 1)
        _meanlog2 = open_log(experiment, 2)
        _meanlog3 = open_log(experiment, 3)
        # merge logs of all three agents to one by taking for each index the average
        df = pd.DataFrame([_meanlog1, _meanlog2, _meanlog3])
        _meanlog_merged = dict(df.mean())

        # Plot and Save the Graphs
        mean_lists = sorted(_meanlog_merged.items())  # sorted by key, return a list of tuples

        mean_x, mean_y = zip(*mean_lists)  # unpack a list of pairs into two tuples

        plt.figure(1)
        plt.plot(mean_x, mean_y, label='Mean Rewards: ' + str(experiment))
    plt.grid(True)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('agents merged Training Profile')
    plt.legend(loc='best')
    plt.savefig(os.getcwd() + '/MultiAgentProfiling/data/agents_merged Training Profile.png')


experiment1 = "300_standard"
experiment2 = "300_steady_inject_rate_0.1"
experiment3 = "300_initially_high_inject_rate_0.5"

plot_and_save([experiment1, experiment2, experiment3])
