'''
5/3/22
Owen Eskandari
Thesis work 22S

This file is to make a plot but where the color of the points depends on when they were found
'''


import pickle
from PulseSequences import PulseSequence as PS
from PulseSequences import PulseSequences as PSS
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Open file and get PSS object

f = '/Users/oweneskandari/Desktop/PSS_with_Fids_20220502_123919.pkl'                        # TODO: Change
with open(f, 'rb') as f:
    pss = pickle.load(f)

ps_dict = pss.get_ps_dict()
print(len(ps_dict))

pte_vals = [0, 0.0001, 0.001, 0.01, 0.02]
for val in range(1, 5):

    colors = []

    x, y = [], []

    run = ''          # Normalizing
    count = 0

    for sequence in ps_dict:
        for idx, ps_obj in enumerate(ps_dict[sequence]):

            # Normalizing the count per run
            id = ps_obj.get_orig_jobid()

            if run != id and run is not None:      # Doesn't consider non histogram stuff
                # Initial condition
                if count == 0:
                    run = id
                else:
                    for index in range(count):
                        colors.append(100 * index / count)
                    run = id
                    count = 0

            if 'Y' in ps_obj.get_hists():
                no_err_reward = ps_obj.get_no_error_fid()
                if no_err_reward == 200:        # For now
                    no_err_reward = 10
                    # print(ps_obj.get_ps())
                ptes = ps_obj.get_pte_fids()
                if ptes[0] == 200:
                    ptes[0] = 10
                x.append(no_err_reward)
                y.append(ptes[val])

                # colors.append(int(ps_obj.get_orig_name().split('_')[3]))
                count += 1

    name = 'Rewards Over Time'                                          # TODO
    path = '/Users/oweneskandari/PycharmProjects/21X/rl_pulse/22S/Thesis_Plots_All/Gradient/'
    additional = 'All Runs PTE= ' + str(pte_vals[val])

    vals = np.linspace(-0.5, 10, 1000)       # cutoff --> -0.5 for when the cutoff is -1
    plt.plot(vals, vals)

    plt.scatter(x, y, alpha=0.3, label='', c=colors, cmap='viridis')        # Greys; Meme: prism
    plt.colorbar()
    # plt.legend()
    plt.title(name)
    plt.xlabel('Reward (no error)', fontsize=12)
    plt.ylabel('Reward (Fractional Phase Transient Error= ' + str(pte_vals[val]) + ')', fontsize=12)    # TODO
    # plt.xlim(-0.5, 5)
    # plt.ylim(-0.5, 5)
    plt.xlim(-1, 10.5)
    # plt.show()
    plt.savefig(path + name + ' ' + additional + '.png', dpi=500)                             # TODO
    plt.clf()
