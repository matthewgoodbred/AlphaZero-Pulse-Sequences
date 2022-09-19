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

runname = 1260860       # TODO: Change

angle_vals = [0, 0.025, 0.05, 0.075, 0.1]
for val in range(1, 5):

    colors = []

    no_error = []
    pte_error = []

    x, y = [], []

    run = ''          # Normalizing
    count = 0

    for sequence in ps_dict:
        for idx, ps_obj in enumerate(ps_dict[sequence]):

            # Normalizing the count per run (so that the
            id = ps_obj.get_orig_jobid()

            if run != id and run is not None:      # Does this deal with non histogram stuff? TODO
                # Initial condition
                if count == 0:
                    run = id
                else:
                    for index in range(count):
                        colors.append(100 * index / count)
                    run = id
                    count = 0

            if runname in ps_obj.get_jobids():        # Only look at a specific run
                no_err_reward = ps_obj.get_no_error_fid()
                if no_err_reward == 200:        # For now
                    no_err_reward = 10
                    # print(ps_obj.get_ps())
                aerrs = ps_obj.get_rot_error_fids()
                if aerrs[0] == 200:
                    aerrs[0] = 10
                x.append(no_err_reward)
                y.append(aerrs[val])

                # colors.append(int(ps_obj.get_orig_name().split('_')[3]))
                count += 1

    name = 'Rewards Over Time for Run ' + str(runname)                                          # TODO
    path = '/Users/oweneskandari/PycharmProjects/21X/rl_pulse/22S/Thesis_Plots_All/Gradient/'
    additional = str(runname) + ' Angle= ' + str(angle_vals[val])

    vals = np.linspace(-0.5, 10, 1000)       # cutoff --> -0.5 for when the cutoff is -1
    plt.plot(vals, vals)

    plt.scatter(x, y, alpha=0.3, label='', c=colors, cmap='viridis')        # Greys; Meme: prism
    plt.colorbar()
    # plt.legend()
    plt.title(name)
    plt.xlabel('Reward (no error)', fontsize=12)
    plt.ylabel('Reward (Fractional Angle Error= ' + str(angle_vals[val]) + ')', fontsize=12)    # TODO
    # plt.xlim(-0.5, 5)
    # plt.ylim(-0.5, 5)
    plt.xlim(-1, 10.5)
    plt.show()
    # plt.savefig(path + name + ' ' + additional + '.png', dpi=500)                             # TODO
    plt.clf()
