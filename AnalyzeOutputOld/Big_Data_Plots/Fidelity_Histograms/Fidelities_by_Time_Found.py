'''
5/3/22
Owen Eskandari
Thesis work 22S

This file is to make a plot where the x axis is the time in the run when the sequence was found and the y axis is
the reward of the sequence (no errors)

Probably an easier way to make this plot but I'm basing it off of Gradient_Offset_Error.py
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

off_vals = [-1000, -100, -10, -1, 0, 1, 10, 100, 1000]
val = 4

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
                    colors.append(index / count)
                run = id
                count = 0

        if 'Y' in ps_obj.get_hists():
            no_err_reward = ps_obj.get_no_error_fid()
            if no_err_reward == 200:        # For now
                no_err_reward = 10
                # print(ps_obj.get_ps())
            offs = ps_obj.get_offset_fids()
            for j, off in enumerate(offs):
                if off == 200:
                    offs[j] = 10
            x.append(no_err_reward)
            y.append(offs[val])

            # colors.append(int(ps_obj.get_orig_name().split('_')[3]))
            count += 1

name = 'Sequence Rewards'                                          # TODO
path = '/Users/oweneskandari/PycharmProjects/21X/rl_pulse/22S/Thesis_Plots_All/Gradient/'
additional = 'All Runs Histogram No Error'

plt.scatter(colors, y, c=y, alpha=0.3, cmap='viridis_r')        # Greys; Meme: prism
plt.title(name)
plt.xlabel('Sequence number (proportion of total sequences found in its run)', fontsize=12)
plt.ylabel('Reward (no error)', fontsize=12)    # TODO
plt.xlim(-0.1, 1.1)
plt.ylim(5.2, 10.8)

plt.show()
# plt.savefig(path + name + ' ' + additional + '.png', dpi=500)                             # TODO
plt.clf()
