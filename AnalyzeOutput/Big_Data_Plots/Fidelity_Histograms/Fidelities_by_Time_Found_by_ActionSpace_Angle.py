'''
5/9/22
Owen Eskandari
Thesis work 22S

This file is to make a plot where the x axis is the sequence number and the y axis is
the reward of the sequence (rotation errors)

Probably an easier way to make this plot but I'm basing it off of Gradient_Offset_Error.py
'''


import pickle
from PulseSequences import PulseSequence as PS
from PulseSequences import PulseSequences as PSS
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as cs

# Open file and get PSS object

f = '/Users/oweneskandari/Desktop/PSS_with_Fids_20220502_123919.pkl'                        # TODO: Change
with open(f, 'rb') as f:
    pss = pickle.load(f)

ps_dict = pss.get_ps_dict()
print(len(ps_dict))

maximum = 722320        # Max count

# runname = 1260860       # TODO: Change
runnames = [1146029, 1196362, 1196364, 1252201, 1253817, 1260860, 1750485, 1750487, 1258278, 1259819, 1260857, 1260859,
            1260856, 1291543, 1390117, 1678985, 1457761, 1750484, 2032111, 2317153, 2403821]

actionspace = 'SED'      # TODO: Change
length = 36             # TODO: Change


angle_vals = [0, 0.025, 0.05, 0.075, 0.1]
ylims = [0, 6, 4.7, 4.2, 4.2]
for val in range(1, 5):
    print(val)

    for runname in runnames:
        colors = []

        x, y = [], []

        run = ''          # Normalizing
        count = 0
        counts = []      # Number of times each sequence was found

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

                if actionspace in ps_obj.get_action_spaces() and length == ps_obj.get_length():   # TODO: action space specific

                    if runname in ps_obj.get_jobids():  # Only look at a specific run

                        errs = ps_obj.get_rot_error_fids()
                        for j, off in enumerate(errs):
                            if off == 200:
                                errs[j] = 10

                        y.append(errs[val])
                        counts.append(ps_obj.get_orig_count())

                        x.append(int(ps_obj.get_orig_name().split('_')[3]))
                        count += 1

        new_counts = [0.4*(np.log(i))**3 + 3 for i in counts]
        lines = [i**(1/3)/10 for i in new_counts]

        if len(y) > 0:
            print(runname)
            # Count represented by color
            # plt.scatter(colors, y, c=counts, alpha=0.3, cmap='viridis_r', norm=cs.LogNorm(vmin=1, vmax=maximum))

            # Count represented by size and color
            plt.scatter(x, y, alpha=0.5, s=new_counts, edgecolors='black', linewidth=lines,
                        c=counts, cmap='viridis', norm=cs.LogNorm(vmin=1, vmax=maximum))

    name = 'Rewards Over Time for ' + str(length) + r'$\tau$ ' + actionspace + ' Sequences'                # TODO

    path = '/Users/oweneskandari/PycharmProjects/21X/rl_pulse/22S/Thesis_Plots_All/Time_Spent/Rot/'      # TODO
    additional = 'Angle Error = ' + str(angle_vals[val])
    plt.colorbar()
    plt.title(name)
    plt.xlabel('Sequence number', fontsize=12)
    plt.ylabel('Reward (rotation error = ' + str(angle_vals[val]) + ')', fontsize=12)    # TODO
    # plt.xlim(-0.1, 1.1)
    plt.ylim(-ylims[val]/20, ylims[val])

    # plt.show()
    plt.savefig(path + actionspace + str(length) + ' ' + additional + '.png', dpi=500)                             # TODO
    plt.clf()


# Repeat these plots for errors!
