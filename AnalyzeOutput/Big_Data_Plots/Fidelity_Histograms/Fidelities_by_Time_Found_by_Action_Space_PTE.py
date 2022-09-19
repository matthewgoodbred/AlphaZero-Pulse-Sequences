'''
5/3/22
Owen Eskandari
Thesis work 22S

This file is to make a plot where the x axis is the time in the run when the sequence was found and the y axis is
the reward of the sequence (no errors)

Probably an easier way to make this plot but I'm basing it off of Gradient_Offset_Error.py

Color code by action space
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

pte_vals = [0.0001, 0.001, 0.01, 0.02]
ylims = [11, 11, 9, 6]
for val in range(4):

    colors = []

    x, y = [], []

    no_error = []
    pte_error = []

    ox, oy = [], []
    sex, sey = [], []
    sedx, sedy = [], []
    seddx, seddy = [], []
    bx, by = [], []

    compx, compy = [], []

    sed48x, sed48y = [], []
    cory48x, cory48y = [], []
    sed48ptex, sed48ptey = [], []

    run = ''  # Normalizing
    count = 0
    tot = 0

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

                ptes = ps_obj.get_pte_fids()

                for j, off in enumerate(ptes):
                    if off == 200:
                        ptes[j] = 10

                spaces = ps_obj.get_action_spaces()
                # Color code by action space
                if 'O' in spaces:
                    ox.append(tot)
                    oy.append(ptes[val])
                if 'SE' in spaces:
                    sex.append(tot)
                    sey.append(ptes[val])
                if 'SED' in spaces:
                    sedx.append(tot)
                    sedy.append(ptes[val])
                if 'SEDD' in spaces:
                    seddx.append(tot)
                    seddy.append(ptes[val])
                if 'B' in spaces:
                    bx.append(tot)
                    by.append(ptes[val])
                if '+' in ps_obj.get_orig_name():
                    compx.append(tot)
                    compy.append(ptes[val])
                if '2*l24+2x seae24f' in ps_obj.get_orig_name():
                    # print(tot, ptes[val])
                    sed48x.append(tot)
                    sed48y.append(ptes[val])
                if 'CORY48' in ps_obj.get_orig_name():
                    # print(tot, ptes[val])
                    cory48x.append(tot)
                    cory48y.append(ptes[val])
                if '1259819_72_SED_15144' in ps_obj.get_orig_name():
                    # print(tot, ptes[val])
                    sed48ptex.append(tot)
                    sed48ptey.append(ptes[val])

                # colors.append(int(ps_obj.get_orig_name().split('_')[3]))
                count += 1
                tot += 1

    # Get the correct x lists
    oxx, sexx, sedxx, seddxx, sed48ptexx = [], [], [], [], []
    for i in ox:
        oxx.append(colors[i])
    for i in sex:
        sexx.append(colors[i])
    for i in sedx:
        sedxx.append(colors[i])
    for i in seddx:
        seddxx.append(colors[i])
    for i in sed48ptex:
        sed48ptexx.append(colors[i])

    for actionspace in ['Original', 'SE', 'SED', 'SEDD']:

        name = actionspace + ' Sequence Rewards'                                          # TODO
        path = '/Users/oweneskandari/PycharmProjects/21X/rl_pulse/22S/Thesis_Plots_All/Histograms/'
        additional = actionspace + ' Runs Histogram PTE= ' + str(pte_vals[val])

        if actionspace == 'Original':
            plt.scatter(oxx, oy, alpha=0.3, label='O')
        if actionspace == 'SE':
            plt.scatter(sexx, sey, alpha=0.3, label='SE')
        if actionspace == 'SED':
            plt.scatter(sedxx, sedy, alpha=0.3, label='SED')
            plt.scatter(sed48ptexx, sed48ptey, alpha=1, label='SED48_pte', marker=(5, 1), c='black', s=200)
        if actionspace == 'SEDD':
            plt.scatter(seddxx, seddy, alpha=0.3, label='SEDD')

        # plt.legend()
        plt.title(name)
        plt.xlabel('Sequence number (proportion of total sequences found in its run)', fontsize=12)

        plt.ylabel('Reward (Fractional Phase Transient Error = ' + str(pte_vals[val]) + ')', fontsize=12)    # TODO
        plt.xlim(-0.1, 1.1)
        # plt.ylim(-1, 11)
        plt.ylim(-ylims[val]/20, ylims[val])

        # plt.show()
        plt.savefig(path + name + ' ' + additional + '.png', dpi=500)                             # TODO
        plt.clf()



