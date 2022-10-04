'''
5/3/22
Owen Eskandari
Thesis Work 22S

This is to make some plots of rewards versus phase transient error.
-By pbc
    -Specifically, for the 2,3,4,5 pbc runs for SE in the fall
'''

import pickle
from PulseSequences import PulseSequence as PS
from PulseSequences import PulseSequences as PSS
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Open file and get PSS object

f = '/Users/oweneskandari/Desktop/PBC_Plot_PSS_with_Fids_20220503_135722.pkl'                        # TODO: Change
with open(f, 'rb') as f:
    pss = pickle.load(f)

ps_dict = pss.get_ps_dict()
print(len(ps_dict))

shapes = ['X', 'o', 'v', '^']

pte_vals = [0, 0.0001, 0.001, 0.01, 0.02]
for val in range(5):

    pbc2x, pbc2y = [], []
    pbc3x, pbc3y = [], []
    pbc4x, pbc4y = [], []
    pbc5x, pbc5y = [], []

    for sequence in ps_dict:
        for idx, ps_obj in enumerate(ps_dict[sequence]):
            no_err_reward = ps_obj.get_no_error_fid()
            if no_err_reward == 200:        # For now
                no_err_reward = 10
                # print(ps_obj.get_ps())
            ptes = ps_obj.get_pte_fids()

            pbc = ps_obj.get_orig_pbc()

            spaces = ps_obj.get_action_spaces()

            # Color code by pbc value
            if pbc == 2.0:
                pbc2x.append(no_err_reward)
                pbc2y.append(ptes[val])
            if pbc == 3.0:
                pbc3x.append(no_err_reward)
                pbc3y.append(ptes[val])
            if pbc == 4.0:
                pbc4x.append(no_err_reward)
                pbc4y.append(ptes[val])
            if pbc == 5.0:
                pbc5x.append(no_err_reward)
                pbc5y.append(ptes[val])


    name = 'Rewards by Initial Exploration Value (SE Action Space)'                                          # TODO
    path = '/Users/oweneskandari/PycharmProjects/21X/rl_pulse/22S/Thesis_Plots_All/PBC/To_Use/'
    additional = 'PBC PTE = ' + str(pte_vals[val])

    vals = np.linspace(-0.5, 6, 1000)       # cutoff --> -0.5 for when the cutoff is -1
    plt.style.use('tableau-colorblind10')
    plt.plot(vals, vals)

    # plt.scatter(pbc5x, pbc5y, alpha=0.3, label='5')
    # plt.scatter(pbc4x, pbc4y, alpha=0.3, label='4')
    # plt.scatter(pbc3x, pbc3y, alpha=0.3, label='3')
    # plt.scatter(pbc2x, pbc2y, alpha=0.3, label='2')

    plt.scatter(pbc2x, pbc2y, alpha=0.3, label='2', marker=shapes[0])
    plt.scatter(pbc3x, pbc3y, alpha=0.8, label='3', marker=shapes[1])
    plt.scatter(pbc4x, pbc4y, alpha=0.3, label='4', marker=shapes[2])
    plt.scatter(pbc5x, pbc5y, alpha=0.0, marker=shapes[3])
    plt.scatter(pbc5x, pbc5y, alpha=0.3, label='5', marker=shapes[3])



    # plt.scatter(sed48x, sed48y, alpha=1, label='SED48', marker=(5, 1), c='crimson', s=200)
    # plt.scatter(cory48x, cory48y, alpha=1, label='CORY48', marker=(5, 1), c='b', s=200)
    # plt.scatter(sed48ptex, sed48ptey, alpha=1, label='SED48_pte', marker=(5, 1), c='black', s=200)

    plt.legend()
    plt.title(name)
    plt.xlabel('Reward (no error)', fontsize=12)
    plt.ylabel('Reward (fractional phase transient error = ' + str(pte_vals[val]) + ')', fontsize=12)    # TODO
    # plt.xlim(2, 4.5)
    plt.xlim(-0.5, 5)
    plt.ylim(-0.5, 5)
    # plt.show()
    plt.savefig(path + 'CB_' + name + ' ' + additional + '.png', dpi=500)                             # TODO
    plt.clf()
