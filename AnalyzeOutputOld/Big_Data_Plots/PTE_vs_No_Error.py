'''
5/2/22
Owen Eskandari
Thesis Work 22S

Phase transient error versus no error
='''

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

for sequence in ps_dict:
    for idx, ps_obj in enumerate(ps_dict[sequence]):
        no_err_reward = ps_obj.get_no_error_fid()
        if no_err_reward == 200:        # For now
            no_err_reward = 10
            # print(ps_obj.get_ps())
        ptes = ps_obj.get_pte_fids()

        spaces = ps_obj.get_action_spaces()

        # Color code by action space
        if 'O' in spaces:
            ox.append(no_err_reward)
            oy.append(ptes[2])
        if 'SE' in spaces:
            sex.append(no_err_reward)
            sey.append(ptes[2])
        if 'SED' in spaces:
            sedx.append(no_err_reward)
            sedy.append(ptes[2])
        if 'SEDD' in spaces:
            seddx.append(no_err_reward)
            seddy.append(ptes[2])
        if 'B' in spaces:
            bx.append(no_err_reward)
            by.append(ptes[2])
        if '+' in ps_obj.get_orig_name():
            compx.append(no_err_reward)
            compy.append(ptes[2])
        if '2*l24+2x seae24f' in ps_obj.get_orig_name():
            print(no_err_reward, ptes[2])
            sed48x.append(no_err_reward)
            sed48y.append(ptes[2])
        if 'CORY48' in ps_obj.get_orig_name():
            print(no_err_reward, ptes[2])
            cory48x.append(no_err_reward)
            cory48y.append(ptes[2])
        if '1259819_72_SED_15144' in ps_obj.get_orig_name():
            print(no_err_reward, ptes[2])
            sed48ptex.append(no_err_reward)
            sed48ptey.append(ptes[2])


name = 'All Action Spaces: Fractional Phase Transient Error = 0.001'                                          # TODO
path = '/Users/oweneskandari/PycharmProjects/21X/rl_pulse/22S/Thesis_Plots_All/'
additional = 'All Cory SED pte'

vals = np.linspace(-0.5, 6, 1000)       # cutoff --> -0.5 for when the cutoff is -1
plt.plot(vals, vals)
plt.scatter(ox, oy, alpha=0.1, label='O')
plt.scatter(sex, sey, alpha=0.1, label='SE')
# plt.scatter(sedx, sedy, alpha=0.1, label='AlphaZero')
plt.scatter(sedx, sedy, alpha=0.1, label='SED')     # TODO: Same as above
plt.scatter(seddx, seddy, alpha=0.1, label='SEDD')
# plt.scatter(bx, by, alpha=0.1, label='B')
# plt.scatter(compx, compy, alpha=0.5, label='Computationally Combined')
plt.scatter(compx, compy, alpha=0.5, label='Comp')      # TODO: Same as above

plt.scatter(sed48x, sed48y, alpha=1, label='SED48', marker=(5, 1), c='crimson', s=200)
plt.scatter(cory48x, cory48y, alpha=1, label='CORY48', marker=(5, 1), c='b', s=200)
plt.scatter(sed48ptex, sed48ptey, alpha=1, label='SED48_pte', marker=(5, 1), c='black', s=200)

plt.legend()
plt.title(name)
plt.xlabel('Reward (no error)', fontsize=12)
plt.ylabel('Reward (Fractional phase transient error relative to ' + r'$\pi$' + '/2 pulse)', fontsize=12)    # TODO
plt.xlim(-1, 10.5)
# plt.show()
plt.savefig(path + name + ' ' + additional + '.png', dpi=500)                             # TODO
