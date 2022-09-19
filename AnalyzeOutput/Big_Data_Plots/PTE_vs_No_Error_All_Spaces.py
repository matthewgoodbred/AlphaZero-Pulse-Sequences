'''
5/2/22
Owen Eskandari
Thesis Work 22S

Phase transient error versus no error
-Color coded by action space
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

no_error = []
pte_error = []

ox, oy = [], []
sex, sey = [], []
sedx, sedy = [], []
seddx, seddy = [], []
bx, by = [], []

for sequence in ps_dict:
    for idx, ps_obj in enumerate(ps_dict[sequence]):
        if ps_obj.get_no_error_fid() != 200:        # For now
            no_err_reward = ps_obj.get_no_error_fid()
            ptes = ps_obj.get_pte_fids()

            spaces = ps_obj.get_action_spaces()

            # Color code by action space
            if 'O' in spaces:
                ox.append(no_err_reward)
                oy.append(ptes[3])
            if 'SE' in spaces:
                sex.append(no_err_reward)
                sey.append(ptes[3])
            if 'SED' in spaces:
                sedx.append(no_err_reward)
                sedy.append(ptes[3])
            if 'SEDD' in spaces:
                seddx.append(no_err_reward)
                seddy.append(ptes[3])
            if 'B' in spaces:
                bx.append(no_err_reward)
                by.append(ptes[3])

name = 'Fractional Phase Transient Error = 0.01'                                          # TODO
path = '/Users/oweneskandari/PycharmProjects/21X/rl_pulse/22S/Thesis_Plots_All/'
additional = 'By action space'

vals = np.linspace(-0.5, 6, 1000)       # cutoff --> -0.5 for when the cutoff is -1
plt.plot(vals, vals)
plt.scatter(ox, oy, alpha=0.1, label='O')
plt.scatter(sex, sey, alpha=0.1, label='SE')
plt.scatter(sedx, sedy, alpha=0.1, label='SED')
plt.scatter(seddx, seddy, alpha=0.1, label='SEDD')
plt.scatter(bx, by, alpha=0.1, label='B', color='black')
plt.legend()
plt.title(name)
plt.xlabel('Reward (no error)')
plt.ylabel('Reward (Fractional phase transient error = 0.01)')                            # TODO
# plt.show()
plt.savefig(path + name + ' ' + additional + '.png', dpi=500)                             # TODO
