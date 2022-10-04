'''
4/28/22
Owen Eskandari
Thesis Work 22S

This is a trial to reproduce the plot from a few days ago in a much faster manner now that my PSS has fidelity data
'''

import pickle
from PulseSequences import PulseSequence as PS
from PulseSequences import PulseSequences as PSS
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Open file and get PSS object

f = '/Users/oweneskandari/Desktop/PSS_with_Fids_20220502_123919.pkl'        # TODO: Change
with open(f, 'rb') as f:
    pss = pickle.load(f)

ps_dict = pss.get_ps_dict()
print(len(ps_dict))

no_error = []
pte_error = []

for sequence in ps_dict:
    for idx, ps_obj in enumerate(ps_dict[sequence]):
        if ps_obj.get_no_error_fid() != 200:        # For now
            no_error.append(ps_obj.get_no_error_fid())
            pte_error.append(ps_obj.get_pte_fids()[1])

name = 'Fractional Phase Transient Error = 0.0001'
path = '/Users/oweneskandari/PycharmProjects/21X/rl_pulse/22S/Thesis_Plots_All/'

vals = np.linspace(-0.5, 6, 1000)       # cutoff --> -0.5 for when the cutoff is -1
plt.plot(vals, vals)
plt.scatter(no_error, pte_error, alpha=0.5, label='Phase Transient Error')
plt.legend()
plt.title(name)
plt.xlabel('Reward (no error)')
plt.ylabel('Reward (Fractional phase transient error = 0.0001)')
plt.savefig(path + name + '.png', dpi=500)
