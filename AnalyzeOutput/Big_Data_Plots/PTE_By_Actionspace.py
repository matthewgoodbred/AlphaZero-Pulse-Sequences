'''
5/2/22
Owen Eskandari
Thesis Work 22S

This is to make some plots of rewards versus phase transient error.
Ideas:
-By action space
-By pbc
    -Specifically, for the 2,3,4,5 pbc runs for SE in the fall
-By error/no error
-Gradient for time (by serial number)
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

print(pss)
ps_dict = pss.get_ps_dict()
print(len(ps_dict))

no_error = []
pte1 = []
pte2 = []
pte3 = []
pte4 = []

special_x = []
special_y = []
for sequence in ps_dict:
    for idx, ps_obj in enumerate(ps_dict[sequence]):
        # if ps_obj.get_orig_name() == 'CORY48':
        #     print("HERE!!!")
        #     special_x.append(ps_obj.get_no_error_fid())
        #     special_y.append(ps_obj.get_pte_fids()[2])

        # if ps_obj.get_orig_name() == '1259819_72_SED_15144':
        #     special_x.append(ps_obj.get_no_error_fid())
        #     special_y.append(ps_obj.get_pte_fids()[2])
        #     print(ps_obj.get_pte_fids())

        if ps_obj.get_no_error_fid() != 200:        # For now
            no_error.append(ps_obj.get_no_error_fid())
            pte1.append(ps_obj.get_pte_fids()[1])
            pte2.append(ps_obj.get_pte_fids()[2])
            pte3.append(ps_obj.get_pte_fids()[3])
            pte4.append(ps_obj.get_pte_fids()[4])

            # Color code by action space
            spaces = ps_obj.get_action_spaces()

            # Color code by action space
            if 'O' in spaces:
                x1.append(reward1)
                y1.append(reward2)
            if 'SE' in spaces:
                x2.append(reward1)
                y2.append(reward2)
            if 'SED' in spaces:
                x3.append(reward1)
                y3.append(reward2)
            if 'SEDD' in spaces:
                x4.append(reward1)
                y4.append(reward2)
            if 'B' in spaces:
                x5.append(reward1)
                y5.append(reward2)


name = 'Reward over 288 ' + r'$\tau$' + ' due to Phase Transient Error'
path = '/Users/oweneskandari/PycharmProjects/21X/rl_pulse/22S/Thesis_Plots_All/'

# vals = np.linspace(-0.5, 6, 1000)       # cutoff --> -0.5 for when the cutoff is -1
# plt.plot(vals, vals)
plt.scatter([0]*len(no_error), no_error, alpha=0.5, color='b')
plt.scatter([0.0001]*len(pte1), pte1,  alpha=0.5, color='b')
plt.scatter([0.001]*len(pte2), pte2, alpha=0.5, color='b')
plt.scatter([0.01]*len(pte3), pte3, alpha=0.5, color='b')
plt.scatter([0.02]*len(pte4), pte4, alpha=0.5, color='b')


# plt.scatter(special_x, special_y, alpha=0.5, label='Special')
# plt.legend()
plt.xscale('log')
plt.title(name, fontsize=14)
plt.xlabel('Fractional Phase Transient Error (relative to ' + r'$\pi$' + '/2 pulse)', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.show()
# plt.savefig(path + name + '.png', dpi=500)