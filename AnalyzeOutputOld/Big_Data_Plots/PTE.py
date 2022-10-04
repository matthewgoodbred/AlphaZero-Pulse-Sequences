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

sp0, sp1, sp2, sp3, sp4 = [], [], [], [], []
for sequence in ps_dict:
    for idx, ps_obj in enumerate(ps_dict[sequence]):
        # if ps_obj.get_orig_name() == 'CORY48':
        #     print("HERE!!!")
        #     sp0.append(ps_obj.get_no_error_fid())
        #     sp1.append(ps_obj.get_pte_fids()[1])
        #     sp2.append(ps_obj.get_pte_fids()[2])
        #     sp3.append(ps_obj.get_pte_fids()[3])
        #     sp4.append(ps_obj.get_pte_fids()[4])

        if ps_obj.get_orig_name() == '1259819_72_SED_15144':
            sp0.append(ps_obj.get_no_error_fid())
            sp1.append(ps_obj.get_pte_fids()[1])
            sp2.append(ps_obj.get_pte_fids()[2])
            sp3.append(ps_obj.get_pte_fids()[3])
            sp4.append(ps_obj.get_pte_fids()[4])

        if ps_obj.get_no_error_fid() != 200:        # For now
            no_error.append(ps_obj.get_no_error_fid())
            pte1.append(ps_obj.get_pte_fids()[1])
            pte2.append(ps_obj.get_pte_fids()[2])
            pte3.append(ps_obj.get_pte_fids()[3])
            pte4.append(ps_obj.get_pte_fids()[4])


name = 'Reward over 288 ' + r'$\tau$' + ' due to Phase Transient Error'
path = '/Users/oweneskandari/PycharmProjects/21X/rl_pulse/22S/Thesis_Plots_All/'

plt.scatter([0]*len(no_error),  no_error, alpha=0.5, color='b')
plt.scatter([0.0001]*len(pte1), pte1,     alpha=0.5, color='b')
plt.scatter([0.001]*len(pte2),  pte2,     alpha=0.5, color='b')
plt.scatter([0.01]*len(pte3),   pte3,     alpha=0.5, color='b')
plt.scatter([0.02]*len(pte4),   pte4,     alpha=0.5, color='b')


plt.scatter([0]*     len(sp0), sp0,     alpha=0.5, color='g', label='1259819_72_SED_15144')
plt.scatter([0.0001]*len(sp1), sp1,     alpha=0.5, color='g')
plt.scatter([0.001]* len(sp2), sp2,     alpha=0.5, color='g')
plt.scatter([0.01]*  len(sp3), sp3,     alpha=0.5, color='g')
plt.scatter([0.02]*  len(sp4), sp4,     alpha=0.5, color='g')

# plt.scatter(special_x, special_y, alpha=0.5, label='Special')
plt.legend()
plt.xscale('log')
plt.title(name, fontsize=14)
plt.xlabel('Fractional Phase Transient Error (relative to ' + r'$\pi$' + '/2 pulse)', fontsize=12)
plt.ylabel('Reward', fontsize=12)
# plt.show()
plt.savefig(path + name + '.png', dpi=500)
