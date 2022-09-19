'''
5/2/22
Owen Eskandari
Thesis work 22S

This is to find all pulse sequences at or above thresholds for no error and all three types of error
Will I find new really good sequences? We'll see
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

for sequence in ps_dict:
    for idx, ps_obj in enumerate(ps_dict[sequence]):
        no_err_reward = ps_obj.get_no_error_fid()
        if no_err_reward == 200:        # For now
            no_err_reward = 10
        aerrs = ps_obj.get_rot_error_fids()
        ptes = ps_obj.get_pte_fids()
        offs = ps_obj.get_offset_fids()
        for idx, i in enumerate(offs):
            if i == 200:
                offs[idx] = 10

        spaces = ps_obj.get_action_spaces()

        # if no_err_reward > 6 and aerrs[2] > 2.33 and ptes[3] > 0.72 and offs[7] > 3.26:
        #     print(ps_obj.get_names())
        #     print(ps_obj.get_ps())
        if no_err_reward > 6 and aerrs[2] > 2.33 and ptes[3] > 0.72 and offs[6] > 5.15:
            print(ps_obj.get_names())
            print(ps_obj.get_ps())
