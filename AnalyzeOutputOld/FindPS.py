'''
4/27/22
Owen Eskandari

Thesis Work 22S

This file is to be used to find a specific pulse sequence
'''

from PulseSequence import PulseSequence as PS
from PulseSequences import PulseSequences as PSS
import pickle
from Convert import convert
from Matrix_Representation_Standalone import visual_representation

# to_find = [4, 1, 0, 3, 1, 0, 1, 3, 0, 1, 3, 0, 2, 4, 0, 1, 3, 0, 4, 2, 0, 3, 1, 0, 3, 1, 0, 2, 3, 0, 2, 4, 0, 1, 4, 0, 1, 4, 0, 4, 2, 0, 2, 4, 0, 1, 4, 0, 3, 2, 0, 3, 2, 0, 3, 2, 0, 4, 2, 0, 2, 3, 0, 2, 3, 0, 4, 1, 0, 4, 1, 0]
#
# print(convert([5, 1, 3, 1, 5, 6, 8, 2, 4, 2, 8, 7, 6, 1, 1, 3, 8, 7, 8, 7, 3, 4, 4, 5], 'SEDD', 'O'))
#
# File with PSS data
# f = '/Users/oweneskandari/Desktop/PSS_with_Fids_20220502_123919.pkl'
#
# with open(f, 'rb') as f:
#     pss = pickle.load(f)
#
# pss_dict = pss.get_ps_dict()
#
# # if str(to_find) in pss_dict:
# #     ps = pss_dict[str(to_find)][0]
# #
# #     print(ps.get_names())
# #     print(ps.get_rot_errors())
# #
# # ps.help()
#
# name = '1196362_24_SE_1075'
# name = 'seae24f'
# name = '1146029_48_SE_3'
# name = '1260857_36_SED_955'
# name = '2*l24+2x seae24f'
#
# for ps in pss_dict:
#     for idx, value in enumerate(pss_dict[ps]):
#         for n in value.get_names():
#             if name in n:
#                 print(value.get_names())
#                 print(value.get_ps())
#                 print(ps)
#                 quit()


# [4, 2, 2, 4, 4, 2, 3, 1, 2, 3, 4, 1, 4, 1, 2, 4, 2, 3, 1, 3, 1, 3, 3, 2]
# [4, 2, 2, 4, 4, 2, 3, 1, 2, 3, 4, 1, 4, 1, 2, 4, 2, 3, 1, 3, 1, 3, 3, 2]

folder = '/Users/oweneskandari/PycharmProjects/21X/rl_pulse/22S/Thesis_Plots_All/Matrix/'
# seq = [2, 3, 0, 4, 2, 0, 3, 2, 0, 1, 3, 0, 4, 1, 0, 2, 4, 0, 1, 4, 0, 1, 4, 0, 3, 2, 0, 2, 4, 0, 3, 1, 0, 3, 1, 0,
#        3, 1, 0, 3, 2, 0, 2, 3, 0, 4, 2, 0, 4, 1, 0, 1, 4, 0, 1, 3, 0, 2, 4, 0, 1, 4, 0, 3, 2, 0, 4, 2, 0, 1, 3, 0,
#        3, 1, 0, 1, 3, 0, 2, 3, 0, 1, 3, 0, 3, 1, 0, 3, 2, 0, 4, 2, 0, 1, 4, 0, 2, 4, 0, 1, 4, 0, 4, 2, 0, 4, 1, 0,
#        3, 2, 0, 1, 3, 0, 1, 3, 0, 2, 3, 0, 4, 2, 0, 4, 1, 0, 4, 2, 0, 4, 1, 0, 2, 3, 0, 2, 4, 0, 2, 4, 0, 3, 1, 0]

seq = [1,4,1,1,4,1]
seq = [4,1,1,4,1,1]
seq = [1, 3, 0, 2, 4, 0, 3, 2, 0, 3, 2, 0, 3, 1, 0, 2, 3, 0, 1, 3, 0, 2, 4, 0, 3, 1, 0, 4, 2, 0, 4, 2, 0, 1, 4, 0]  # SED24
seq = [2, 4, 0, 2, 3, 0, 2, 3, 0, 4, 2, 0, 3, 2, 0, 4, 2, 0, 1, 3, 0, 1, 4, 0, 1, 4, 0, 3, 1, 0, 4, 1, 0, 3, 1, 0,
       1, 3, 0, 3, 1, 0, 3, 1, 0, 4, 1, 0, 2, 3, 0, 2, 4, 0, 2, 4, 0, 2, 3, 0, 4, 1, 0, 4, 2, 0, 4, 2, 0, 1, 3, 0]  #SED48

print(visual_representation(seq, 'O', name='SED48', plot=True, folder=folder, dpi=500))
quit()
# # 5/10/22
# print(len(pss_dict))
# for sequence in pss_dict:
#     for idx, ps_obj in enumerate(pss_dict[sequence]):
#         errs = ps_obj.get_pte_fids()  # TODO
#         if errs[2] > 6:
#             print(ps_obj.get_names())
#             print(ps_obj.get_ps())

# print(len(pss_dict))
# for sequence in pss_dict:
#     for idx, ps_obj in enumerate(pss_dict[sequence]):
#         errs = ps_obj.get_rot_error_fids()  # TODO
#         if errs[2] > 3:
#             print(ps_obj.get_names())
#             print(ps_obj.get_ps())
#             print(errs)

print(len(pss_dict))
for sequence in pss_dict:
    for idx, ps_obj in enumerate(pss_dict[sequence]):
        errs = ps_obj.get_offset_fids()  # TODO
        if errs[8] > 2 and errs[7] > 3 :
            print(ps_obj.get_names())
            print(ps_obj.get_ps())
            print(errs[4:])


# 5/10/22
CORY48_se = [1,3,2,3,1,3,1,3,1,4,1,3,4,2,3,2,4,2,4,2,4,1,4,2,2,3,2,4,2,3,1,4,2,4,1,4,3,2,3,1,3,2,4,1,3,1,4,1]
CORY48 = [1,3,0,2,3,0,1,3,0,1,3,0,1,4,0,1,3,0,4,2,0,3,2,0,4,2,0,4,2,0,4,1,0,4,2,0,2,3,0,2,4,0,2,3,0,1,4,0,2,4,0,1,4,0,3,2,0,3,1,0,3,2,0,4,1,0,3,1,0,4,1,0]
folder = '/Users/oweneskandari/PycharmProjects/21X/rl_pulse/22S/Thesis_Plots_All/Matrix/'
# visual_representation(CORY48_se, 'O', name='CORY48 SE', folder=folder, plot=True)