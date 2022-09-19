'''
4/27/22
Owen Eskandari

Thesis work 22S

This is to merge two PSS objects into one and keep all of the information about it in a third PSS object.
Note: Use for two disjoint dictionaries. If the same sequence name (serial number) is in both, it will be copied
twice into the new pss object (I didn't find a quick solution to this problem)
'''

from PulseSequences import PulseSequences as PSS
from PulseSequence import PulseSequence as PS
import pickle
from datetime import datetime


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# # File with PSS data
# f = '/Users/oweneskandari/Desktop/PSS_output_20220426_190643.pkl'
#
# with open(f, 'rb') as f:
#     pss = pickle.load(f)


def merge_pss(pss1, pss2):

    new = PSS()     # Create new PSS object

    dict1 = pss1.get_ps_dict()      # Get the dictionary of sequences from PSS1
    for key in dict1:               # Add PS objects to the new PSS
        for idx, value in enumerate(dict1[key]):
            new.add_ps_object(value)

    dict2 = pss2.get_ps_dict()      # Repeat for PSS2
    for key in dict2:
        for idx, value in enumerate(dict2[key]):
            new.add_ps_object(value)

    new.make_entries_singular()     # Reduce dictionary to one PS object per entry

    return new      # Return the new PSS object


# ps_dict = pss.get_ps_dict()
#
# # Get some random ps objects that have one find
# objs = []
# i = 0
# for key in ps_dict:
#     for idx, value in enumerate(ps_dict[key]):
#         if len(objs) >= 3:
#             break
#         n = value.get_timesfound()
#         name = value.get_names()
#         if n == 1:
#             objs.append(value)
#
#
# for ps_obj in objs:
#     print(ps_obj.get_names())
#
# # Duplicate a sequence (the second one) and add to the new pss object
# dup = PS(objs[1].get_ps(), objs[1].get_orig_row_data(), 22, 2, actionspace='O')
# dup1 = PS(objs[2].get_ps(), objs[2].get_orig_row_data(), 10938462, 2, actionspace='O')
#
# dup2 = PS(objs[1].get_ps(), objs[1].get_orig_row_data(), 222, 2, actionspace='O')
#
#
# pss1 = PSS()
# pss2 = PSS()
# for idx, ps_obj in enumerate(objs):
#     pss1.add_ps_object(ps_obj)
#
#
# pss1.add_ps_object(dup)
# pss1.add_ps_object(dup1)
# pss1.make_entries_singular()
#
# pss2.add_ps_object(dup2)
# pss2.make_entries_singular()
#
# print(pss1.get_ps_dict())
# print(pss2.get_ps_dict())
#
# print('')
# print(pss1.get_repeated_objects())
# print(pss2.get_repeated_objects())
#
#
# print('')
# # quit()
#
# new_pss = merge_pss(pss1, pss2)
#
# dictnew = new_pss.get_ps_dict()  # Get the dictionary of sequences from PSS1
# for key in dictnew:  # Add PS objects to the new PSS
#     for idx, value in enumerate(dictnew[key]):
#         print(value.get_names())

# # 4/29/22 First merge of old and manually created data
# # Files with PSS data
# f1 = '/Users/oweneskandari/Desktop/Manually_Created_Sequences_PSS_Data.pkl'
#
# with open(f1, 'rb') as f:
#     hist_data = pickle.load(f)
#
# f2 = '/Users/oweneskandari/Desktop/Old_Sequences_PSS_Data_updated_4_29.pkl'
#
# with open(f2, 'rb') as f:
#     old_data = pickle.load(f)
#
# total = merge_pss(hist_data, old_data)
#
# print(total.get_length())
# print(len(total.get_repeated_objects()))
# print(total.get_repeated_objects()[0].get_names())
#
# path = '/Users/oweneskandari/Desktop/'
# outf = path + 'PSS_Non_Hist_Sequences_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.pkl'
# save_object(total, outf)     # Don't want to accidentally save again


# 4/29/22  Merge hist with non hist data
# Files with PSS data
f1 = '/Users/oweneskandari/Desktop/PSS_output_20220428_130335.pkl'

with open(f1, 'rb') as f:
    hist_data = pickle.load(f)

f2 = '/Users/oweneskandari/Desktop/PSS_Non_Hist_Sequences_20220429_161753.pkl'

with open(f2, 'rb') as f:
    old_data = pickle.load(f)

total = merge_pss(hist_data, old_data)

print(total.get_length())
print(len(total.get_repeated_objects()))

path = '/Users/oweneskandari/Desktop/'
outf = path + 'PSS_All_Sequences_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.pkl'
save_object(total, outf)     # Don't want to accidentally save again
