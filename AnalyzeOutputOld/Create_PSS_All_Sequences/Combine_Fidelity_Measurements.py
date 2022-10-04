'''
4/28/22
Owen Eskandari
Thesis Work 22S

This file is to combine all of the fidelity measurements generated from the scripts in Add_Fidilities and make a new,
relatively complete PSS data object

+Get all of the data from the data files from the scripts in Add_Fidelities
+Combine into one large dictionary where the key is the string of the pulse sequence
+Get the PSS data
+Make a new PSS data object, and for every key in the combined dictionary, make a PS object combining the information
    from the opened PSS data object with the relevant fidelity information
+Save this new PSS object as PSS_with_Fids_datetime.pkl
'''

import pickle5
import pickle
from PulseSequences import PulseSequence as PS
from PulseSequences import PulseSequences as PSS
import os
from datetime import datetime


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# Get all of the data from the relevant data files

'''
Order of files:
large_neg_offset_fids
large_pos_offset_fids
large_pte_fids
large_rot_err_fids
no_err_fids
small_neg_offset_fids
small_pos_offset_fids
small_pte_fids
small_rot_err_fids
'''

large_neg_offset_fids = {}
large_pos_offset_fids = {}
large_pte_fids = {}
large_rot_err_fids = {}
no_err_fids = {}
small_neg_offset_fids = {}
small_pos_offset_fids = {}
small_pte_fids = {}
small_rot_err_fids = {}

path = '/Users/oweneskandari/Desktop/Add_Fidelities_Discovery_Data/'


for idx, df in enumerate(os.listdir(path)):
    with open(path + df, 'rb') as f:
        print(df)
        if idx == 0:
            no_err_fids = pickle5.load(f)
        if idx == 1:
            large_rot_err_fids = pickle5.load(f)
        if idx == 2:
            small_pte_fids = pickle5.load(f)
        if idx == 3:
            small_rot_err_fids = pickle5.load(f)
        if idx == 4:
            small_neg_offset_fids = pickle5.load(f)
        if idx == 5:
            large_neg_offset_fids = pickle5.load(f)
        if idx == 6:
            large_pos_offset_fids = pickle5.load(f)
        if idx == 7:
            small_pos_offset_fids = pickle5.load(f)
        if idx == 8:
            large_pte_fids = pickle5.load(f)


dictionaries = [             # TODO: pickle does not run in order; reorder to fit (print list first)
                no_err_fids,
                large_rot_err_fids,
                small_pte_fids,
                small_rot_err_fids,
                small_neg_offset_fids,
                large_neg_offset_fids,
                large_pos_offset_fids,
                small_pos_offset_fids,
                large_pte_fids]

print(len(dictionaries))

# Combine all of the dictionaries
fidelities_dictionary = {}

# for key in no_err_fids:
#     print(key)
#     input()

for key in dictionaries[0]:
    '''
    Order to add:
    no_err_fids
    
    small_rot_err_fids
    no_err_fids
    large_rot_err_fids
    
    small_pte_fids
    no_err_fids
    large_pte_fids
    
    large_neg_offset_fids
    small_neg_offset_fids
    no_err_fids
    small_pos_offset_fids
    large_pos_offset_fids
    '''

    fidelities_dictionary[key] = [no_err_fids[key],

                                  no_err_fids[key],
                                  small_rot_err_fids[key][0], small_rot_err_fids[key][1],
                                  large_rot_err_fids[key][0], large_rot_err_fids[key][1],

                                  no_err_fids[key],
                                  small_pte_fids[key][0], small_pte_fids[key][1],
                                  large_pte_fids[key][0], large_pte_fids[key][1],

                                  large_neg_offset_fids[key][1], large_neg_offset_fids[key][0],
                                  small_neg_offset_fids[key][1], small_neg_offset_fids[key][0], no_err_fids[key],
                                  small_pos_offset_fids[key][0], small_pos_offset_fids[key][1],
                                  large_pos_offset_fids[key][0], large_pos_offset_fids[key][1]]


# Get the PSS data
no_fid_pss_file = '/Users/oweneskandari/Desktop/PSS_All_Sequences_20220429_162015.pkl'      # TODO: Change
with open(no_fid_pss_file, 'rb') as f:
    no_fid_pss = pickle.load(f)

basic = no_fid_pss.get_ps_dict()
for ps in basic:
    # print(ps)
    # for idx, value in enumerate(basic[ps]):
    #     print(value.get_orig_row_data())
    pass


# Make a new PSS data object, and for every key in the combined dictionary, make a PS object combining the information
# from the opened PSS data object with the relevant fidelity information

fid_pss = PSS()
no_fid_dict = no_fid_pss.get_ps_dict()


for ps in fidelities_dictionary:        # ps is the string of the sequence
    if ps in no_fid_dict:       # If it's in the old PSS object (it should be)

        # Make a new PS object
        ps_obj_with_fid = no_fid_dict[ps][0]      # Find the PS object with the sequence ps

        # Add fidelity information
        no_err_info = fidelities_dictionary[ps][0]
        rot_err_info = fidelities_dictionary[ps][1:6]
        pte_err_info = fidelities_dictionary[ps][6:11]
        offset_info = fidelities_dictionary[ps][11:]

        ps_obj_with_fid.set_no_error_fid(no_err_info)
        ps_obj_with_fid.set_rot_error_fids(rot_err_info)
        ps_obj_with_fid.set_pte_fids(pte_err_info)
        ps_obj_with_fid.set_offset_fids(offset_info)

        # Add PS object to new PSS
        fid_pss.add_ps_object(ps_obj_with_fid)


# # Suspect
# print(fid_pss.get_repeated_objects())
# print(fid_pss.make_entries_singular())
# print(fid_pss.get_length())
# for ps in fid_pss.get_repeated_objects():
#     print(ps.get_names())
# print(fid_pss.get_repeated_objects())

# Save the output
outf = '/Users/oweneskandari/Desktop/PSS_with_Fids_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.pkl'
print(outf)
save_object(fid_pss, outf)     # TODO: Don't want to accidentally save again
