'''
4/26/22
Owen Eskandari
Thesis work 22S

This file is to add the data from a new run to the pre-existing PulseSequences object and save it in a new dated file
'''

# Imports
from PulseSequence import PulseSequence as PS
from PulseSequences import PulseSequences as PSS
import pickle
import pandas as pd
import json
import ast      # For converting '[1, 2, 3...]' into [1, 2, 3...]
from datetime import datetime
import os


def add_new_sequences(old_file, max_iteration_df, csv):
    '''

    :param old_file: Str. Path and name of the old file containing all of the relevant data (PSS data)
    :param max_iteration_df: Str. Path and name of file containing the data to add
    :param csv: Str. Path and name of the excel or csv file containing the run information
    :return: Str. Name and path of the new file
    '''
    # First, get the old data
    with open(old_file, 'rb') as f:
        pss = pickle.load(f)

    # Then add the new data
    paths = max_iteration_df.split('/')
    row = paths[-3]

    df = pd.read_csv(csv)
    row_info = pd.read_csv(csv).loc[df['Folder name'] == row].values.tolist()[0]  # Relevant information

    with open(max_iteration_df, 'r') as f:
        count_dict = json.load(f)

    pulse_sequences = []

    for i, key in enumerate(count_dict):
        pulse_sequences.append([ast.literal_eval(key), i, count_dict[key]])  # ps, num, count

    for pulse_sequence in pulse_sequences:
        ps_obj = PS(pulse_sequence[0], row_info, pulse_sequence[1], pulse_sequence[2])
        pss.add_ps_object(ps_obj)

    pss.make_entries_singular()

    # And finally, save in a new dated file
    outfile = os.path.dirname(old_file) + '/PSS_output_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.pkl'

    print(outfile)

    with open(outfile, 'wb') as f:  # Overwrites any existing file.
        pickle.dump(pss, f, pickle.HIGHEST_PROTOCOL)

    return outfile


# in_f = add_new_sequences('/Users/oweneskandari/Desktop/test3_all_pulse_sequences_4_26.pkl',
#                   '/Volumes/f003dkt/Both/20220421-152316/dict_info/pulse_counts_150000.csv',
#                   '/Users/oweneskandari/Desktop/ThesisResults_OE_4_26.csv')
#
# add_new_sequences(in_f,
#                   '/Volumes/f003dkt/Both/20220423-152318/dict_info/pulse_counts_115000.csv',
#                   '/Users/oweneskandari/Desktop/ThesisResults_OE_4_26.csv')
