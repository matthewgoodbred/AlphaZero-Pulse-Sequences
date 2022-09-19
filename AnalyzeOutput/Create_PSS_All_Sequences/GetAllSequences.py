'''
Created on 4/26/22
Owen Eskandari
Thesis work 22S

Time to do real data stuff here.
-Find, name, and categorize all pulse sequences found with the histogram method
    -Find and plot all sequences found in multiple runs (against their fidelities?)
'''

import pandas as pd
from PulseSequence import PulseSequence as PS, pulse_list_dict
import pickle
import json
import ast      # For converting '[1, 2, 3...]' into [1, 2, 3...]
from PulseSequences import PulseSequences as PSS
from datetime import datetime


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


rows = ['20220204-010509',
        '20220208-132247',
        '20220208-132248',
        '20220214-125827',
        '20220216-041826',
        '20220223-114317',
        '20220329-145743',
        '20220403-145801',
        '20220217-164952',
        '20220218-093251',
        '20220219-131449',
        '20220219-171501',
        '20220225-185815',
        '20220303-140810',
        '20220309-135645',
        '20220314-145620',
        '20220319-145648',
        '20220324-145718',
        '20220407-170829',
        '20220421-152316',
        '20220423-152318',
        '20220425-181127',
        '20220425-181204']     # TODO: change each time


fcounts = ['pulse_counts_20000.csv',
           'pulse_counts_145000.csv',
           'pulse_counts_245000.csv',
           'pulse_counts_180000.csv',
           'pulse_counts_25000.csv',
           'pulse_counts_190000.csv',
           'pulse_counts_340000.csv',
           'pulse_counts_165000.csv',
           'pulse_counts_25000.csv',
           'pulse_counts_185000.csv',
           'pulse_counts_10000.csv',
           'pulse_counts_135000.csv',
           'pulse_counts_140000.csv',
           'pulse_counts_170000.csv',
           'pulse_counts_300000.csv',
           'pulse_counts_325000.csv',
           'pulse_counts_290000.csv',
           'pulse_counts_300000.csv',
           'pulse_counts_285000.csv',
           'pulse_counts_150000.csv',
           'pulse_counts_115000.csv',
           'pulse_counts_125000.csv',
           'pulse_counts_55000.csv']    # TODO: change each time
# frewards = []     # For now

print(len(fcounts))
print(len(rows))

server_path = '/Volumes/f003dkt/'
path = '/Users/oweneskandari/Desktop/'

csv = '/Users/oweneskandari/Desktop/ThesisResults_OE_4_48.csv'      # TODO: change each time

# Results from the Excel spreadsheet of runs
df = pd.read_csv(csv)

pss = PSS()     # Start a PulseSequences object to add the pulse sequences to from all of the files

for idx, row in enumerate(rows):

    row_info = pd.read_csv(csv).loc[df['Folder name'] == row].values.tolist()[0]        # Relevant information

    # Opening dictionaries from output files TODO: change each time
    if idx < 8:
        filename = server_path + 'SolidEcho/' + row + '/dict_info/' + fcounts[idx]
    elif idx < 19:
        filename = server_path + 'SED/' + row + '/dict_info/' + fcounts[idx]
    elif idx < 21:
        filename = server_path + 'Both/' + row + '/dict_info/' + fcounts[idx]
    else:
        filename = server_path + 'O/' + row + '/dict_info/' + fcounts[idx]

    print(filename)

    with open(filename, 'r') as f:
        count_dict = json.load(f)
    # with open(frewards[idx], 'r') as f:       # Don't need this for now
    #     reward_dict = json.load(f)

    pulse_sequences = []

    for i, key in enumerate(count_dict):
        pulse_sequences.append([ast.literal_eval(key), i, count_dict[key]])  # ps, num, count, reward_dict[key] fidelity

    for pulse_sequence in pulse_sequences:
        ps_obj = PS(pulse_sequence[0], row_info, pulse_sequence[1], pulse_sequence[2])

        pss.add_ps_object(ps_obj)


pss.make_entries_singular()

print(len(pss.get_repeated_objects()))
print(pss.get_length())

outf = path + 'PSS_output_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.pkl'
# save_object(pss, outf)     # Don't want to accidentally save again

