'''
5/3/22
Owen Eskandari
Thesis Work 22S

This is to take data from text files, find the relevant pulse sequences, and plot the data for four runs
All SE action space, short time period (10-48 hours), pbc values of 2, 3, 4, 5
'''

import numpy as np
import pandas as pd
from PulseSequence import PulseSequence as PS
from PulseSequences import PulseSequences as PSS
import os
import ast
import pickle
from datetime import datetime


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

path = '/Users/oweneskandari/Desktop/Pbc_plot/'
csv = '/Users/oweneskandari/Desktop/ThesisResults_OE_4_48.csv'      # TODO: change each time

# Results from the Excel spreadsheet of runs
df = pd.read_csv(csv)

pss = PSS()     # Start a PulseSequences object to add the pulse sequences to from all of the files

rows = ['20211030-024222',
        '20211030-125220',
        '20211029-023309',
        '20211030-233234']

for f in os.listdir(path):

    if f == '2.txt':
        row_info = pd.read_csv(csv).loc[df['Folder name'] == rows[0]].values.tolist()[0]  # Relevant information
    if f == '3.txt':
        row_info = pd.read_csv(csv).loc[df['Folder name'] == rows[1]].values.tolist()[0]  # Relevant information
    if f == '4.txt':
        row_info = pd.read_csv(csv).loc[df['Folder name'] == rows[2]].values.tolist()[0]  # Relevant information
    if f == '5.txt':
        row_info = pd.read_csv(csv).loc[df['Folder name'] == rows[3]].values.tolist()[0]  # Relevant information

    f = os.path.join(path, f)

    d = {}

    with open(f, 'r') as f:
        for line in f:
            words = line.split(' ')
            if 'candidate' in words:
                words = line.split('(')
                sequence = words[1][0:72]

                if sequence in d:
                    d[sequence] += 1
                else:
                    d.update({sequence: 1})

    pulse_sequences = []

    for i, key in enumerate(d):
        pulse_sequences.append([ast.literal_eval(key), i, d[key]])

    for pulse_sequence in pulse_sequences:
        '''
        Things I need
        +row info
        +number
        +count
        '''
        ps_obj = PS(pulse_sequence[0], row_info, pulse_sequence[1], pulse_sequence[2])

        pss.add_ps_object(ps_obj)


pss.make_entries_singular()     # No repeated objects

# Save the output
outf = '/Users/oweneskandari/Desktop/PBC_Plot_PSS' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.pkl'
print(outf)
save_object(pss, outf)     # TODO: Don't want to accidentally save again
