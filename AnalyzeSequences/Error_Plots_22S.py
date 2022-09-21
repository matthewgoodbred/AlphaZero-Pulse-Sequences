# Created 2/10/22 from Reproduce_Plots.py
# Owen Eskandari
# This file is to create plots showing robustness against angle error, phase transient error, and offset error

import qutip as qt
import pulse_sequences_not_stochastic as ps        # TODO: change back to create_taus
# import pulse_sequences as ps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import string       # For making a list of names
import datetime
from Convert import convert
# yxx24 = [4, 1, 2, 3, 2, 2, 3, 2, 1, 4, 1, 1, 3, 2, 1, 4, 1, 1, 4, 1, 2, 3, 2, 2]        # 24 tau: 6.43; 288 tau: 4.32
# angle12 = [4, 1, 2, 3, 2, 2, 4, 1, 2, 3, 1, 1]      # 12 tau: 6.02; 288 tau: 3.29
# PW12 = [4, 2, 2, 1, 1, 3, 2, 2, 4, 3, 1, 1]         # 12 tau: 5.98; 288 tau: 3.18
# Ideal6 = [3, 1, 1, 3, 2, 2]
# yxx48 = [3,2,2,3,2,2,4,1,1,3,2,2,4,1,1,4,1,1,3,2,2,3,2,2,4,1,1,3,2,2,4,1,1,4,1,1,3,2,2,4,1,1,3,2,2,4,1,1]
# Offset48 = [4,4,4,2,4,4,4,4,2,2,2,4,2,2,4,4,4,2,2,2,4,4,4,4,2,2,4,2,2,2,2,2,2,4,4,4,2,2,2,2,2,2,4,2,2,2,4,2]
#
# # CORY48 is 72 tau
# CORY48 = [1,3,0,2,3,0,1,3,0,1,3,0,1,4,0,1,3,0,4,2,0,3,2,0,4,2,0,4,2,0,4,1,0,4,2,0,2,3,0,2,4,0,2,3,0,1,4,0,2,4,0,1,4,0,3,2,0,3,1,0,3,2,0,4,1,0,3,1,0,4,1,0]
# # Transcribe CORY48 into other action spaces
# CORY48_SE = [1, 0, 3, 0, 1, 0, 1, 0, 2, 0, 1, 0, 8, 0, 6, 0, 8, 0, 8, 0, 7, 0, 8, 0, 3, 0, 4, 0, 3, 0, 2, 0, 4, 0, 2, 0, 6, 0, 5, 0, 6, 0, 7, 0, 5, 0, 7, 0]
# CORY48_SED = [1, 3, 1, 1, 2, 1, 8, 6, 8, 8, 7, 8, 3, 4, 3, 2, 4, 2, 6, 5, 6, 7, 5, 7]
# MREV8 = [0, 1, 4, 0, 3, 2, 0, 2, 4, 0, 3, 1]
#
#
# # From WILL
# az_all_err_12 = [
#     4, 2, 3, 3, 2, 3, 3, 2, 4, 4, 2, 4
# ]
# az_all_err_24 = [
#     4, 4, 2, 4, 4, 2, 3, 2, 3, 3, 2, 3,
#     1, 3, 1, 1, 3, 1, 4, 4, 1, 4, 4, 1
# ]
# az_all_err_48 = [
#     1, 1, 4, 1, 1, 4, 2, 4, 4, 2, 4, 4,
#     2, 4, 4, 2, 4, 4, 3, 3, 3, 2, 3, 3,
#     2, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 2,
#     1, 4, 4, 1, 4, 4, 4, 2, 2, 4, 2, 2
# ]

# SED
#pulse_list = [['X', 'Y', 'D'], ['X', '-Y', 'D'], ['-X', 'Y', 'D'], ['-X', '-Y', 'D'],
#              ['Y', 'X', 'D'], ['Y', '-X', 'D'], ['-Y', 'X', 'D'], ['-Y', '-X', 'D']]
# Primitive 
pulse_list = [['D'],  ['X'], ['-X'], ['Y'], ['-Y']]

def error_plots(pulse_sequences, error_type, name=None, show_fig=False, save_fig=False, granular=False, compare=None,
                folder=None, delay=None, pulse_width=None):
    '''

    :param pulse_sequences: 2-Tuple. (Sequence, name). Sequence is a list, name is a string
    :param error_type:
    :param name:
    :param show_fig:
    :param save_fig:
    :param granular:
    :param compare (list): Same format as pulse_sequences. This is if pulse_sequences is a list of standard for each
        pulse sequence in compare to be compared to. Defaults to None.
    :return:
    '''

    # New as of 5/23/22
    shapes = ['*', 'X', 'o', 'v', '^']
    shapes = ['-', ':', '-.', '--', '-']

    # pulse_sequences = list of (pulse sequence, label, action space)

    # Test the validity of the reward given to a sequence in a run
    def get_propagators(sequence):
        if len(sequence) == 0:
            return ([qt.identity(ps_config.Utarget.dims[0])]
                    * ps_config.ensemble_size)
        else:
            propagators = get_propagators(sequence[:-1])
            propagators = [prop.copy() for prop in propagators]
            for s in range(ps_config.ensemble_size):
                propagators[s] = (ps_config.pulses_ensemble[s][sequence[-1]].get_pulse() * propagators[s])
            return propagators

    def get_reward(sequence):
        propagators = get_propagators(sequence)
        fidelity = 0
        for s in range(ps_config.ensemble_size):
            fid = np.clip(
                qt.metrics.average_gate_fidelity(
                    propagators[s],
                    ps_config.Utarget
                ), 0, 1
            )
            fidelity += fid
        fidelity *= 1 / ps_config.ensemble_size
        reward = -1 * np.log10(1 - fidelity + 1e-200)
        return reward

    dipolar_strength = 1e2  # Originally 1e2 (in krad/sec)
    pulse_width = 0 if pulse_width is None else pulse_width      # Instantaneous pulses (TODO: was 0)
    # delay = 1e-4  # Delay (in seconds) between pulses (orig: 1e-4)
    delay = 1e-4 if delay is None else delay
    N = 4  # Number of spin systems (TODO: was orig: 3)
    # Ensemble_size gives the size of the different systems you're using (they differ by 'offset' which affects H_cs)
    # Offset occurs when rf field not matched to Larmor frequency of spins (this is a type of error)
    rot_error = 0
    phase_transient_error = 0
    offset_error = 0

    Utarget = qt.identity([2] * N)  # Target propagator. This is an N-dimensional object

    if not granular:
        rotation_angles = list(np.arange(-0.1, -0.05, 0.001)) + list(np.arange(-0.05, 0, 0.001)) + \
                          [0] + list(np.arange(0, 0.05, 0.001)) + list(np.arange(0.05, 0.1, 0.001))
        offset_errors = [0] + list(np.logspace(-1, 3, 100))     # Changed 3.1 --> 3 for thesis plots
        tau_lengths = np.logspace(-7, -4, 30)  # Original Values: -3.8, -2.8, 30 todo need to play around with this
        ptes = [0] + list(np.logspace(-5, -1.7, 50))        # Changed -1.5 --> -1.7 for thesis plots

    if granular:        # These plots have lower quality but run faster (Note: 10 seconds for 4 288 tau pulse sequences)
        rotation_angles = list(np.arange(-0.1, -0.01, 0.015)) + list(np.arange(-0.01, 0, 0.0025)) + \
                          [0] + list(np.arange(0, 0.01, 0.0025)) + list(np.arange(0.01, 0.1, 0.015))
        ptes = [0] + list(np.logspace(-5, -1.5, 15))
        offset_errors = [ele for ele in reversed([-1*i for i in list(np.logspace(-1, 3.1, 15))])] + [0] + list(np.logspace(-1, 3.1, 15))
        offset_errors = [0] + list(np.logspace(-1, 3.1, 15))     # granular
        tau_lengths = np.logspace(-7, -4, 30)  # Original Values: -3.8, -2.8, 30 todo need to play around with this

    pulse_sequences_orig, labels = [], []
    #Load desired pulse sequences into pulse_sequence_orig array and labels into labels array
    for pulse_sequence in pulse_sequences:
        pulse_sequence_orig = pulse_sequence[0]

        pulse_sequences_orig.append(pulse_sequence_orig)
        labels.append(pulse_sequence[1])

    # For plot generation
    if error_type == 'angle' or error_type == 'all':
        sequences_rewards = []
        for sequence in pulse_sequences_orig:
            sequences_rewards.append([])

        for rot_error in rotation_angles:
            ps_config = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                               ensemble_size=1,
                                               max_sequence_length=len(pulse_sequences_orig[0]),
                                               dipolar_strength=dipolar_strength,
                                               pulse_width=pulse_width, delay=delay,
                                               rot_error=rot_error,
                                               phase_transient_error=0,
                                               offset_error=0,
                                               pulse_list=pulse_list,
                                               stochastic=False)

            for idx, sequence in enumerate(pulse_sequences_orig):
                ps_config.reset()
                for pulse in sequence:
                    ps_config.apply(pulse)
                sequences_rewards[idx].append(get_reward(sequence))

        plot1 = plt.figure(1)
        plt.clf()
        plt.style.use('tableau-colorblind10')
        for shape, each in enumerate(sequences_rewards):
            plt.plot(rotation_angles, each, linestyle=shapes[shape])

        plt.xticks(fontsize=14)
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))  # 2 decimal places
        plt.locator_params(axis="x", nbins=6)
        plt.yticks(fontsize=14)
        plt.xlabel('Fractional Rotation Error (relative to ' + r'$\pi$' + '/2 pulse)', fontsize=14)
        plt.ylabel('Reward', fontsize=14)
        plt.title("Reward over 288" + r'$\tau$', fontsize=16)
        plt.legend(labels, fontsize=14)
        if save_fig:
            if name and folder:
                plt.savefig(folder + name + '_Rotation_error.png', dpi=500)
            elif name:
                plt.savefig('Error_Plots/' + name + '_Rotation_error.png')
            else:
                plt.savefig('Error_Plots/Rotation_error.png')
        if show_fig:
            # plt.show()
            pass
        # plt.close()

    if error_type == 'phase' or error_type == 'all':
        sequences_rewards = []
        for sequence in pulse_sequences_orig:
            sequences_rewards.append([])

        for phase_transient_error in ptes:
            #Creat pulse sequence config object to run the simulation with specific error 
            ps_config = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                               ensemble_size=1,
                                               max_sequence_length=len(pulse_sequences_orig[0]),
                                               dipolar_strength=dipolar_strength,
                                               pulse_width=pulse_width, delay=delay,
                                               rot_error=0,
                                               phase_transient_error=phase_transient_error,
                                               offset_error=0,
                                               pulse_list=pulse_list,
                                               stochastic=False)

            for idx, sequence in enumerate(pulse_sequences_orig):
                ps_config.reset()
                for pulse in sequence:
                    ps_config.apply(pulse)
                sequences_rewards[idx].append(get_reward(sequence))
                # reward = get_reward(sequence)
                # if phase_transient_error == 0:
                #     print(reward)
                # sequences_rewards[idx].append(reward)
        plot2 = plt.figure(2)
        plt.clf()
        plt.style.use('tableau-colorblind10')
        for shape, each in enumerate(sequences_rewards):
            plt.plot(ptes, each, linestyle=shapes[shape])

        plt.xscale('log')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Fractional Phase transient error (relative to ' + r'$\pi$' + '/2 pulse)', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.title("Reward over 288" + r'$\tau$', fontsize=14)
        plt.legend(labels, fontsize=12)
        if save_fig:
            if name and folder:
                plt.savefig(folder + name + '_Phase_transient_error.png', dpi=500)
            elif name:
                plt.savefig('Error_Plots/' + name + '_Phase_transient_error.png')
            else:
                plt.savefig('Error_Plots/Phase_transient_error.png')
        if show_fig:
            # plt.show()
            pass
        # plt.close()

    if error_type == 'offset' or error_type == 'all':
        sequences_rewards = []
        for sequence in pulse_sequences_orig:
            sequences_rewards.append([])

        for offset_error in offset_errors:
            ps_config = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                               ensemble_size=1,
                                               max_sequence_length=len(pulse_sequences_orig[0]),
                                               dipolar_strength=dipolar_strength,
                                               pulse_width=pulse_width, delay=delay,
                                               rot_error=0,
                                               phase_transient_error=0,
                                               offset_error=offset_error,
                                               pulse_list=pulse_list,
                                               stochastic=False)

            for idx, sequence in enumerate(pulse_sequences_orig):
                ps_config.reset()
                for pulse in sequence:
                    ps_config.apply(pulse)
                sequences_rewards[idx].append(get_reward(sequence))

        plot3 = plt.figure(3)       # To see all of the figures at once
        plt.clf()
        plt.style.use('tableau-colorblind10')
        lines = []      # To remove lines when comparing to standards
        for shape, each in enumerate(sequences_rewards):
            lines.append(plt.plot(offset_errors, each, linestyle=shapes[shape]))

        # lines[0].pop(0).remove()

        plt.xscale('log')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Fractional Offset error (relative to chemical shift strength)', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.title("Reward over 288" + r'$\tau$', fontsize=14)
        plt.legend(labels, fontsize=12)
        if save_fig:
            if name and folder:
                plt.savefig(folder + name + '_Offset_error.png', dpi=500)
            elif name:
                plt.savefig('Error_Plots/' + name + '_Offset_error.png')
            else:
                plt.savefig('Error_Plots/Offset_error.png')
        if show_fig:
            # plt.show()
            pass
        # plt.close()

    if show_fig:
        plt.show()

    # Note: got rid of the tau plots (I will use Wynter's Matlab code for this)


# Need to comment this out when using
CORY48 = [1,3,0,2,3,0,1,3,0,1,3,0,1,4,0,1,3,0,4,2,0,3,2,0,4,2,0,4,2,0,4,1,0,4,2,0,2,3,0,2,4,0,2,3,0,1,4,0,2,4,0,1,4,0,3,2,0,3,1,0,3,2,0,4,1,0,3,1,0,4,1,0]
yxx24 = [4, 1, 2, 3, 2, 2, 3, 2, 1, 4, 1, 1, 3, 2, 1, 4, 1, 1, 4, 1, 2, 3, 2, 2]
new = [1, 4, 2, 4, 3, 1, 4, 2, 4, 1, 2, 4, 2, 3, 1, 3, 2, 3, 1, 3, 1, 4, 1, 3, 4, 1, 4, 2, 3, 1, 2, 4, 2, 3, 3, 2, 3, 1, 3, 2, 1, 4, 4, 1, 4, 2, 3, 2]
new = [3, 2, 0, 3, 2, 0, 1, 3, 0, 1, 4, 0, 2, 3, 0, 2, 4, 0, 4, 2, 0, 3, 2, 0, 3, 2, 0, 2, 3, 0, 1, 3, 0, 2, 3, 0, 2, 4, 0, 2, 3, 0, 2, 4, 0, 1, 3, 0, 1, 4, 0, 4, 2, 0, 4, 2, 0, 4, 1, 0, 3, 1, 0, 3, 1, 0, 3, 1, 0, 4, 1, 0]

ps_1259819_72_SED_15144 = [4, 1, 0, 3, 1, 0, 1, 3, 0, 1, 3, 0, 2, 4, 0, 1, 3, 0, 4, 2, 0, 3, 1, 0, 3, 1, 0, 2, 3, 0, 2, 4, 0, 1, 4, 0, 1, 4, 0, 4, 2, 0, 2, 4, 0, 1, 4, 0, 3, 2, 0, 3, 2, 0, 3, 2, 0, 4, 2, 0, 2, 3, 0, 2, 3, 0, 4, 1, 0, 4, 1, 0,
       4, 1, 0, 3, 1, 0, 1, 3, 0, 1, 3, 0, 2, 4, 0, 1, 3, 0, 4, 2, 0, 3, 1, 0, 3, 1, 0, 2, 3, 0, 2, 4, 0, 1, 4, 0, 1, 4, 0, 4, 2, 0, 2, 4, 0, 1, 4, 0, 3, 2, 0, 3, 2, 0, 3, 2, 0, 4, 2, 0, 2, 3, 0, 2, 3, 0, 4, 1, 0, 4, 1, 0,
       4, 1, 0, 3, 1, 0, 1, 3, 0, 1, 3, 0, 2, 4, 0, 1, 3, 0, 4, 2, 0, 3, 1, 0, 3, 1, 0, 2, 3, 0, 2, 4, 0, 1, 4, 0, 1, 4, 0, 4, 2, 0, 2, 4, 0, 1, 4, 0, 3, 2, 0, 3, 2, 0, 3, 2, 0, 4, 2, 0, 2, 3, 0, 2, 3, 0, 4, 1, 0, 4, 1, 0,
       4, 1, 0, 3, 1, 0, 1, 3, 0, 1, 3, 0, 2, 4, 0, 1, 3, 0, 4, 2, 0, 3, 1, 0, 3, 1, 0, 2, 3, 0, 2, 4, 0, 1, 4, 0, 1, 4, 0, 4, 2, 0, 2, 4, 0, 1, 4, 0, 3, 2, 0, 3, 2, 0, 3, 2, 0, 4, 2, 0, 2, 3, 0, 2, 3, 0, 4, 1, 0, 4, 1, 0]

ps_1259819_72_SED_15144 = [4, 1, 0, 3, 1, 0, 1, 3, 0, 1, 3, 0, 2, 4, 0, 1, 3, 0, 4, 2, 0, 3, 1, 0, 3, 1, 0, 2, 3, 0, 2, 4, 0, 1, 4, 0, 1, 4, 0, 4, 2, 0, 2, 4, 0, 1, 4, 0, 3, 2, 0, 3, 2, 0, 3, 2, 0, 4, 2, 0, 2, 3, 0, 2, 3, 0, 4, 1, 0, 4, 1, 0]


no_err_trial = [2, 4, 2, 3, 2, 4, 3, 1, 4, 1, 1, 4, 3, 2, 3, 1, 1, 3, 4, 2, 3, 2, 4, 2]
print(len(no_err_trial))

# Sequences with maxed out fidelities
best1 = [2, 4, 2, 3, 2, 4, 3, 1, 4, 1, 1, 4, 3, 2, 3, 1, 1, 3, 4, 2, 3, 2, 4, 2]
best2 = [3, 2, 4, 2, 4, 2, 2, 4, 1, 4, 1, 3, 3, 1, 3, 1, 2, 3, 3, 2, 4, 1, 1, 3]
best3 = [4, 2, 3, 2, 2, 4, 2, 3, 2, 4, 3, 2, 3, 1, 4, 1, 1, 3, 4, 1, 3, 1, 1, 3]
best4 = [4, 2, 3, 2, 4, 2, 2, 4, 2, 3, 2, 4, 3, 1, 4, 1, 1, 3, 4, 1, 3, 1, 1, 3]
best5 = [4, 2, 3, 2, 2, 4, 2, 3, 2, 4, 3, 2, 2, 3, 1, 3, 3, 1, 1, 3, 1, 4, 4, 1]
print(len(best1))


SED_2 = [3, 1, 0, 1, 3, 0, 2, 3, 0, 1, 3, 0, 3, 1, 0, 3, 2, 0, 4, 2, 0, 1, 4, 0, 2, 4, 0, 1, 4, 0, 4, 2, 0, 4, 1, 0,
         3, 2, 0, 1, 3, 0, 1, 3, 0, 2, 3, 0, 4, 2, 0, 4, 1, 0, 4, 2, 0, 4, 1, 0, 2, 3, 0, 2, 4, 0, 2, 4, 0, 3, 1, 0]

SED48 = [2, 4, 0, 2, 3, 0, 2, 3, 0, 4, 2, 0, 3, 2, 0, 4, 2, 0, 1, 3, 0, 1, 4, 0, 1, 4, 0, 3, 1, 0, 4, 1, 0, 3, 1, 0,
         1, 3, 0, 3, 1, 0, 3, 1, 0, 4, 1, 0, 2, 3, 0, 2, 4, 0, 2, 4, 0, 2, 3, 0, 4, 1, 0, 4, 2, 0, 4, 2, 0, 1, 3, 0]


# Better than CORY48 all around
ps_1678985_36_SED_2167 = [3, 2, 0, 3, 1, 0, 1, 3, 0, 1, 3, 0, 2, 4, 0, 2, 4, 0, 4, 2, 0, 4, 1, 0, 1, 4, 0, 4, 2, 0, 3, 1, 0, 1, 4, 0]
print(len(ps_1678985_36_SED_2167+SED_2))

ps_2032111_72_SED_4863 = [1, 4, 0, 1, 3, 0, 1, 4, 0, 2, 4, 0, 2, 3, 0, 2, 3, 0, 2, 4, 0, 4, 2, 0, 1, 3, 0, 1, 4, 0, 1, 3, 0, 4, 2, 0, 3, 2, 0, 3, 1, 0, 3, 2, 0, 2, 3, 0, 3, 2, 0, 4, 1, 0, 4, 2, 0, 2, 4, 0, 4, 1, 0, 3, 1, 0, 3, 1, 0, 4, 1, 0]

ps_1750484_36_SED_561 = [2, 3, 0, 4, 2, 0, 3, 2, 0, 1, 3, 0, 4, 1, 0, 2, 4, 0, 1, 4, 0, 1, 4, 0, 3, 2, 0, 2, 4, 0, 3, 1, 0, 3, 1, 0]
ps_1750484_36_SED_4202 = [3, 1, 0, 3, 2, 0, 2, 3, 0, 4, 2, 0, 4, 1, 0, 1, 4, 0, 1, 3, 0, 2, 4, 0, 1, 4, 0, 3, 2, 0, 4, 2, 0, 1, 3, 0]
ps_2032111_72_SED_91958 = [3, 2, 0, 4, 1, 0, 1, 4, 0, 2, 4, 0, 1, 3, 0, 3, 2, 0, 2, 4, 0, 1, 4, 0, 1, 3, 0, 3, 2, 0, 3, 1, 0, 2, 3, 0, 1, 3, 0, 1, 4, 0, 3, 1, 0, 4, 1, 0, 2, 3, 0, 4, 1, 0, 3, 1, 0, 4, 2, 0, 2, 4, 0, 2, 3, 0, 4, 2, 0, 4, 2, 0]
comp = [2, 4, 0, 2, 3, 0, 2, 3, 0, 4, 2, 0, 3, 2, 0, 4, 2, 0, 1, 3, 0, 1, 4, 0, 1, 4, 0, 3, 1, 0, 4, 1, 0, 3, 1, 0, 2, 4, 0, 2, 3, 0, 2, 3, 0, 4, 2, 0, 3, 2, 0, 4, 2, 0, 1, 3, 0, 1, 4, 0, 1, 4, 0, 3, 1, 0, 4, 1, 0, 3, 1, 0, 1, 3, 0, 3, 1, 0, 3, 1, 0, 4, 1, 0, 2, 3, 0, 2, 4, 0, 2, 4, 0, 2, 3, 0, 4, 1, 0, 4, 2, 0, 4, 2, 0, 1, 3, 0, 1, 3, 0, 3, 1, 0, 3, 1, 0, 4, 1, 0, 2, 3, 0, 2, 4, 0, 2, 4, 0, 2, 3, 0, 4, 1, 0, 4, 2, 0, 4, 2, 0, 1, 3, 0]

SED96 = ps_1750484_36_SED_561 + ps_1750484_36_SED_4202 + SED_2
print(len(SED96))

#(4*CORY48, 'CORY48'), (12*yxx24, 'yxx24'),
# # Commented out 5/10/22
# error_plots([(4*CORY48, 'CORY48'), (12*yxx24, 'yxx24'), (8*ps_1678985_36_SED_2167, '1678985_36_SED_2167'),
#              (4*SED48, 'SED48')], 'all',
#             show_fig=True, save_fig=False, granular=True, pulse_width=0)

# error_plots([(4*CORY48, 'CORY48'), (4*SED_2, 'SED_2'),
#              (4 * (ps_1750484_36_SED_561 + ps_1750484_36_SED_4202), 'Both'),
#              (2*comp, 'SED48'),
#              (2 * (ps_1750484_36_SED_561 + ps_1750484_36_SED_4202 + SED_2), 'All 3')], 'all',
#             show_fig=True, save_fig=False, granular=True, pulse_width=0)

# error_plots([(4*CORY48, 'CORY48'), (4*SED_2, 'SED48_2'), (4*SED48, 'SED48'),
#              (2 * (ps_1750484_36_SED_561 + ps_1750484_36_SED_4202 + SED_2), '3x Combo')], 'all',
#             show_fig=True, save_fig=False, granular=True, pulse_width=0)

# error_plots([(3*CORY48, 'CORY48'), (9*yxx24, 'yxx24'), (2*(ps_1678985_36_SED_2167+SED_2), 'Both'),
#              (6 * ps_1678985_36_SED_2167, 'ps_1678985_36_SED_2167'),
#              (3 * SED_2, 'SED_2'), (3 * SED48, 'SED48')],
#             'all',
#             show_fig=True, save_fig=False, granular=True, pulse_width=0)

# PTE 0.01 > 6 (5/10/22)
a = [3, 1, 0, 1, 3, 0, 2, 3, 0, 1, 3, 0, 2, 3, 0, 4, 1, 0, 4, 2, 0, 2, 4, 0, 2, 3, 0, 1, 3, 0, 4, 2, 0, 1, 4, 0, 4, 1, 0, 2, 4, 0, 2, 4, 0, 1, 4, 0, 4, 2, 0, 3, 1, 0, 3, 1, 0, 3, 2, 0, 3, 2, 0, 1, 4, 0, 3, 2, 0, 4, 1, 0]
b = [4, 1, 0, 2, 4, 0, 4, 1, 0, 2, 4, 0, 1, 3, 0, 1, 4, 0, 3, 2, 0, 4, 1, 0, 4, 2, 0, 1, 3, 0, 2, 4, 0, 1, 3, 0, 1, 4, 0, 2, 3, 0, 3, 2, 0, 1, 4, 0, 2, 3, 0, 4, 2, 0, 3, 1, 0, 3, 2, 0, 4, 2, 0, 2, 3, 0, 3, 1, 0, 3, 1, 0]
c = [3, 2, 0, 4, 2, 0, 4, 1, 0, 2, 3, 0, 1, 4, 0, 1, 4, 0, 2, 4, 0, 4, 1, 0, 3, 2, 0, 3, 2, 0, 1, 3, 0, 2, 3, 0, 2, 3, 0, 1, 3, 0, 2, 3, 0, 1, 3, 0, 1, 4, 0, 3, 1, 0, 4, 2, 0, 4, 2, 0, 3, 1, 0, 3, 1, 0, 4, 1, 0, 2, 4, 0]
d = [2, 3, 0, 2, 3, 0, 3, 1, 0, 3, 1, 0, 4, 1, 0, 1, 4, 0, 2, 4, 0, 3, 2, 0, 4, 2, 0, 1, 4, 0, 4, 1, 0, 4, 1, 0, 4, 1, 0, 1, 4, 0, 1, 3, 0, 2, 4, 0, 4, 2, 0, 4, 2, 0, 1, 3, 0, 1, 3, 0, 2, 4, 0, 2, 3, 0, 3, 1, 0, 3, 2, 0]
e = [1, 3, 0, 3, 1, 0, 3, 2, 0, 4, 2, 0, 2, 3, 0, 3, 2, 0, 3, 2, 0, 2, 4, 0, 2, 4, 0, 1, 4, 0, 1, 4, 0, 1, 3, 0, 2, 3, 0, 2, 3, 0, 2, 3, 0, 1, 3, 0, 4, 2, 0, 4, 2, 0, 3, 1, 0, 4, 1, 0, 3, 1, 0, 4, 1, 0, 4, 1, 0, 2, 4, 0]
f = [1, 4, 0, 3, 1, 0, 2, 4, 0, 1, 3, 0, 2, 4, 0, 1, 3, 0, 1, 4, 0, 1, 3, 0, 4, 2, 0, 3, 1, 0, 4, 1, 0, 2, 4, 0, 1, 4, 0, 3, 1, 0, 2, 3, 0, 2, 3, 0, 4, 1, 0, 4, 2, 0, 4, 2, 0, 4, 1, 0, 2, 3, 0, 3, 2, 0, 3, 2, 0, 3, 2, 0]
g = [3, 2, 0, 3, 2, 0, 1, 3, 0, 2, 3, 0, 4, 2, 0, 4, 1, 0, 4, 2, 0, 1, 3, 0, 2, 4, 0, 4, 1, 0, 3, 2, 0, 4, 2, 0, 4, 2, 0, 4, 1, 0, 2, 3, 0, 4, 1, 0, 3, 2, 0, 1, 4, 0, 1, 4, 0, 2, 3, 0, 2, 3, 0, 1, 3, 0, 2, 4, 0, 1, 4, 0]
h = [3, 2, 0, 3, 1, 0, 1, 4, 0, 4, 2, 0, 4, 2, 0, 4, 1, 0, 2, 4, 0, 4, 2, 0, 2, 3, 0, 1, 3, 0, 1, 4, 0, 1, 4, 0, 1, 3, 0, 2, 4, 0, 4, 1, 0, 3, 1, 0, 4, 1, 0, 2, 4, 0, 1, 4, 0, 1, 3, 0, 3, 2, 0, 3, 2, 0, 2, 3, 0, 3, 1, 0]
i = [1, 3, 0, 2, 4, 0, 3, 2, 0, 3, 2, 0, 3, 1, 0, 2, 3, 0, 1, 3, 0, 2, 4, 0, 3, 1, 0, 4, 2, 0, 4, 2, 0, 1, 4, 0]

seqs = [a, b, c, d, e, f, g, h]
# error_plots([(4*CORY48, 'CORY48'), (4*ps_1259819_72_SED_15144, '1259819_72_SED_15144'),
#              (4*SED48, 'SED48'), (4*a, 'a'),
#              (4 * f, 'f'),
#              (4 * g, 'g'),
#              (4 * h, 'h'),
#              (8 * i, 'i')],
#             'all',
#             show_fig=True, save_fig=False, granular=True, pulse_width=0)

# error_plots([(4*CORY48, 'CORY48'), (4*ps_1259819_72_SED_15144, '1259819_72_SED_15144'),
#              (4*SED48, 'SED48'), (4*a, 'a'),
#              (4 * b, 'b'),
#              (4 * c, 'c'),
#              (4 * d, 'd'),
#              (4 * e, 'e')],
#             'all',
#             show_fig=True, save_fig=False, granular=True, pulse_width=0)

CORY48_se = [1,3,2,3,1,3,1,3,1,4,1,3,4,2,3,2,4,2,4,2,4,1,4,2,2,3,2,4,2,3,1,4,2,4,1,4,3,2,3,1,3,2,4,1,3,1,4,1]
# CORY48_se = [1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
bad = [1, 3, 0]

SED48_se = [2, 4, 2, 3, 2, 3, 4, 2, 3, 2, 4, 2, 1, 3, 1, 4, 1, 4, 3, 1, 4, 1, 3, 1,
            1, 3, 3, 1, 3, 1, 4, 1, 2, 3, 2, 4, 2, 4, 2, 3, 4, 1, 4, 2, 4, 2, 1, 3]

# error_plots([(4*SED48, 'SED48 SED'), (6*SED48_se, 'SE48 SE')],
#             'all',
#             show_fig=True, save_fig=False, granular=True, pulse_width=0)
# error_plots([(bad, 'bad'), (CORY48_se, 'CORY48 SE')],
#             'all',
#             show_fig=True, save_fig=False, granular=True, pulse_width=0)

# error_plots([(4*CORY48, 'CORY48'), (4*SED_2, 'SED48_2'), (4*SED48, 'SED48'),
#              (2 * (ps_1750484_36_SED_561 + ps_1750484_36_SED_4202 + SED_2), '3x Combo'),
#              (8 * i, 'i')],
#             'all',
#             show_fig=True, save_fig=False, granular=True, pulse_width=0)

# rot 0.05 > 4 (5/10/22)
a = [3, 2, 0, 4, 1, 0, 4, 1, 0, 1, 3, 0, 1, 4, 0, 1, 3, 0, 4, 2, 0, 3, 2, 0, 4, 2, 0, 2, 4, 0, 1, 4, 0, 2, 4, 0]
b = [4, 1, 0, 2, 4, 0, 1, 4, 0, 2, 4, 0, 3, 2, 0, 3, 1, 0, 4, 1, 0, 4, 2, 0, 1, 3, 0, 1, 4, 0, 1, 3, 0, 3, 1, 0]
c = [3, 1, 2, 3, 1, 3, 4, 2, 1, 4, 2, 3, 1, 4, 3, 1, 4, 1, 1, 4, 2, 3, 2, 4, 1, 3, 2, 4, 4, 1, 4, 2, 4, 1, 2, 4, 3, 2, 3, 1, 4, 2, 3, 2, 3, 2, 1, 3]

# > 3
c = [2, 3, 0, 3, 2, 0, 4, 2, 0, 2, 4, 0, 4, 1, 0, 3, 1, 0, 4, 1, 0, 1, 3, 0, 4, 2, 0, 1, 3, 0, 2, 4, 0, 2, 4, 0, 1, 4, 0, 2, 3, 0, 4, 2, 0, 3, 1, 0, 4, 1, 0, 4, 2, 0, 1, 3, 0, 3, 1, 0, 1, 4, 0, 2, 3, 0, 3, 2, 0, 1, 4, 0]
d = [4, 1, 0, 2, 3, 0, 2, 3, 0, 4, 2, 0, 4, 2, 0, 3, 2, 0, 1, 4, 0, 3, 2, 0, 4, 2, 0, 1, 4, 0, 1, 4, 0, 3, 2, 0, 2, 3, 0, 1, 3, 0, 3, 1, 0, 1, 3, 0, 1, 3, 0, 4, 1, 0, 4, 1, 0, 2, 4, 0, 2, 4, 0, 2, 4, 0, 3, 1, 0, 3, 1, 0]

e = [3, 2, 0, 2, 3, 0, 1, 3, 0, 2, 3, 0, 3, 2, 0, 3, 1, 0, 4, 1, 0, 3, 1, 0, 4, 1, 0, 1, 4, 0, 2, 4, 0, 1, 4, 0]

# # Set angle error to ±0.5 rather than ±0.1 to see robustness
# error_plots([(4*CORY48, 'CORY48'), (4*SED_2, 'SED48_2'), (4*SED48, 'SED48'),
#              (2 * (ps_1750484_36_SED_561 + ps_1750484_36_SED_4202 + SED_2), '3x Combo'),
#              (8 * b, 'b'),
#              (8 * e, 'e')],
#             'angle',
#             show_fig=True, save_fig=False, granular=True, pulse_width=0)

test_SE =  [4,	6	,1,	3,	2,	5,	4,	5,	2,	3,	1,	 6] # in SED action space 
test_0 = convert(test_SE,'SED','O')

#test_0 =  [3,1,0,	4,1,0	,1,4,0,	2,4,0,	2,3,0,	3,2,0,	3,1,0,	3,2,0,	2,3,0,	2,4,0,	1,3,0,	 4,1,0] # in primitive action space 

SED24 = i

#print(len(4*CORY48), len(12*yxx24), len(8*SED24), len(4*SED48), len(2*SED96))
#print(SED24)
#print(SED48)
#print(SED96)

# # Great offset robustness
# (12 * [1, 1, 4, 1, 1, 2, 2, 4, 3, 3, 2, 3, 4, 2, 3, 3, 3, 1, 4, 4, 2, 2, 4, 1], 'New'),
#              (4 * [4, 2, 0, 4, 1, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 4, 2, 0, 4, 2, 0, 2, 4, 0, 2, 4, 0, 1, 4, 0, 4, 1, 0, 3, 1, 0, 2, 3, 0, 2, 3, 0, 2, 4, 0, 1, 4, 0, 1, 4, 0, 3, 1, 0, 3, 1, 0, 3, 2, 0, 3, 2, 0, 4, 1, 0, 3, 2, 0, 2, 3, 0], 'New2')

folder = '/Users/admin/Desktop/RL Hamiltonian/OwenE'
#error_plots([(4 * CORY48, 'CORY48'),
#             (12 * yxx24, 'yxx24'),
#             (8 * SED24, 'SED24'),
#             (4 * SED48, 'SED48'),
#             (2 * SED96, 'SED96'),
#             ],
#            'all',
#            show_fig=True, save_fig=False, granular=True, pulse_width=0, folder=folder, name='CB_Best')
error_plots([(4*test_0, 'test'),(3*CORY48,'CORY48'),
             ],
            'all',
            show_fig=True, save_fig=False, granular=False, pulse_width=0, folder=folder, name='CB_Best')

az_all_err_12 = [
    4, 2, 3, 3, 2, 3, 3, 2, 4, 4, 2, 4
]
az_all_err_24 = [
    4, 4, 2, 4, 4, 2, 3, 2, 3, 3, 2, 3,
    1, 3, 1, 1, 3, 1, 4, 4, 1, 4, 4, 1
]
az_all_err_48 = [
    1, 1, 4, 1, 1, 4, 2, 4, 4, 2, 4, 4,
    2, 4, 4, 2, 4, 4, 3, 3, 3, 2, 3, 3,
    2, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 2,
    1, 4, 4, 1, 4, 4, 4, 2, 2, 4, 2, 2
]

# error_plots([(4 * CORY48, 'CORY48'),
#              (24 * az_all_err_12, 'az12'),
#              (12 * az_all_err_24, 'az24'),
#              (6 * az_all_err_48, 'az48')
#              ],
#             'all',
#             show_fig=False, save_fig=True, granular=False, pulse_width=0, folder=folder, name='CB_Orig')
