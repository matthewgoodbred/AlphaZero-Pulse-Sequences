'''
5/3/22
Owen Eskandari

Thesis Work 22S

This is adding all fidelity values to a PSS object
For smaller data sets

Updated 7/26/22 to add finite pulse width fidelities to all pulse sequences (and save as a separate PSS object)
'''


import pickle
import qutip as qt
import numpy as np
import pulse_sequences_not_stochastic as ps
from datetime import datetime
from Counter import counter


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


# Test the validity of the reward given to a sequence in a run
def get_propagators(sequence, ps_config):
    if len(sequence) == 0:
        return ([qt.identity(ps_config.Utarget.dims[0])]
                * ps_config.ensemble_size)
    else:
        propagators = get_propagators(sequence[:-1], ps_config)
        propagators = [prop.copy() for prop in propagators]
        for s in range(ps_config.ensemble_size):
            propagators[s] = (ps_config.pulses_ensemble[s][sequence[-1]].get_pulse() * propagators[s])
        return propagators


def get_reward(sequence, ps_config):
    propagators = get_propagators(sequence, ps_config)
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


current_file_with_PSS_data = 'PSS_All_Sequences_20220429_162015.pkl'      # TODO Change

with open(current_file_with_PSS_data, 'rb') as f:
    pss = pickle.load(f)

# Parameters
N = 4
ensemble_size = 1
dipolar_strength = 1e2
pulse_width = 2e-5     # 0 is infinitesimal pulses TODO: changed to finite pulse widths
delay = 1e-4
rot_error = 0
pte = 0
offset = 0
pulse_list = [['D'], ['X'], ['-X'], ['Y'], ['-Y']]
stochastic = False
Utarget = qt.identity([2] * N)

# Configurations for all data points
config_0 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                  ensemble_size=ensemble_size,
                                  max_sequence_length=48,      # Does this matter?
                                  dipolar_strength=dipolar_strength,
                                  pulse_width=pulse_width,
                                  delay=delay,
                                  rot_error=rot_error,
                                  phase_transient_error=pte,
                                  offset_error=offset,
                                  pulse_list=pulse_list,
                                  stochastic=stochastic)

config_1 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                  ensemble_size=ensemble_size,
                                  max_sequence_length=48,      # Does this matter?
                                  dipolar_strength=dipolar_strength,
                                  pulse_width=pulse_width,
                                  delay=delay,
                                  rot_error=0.025,
                                  phase_transient_error=pte,
                                  offset_error=offset,
                                  pulse_list=pulse_list,
                                  stochastic=stochastic)

config_2 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                  ensemble_size=ensemble_size,
                                  max_sequence_length=48,      # Does this matter?
                                  dipolar_strength=dipolar_strength,
                                  pulse_width=pulse_width,
                                  delay=delay,
                                  rot_error=0.05,
                                  phase_transient_error=pte,
                                  offset_error=offset,
                                  pulse_list=pulse_list,
                                  stochastic=stochastic)

config_3 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                  ensemble_size=ensemble_size,
                                  max_sequence_length=48,      # Does this matter?
                                  dipolar_strength=dipolar_strength,
                                  pulse_width=pulse_width,
                                  delay=delay,
                                  rot_error=0.075,
                                  phase_transient_error=pte,
                                  offset_error=offset,
                                  pulse_list=pulse_list,
                                  stochastic=stochastic)

config_4 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                  ensemble_size=ensemble_size,
                                  max_sequence_length=48,      # Does this matter?
                                  dipolar_strength=dipolar_strength,
                                  pulse_width=pulse_width,
                                  delay=delay,
                                  rot_error=0.1,
                                  phase_transient_error=pte,
                                  offset_error=offset,
                                  pulse_list=pulse_list,
                                  stochastic=stochastic)

config_5 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                  ensemble_size=ensemble_size,
                                  max_sequence_length=48,      # Does this matter?
                                  dipolar_strength=dipolar_strength,
                                  pulse_width=pulse_width,
                                  delay=delay,
                                  rot_error=rot_error,
                                  phase_transient_error=1e-4,
                                  offset_error=offset,
                                  pulse_list=pulse_list,
                                  stochastic=stochastic)

config_6 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                  ensemble_size=ensemble_size,
                                  max_sequence_length=48,      # Does this matter?
                                  dipolar_strength=dipolar_strength,
                                  pulse_width=pulse_width,
                                  delay=delay,
                                  rot_error=rot_error,
                                  phase_transient_error=1e-3,
                                  offset_error=offset,
                                  pulse_list=pulse_list,
                                  stochastic=stochastic)

config_7 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                  ensemble_size=ensemble_size,
                                  max_sequence_length=48,      # Does this matter?
                                  dipolar_strength=dipolar_strength,
                                  pulse_width=pulse_width,
                                  delay=delay,
                                  rot_error=rot_error,
                                  phase_transient_error=1e-2,
                                  offset_error=offset,
                                  pulse_list=pulse_list,
                                  stochastic=stochastic)

config_8 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                  ensemble_size=ensemble_size,
                                  max_sequence_length=48,      # Does this matter?
                                  dipolar_strength=dipolar_strength,
                                  pulse_width=pulse_width,
                                  delay=delay,
                                  rot_error=rot_error,
                                  phase_transient_error=2e-2,
                                  offset_error=offset,
                                  pulse_list=pulse_list,
                                  stochastic=stochastic)

config_9 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                  ensemble_size=ensemble_size,
                                  max_sequence_length=48,      # Does this matter?
                                  dipolar_strength=dipolar_strength,
                                  pulse_width=pulse_width,
                                  delay=delay,
                                  rot_error=rot_error,
                                  phase_transient_error=pte,
                                  offset_error=-1e3,
                                  pulse_list=pulse_list,
                                  stochastic=stochastic)

config_10 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                   ensemble_size=ensemble_size,
                                   max_sequence_length=48,      # Does this matter?
                                   dipolar_strength=dipolar_strength,
                                   pulse_width=pulse_width,
                                   delay=delay,
                                   rot_error=rot_error,
                                   phase_transient_error=pte,
                                   offset_error=-1e2,
                                   pulse_list=pulse_list,
                                   stochastic=stochastic)

config_11 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                   ensemble_size=ensemble_size,
                                   max_sequence_length=48,      # Does this matter?
                                   dipolar_strength=dipolar_strength,
                                   pulse_width=pulse_width,
                                   delay=delay,
                                   rot_error=rot_error,
                                   phase_transient_error=pte,
                                   offset_error=-1e1,
                                   pulse_list=pulse_list,
                                   stochastic=stochastic)

config_12 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                   ensemble_size=ensemble_size,
                                   max_sequence_length=48,      # Does this matter?
                                   dipolar_strength=dipolar_strength,
                                   pulse_width=pulse_width,
                                   delay=delay,
                                   rot_error=rot_error,
                                   phase_transient_error=pte,
                                   offset_error=-1e0,
                                   pulse_list=pulse_list,
                                   stochastic=stochastic)

config_13 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                   ensemble_size=ensemble_size,
                                   max_sequence_length=48,      # Does this matter?
                                   dipolar_strength=dipolar_strength,
                                   pulse_width=pulse_width,
                                   delay=delay,
                                   rot_error=rot_error,
                                   phase_transient_error=pte,
                                   offset_error=1e0,
                                   pulse_list=pulse_list,
                                   stochastic=stochastic)

config_14 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                   ensemble_size=ensemble_size,
                                   max_sequence_length=48,      # Does this matter?
                                   dipolar_strength=dipolar_strength,
                                   pulse_width=pulse_width,
                                   delay=delay,
                                   rot_error=rot_error,
                                   phase_transient_error=pte,
                                   offset_error=1e1,
                                   pulse_list=pulse_list,
                                   stochastic=stochastic)

config_15 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                   ensemble_size=ensemble_size,
                                   max_sequence_length=48,      # Does this matter?
                                   dipolar_strength=dipolar_strength,
                                   pulse_width=pulse_width,
                                   delay=delay,
                                   rot_error=rot_error,
                                   phase_transient_error=pte,
                                   offset_error=1e2,
                                   pulse_list=pulse_list,
                                   stochastic=stochastic)

config_16 = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                   ensemble_size=ensemble_size,
                                   max_sequence_length=48,      # Does this matter?
                                   dipolar_strength=dipolar_strength,
                                   pulse_width=pulse_width,
                                   delay=delay,
                                   rot_error=rot_error,
                                   phase_transient_error=pte,
                                   offset_error=1e3,
                                   pulse_list=pulse_list,
                                   stochastic=stochastic)

configs = [config_0, config_1, config_2, config_3, config_4, config_5, config_6, config_7, config_8,
           config_9, config_10, config_11, config_12, config_13, config_14, config_15, config_16]

sequences = pss.get_ps_obj_list()

small_rot_err_reward_dict = {}

for idx, ps_obj in enumerate(sequences):

    # Assume all sequences are of the correct length (reward != -1)
    for config in configs:
        config.reset()

    short_sequence = ps_obj.get_ps()
    multiplier = 288 // ps_obj.get_length()
    pulse_sequence = multiplier * short_sequence  # Normalize length of sequence to 288 tau

    if len(pulse_sequence) == 288:
        for pulse in pulse_sequence:
            for config in configs:
                config.apply(pulse)
        rewards = []
        for config in configs:
            rewards.append(get_reward(pulse_sequence, config))

        no_err_info = rewards[0]
        rot_err_info = rewards[0:5]
        pte_err_info = [rewards[0], rewards[5], rewards[6], rewards[7], rewards[8]]
        offset_info = [rewards[9], rewards[10], rewards[11], rewards[12], rewards[0],
                       rewards[13], rewards[14], rewards[15], rewards[16]]
        ps_obj.set_no_error_fid(no_err_info)
        ps_obj.set_rot_error_fids(rot_err_info)
        ps_obj.set_pte_fids(pte_err_info)
        ps_obj.set_offset_fids(offset_info)

    if idx % 1000 == 0:
        counter(idx, len(sequences))


# Save the output
outf = 'PSS_with_Finite_Width_Fids_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.pkl'
print(outf)
save_object(pss, outf)     # TODO: Don't want to accidentally save again

