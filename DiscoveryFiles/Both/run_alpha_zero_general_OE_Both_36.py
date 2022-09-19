# Added histogram interface 2/3
# Ran into issues with histogram interface 2/7...added try statements to work around for now; come back later

from datetime import datetime
import random
from time import sleep
import qutip as qt
import sys
import os
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# New imports
import matplotlib.pyplot as plt
import pickle
import numpy as np
import json

sys.path.append(os.path.abspath('..'))

import alpha_zero_general as az
import pulse_sequences_general as ps     # TODO

collect_no_net_procs = 0
collect_no_net_count = 0
collect_procs = 14                       # TODO

buffer_size = int(1e6)
batch_size = 2048
num_iters = int(1e6)

max_sequence_length = 36

print_every = 100
save_every = 250
write_every = 5000    # Write count_dict and reward_dict to files every x iterations (usually int(1e4))

reward_threshold = 3		# Something to consider upping/lowering depending on run (originally 3)

dipolar_strength = 1e2		# Something to consider changing
pulse_width = 2e-5
delay = 1e-4
N = 3
ensemble_size = 50
rot_error = 1e-2						# All errors for this run
phase_transient_error = 1e-4			# All errors for this run
offset_error = 1e1  					# All errors for this run


Utarget = qt.identity([2] * N)

# # SE Pulse List (solid echo)
# pulse_list = [['D'], ['X', 'Y'], ['X', '-Y'], ['-X', 'Y'], ['-X', '-Y'],
#               ['Y', 'X'], ['Y', '-X'], ['-Y', 'X'], ['-Y', '-X']]

# Both Pulse List (SE and solid echo + delay; the building blocks of CORY48)
pulse_list = [['D'], ['X', 'Y'], ['X', '-Y'], ['-X', 'Y'], ['-X', '-Y'],
              ['Y', 'X'], ['Y', '-X'], ['-Y', 'X'], ['-Y', '-X'],
              ['X', 'Y', 'D'], ['X', '-Y', 'D'], ['-X', 'Y', 'D'], ['-X', '-Y', 'D'],
              ['Y', 'X', 'D'], ['Y', '-X', 'D'], ['-Y', 'X', 'D'], ['-Y', '-X', 'D']]

# Can only be one type of delay, otherwise check get_valid_pulses() and augment accordingly
# pulse_list = [['D'], ['X'], ['-X'], ['Y'], ['-Y']]


def collect_data(proc_num, queue, net, ps_count, global_step, lock, pulse_list):
    """
    Args:
        queue (Queue): A queue to add the statistics gathered
            from the MCTS rollouts.
        ps_count (Value): A shared count of how many pulse sequences have been
            constructed so far
    """

    print(datetime.now(), f'collecting data ({proc_num})')
    config = az.Config()
    config.pb_c_init = 4       # Subject to change (start off with the original exploration value)
    ps_config = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                       ensemble_size=ensemble_size,
                                       max_sequence_length=max_sequence_length,
                                       dipolar_strength=dipolar_strength,
                                       pulse_width=pulse_width, delay=delay,
                                       rot_error=rot_error,
                                       phase_transient_error=phase_transient_error,
                                       offset_error=offset_error,
                                       pulse_list=pulse_list)
    while global_step.value < num_iters:
        ps_config.reset()
        output, out_info = az.make_sequence(config, ps_config, network=net,
                                  rng=ps_config.rng, enforce_aht_0=False,
                                  max_difference=2) #, refocus_every=6		# For now, no constraints!

        reward = out_info[-1]  # This value is not rounded; to be used in histogram
        if out_info[-1] > reward_threshold:
            out_info[-1] = np.round(out_info[-1], 2)    # This value is rounded; to be used in output
            print(datetime.now(),
                  f'candidate pulse sequence from {proc_num}',
                  out_info)

            strpsname = []      # This is a string of the pulse sequence
            for psname in out_info[0]:  # For each pulse in the sequence
                # For each pulse in the sequence, find its basic pulses {D, X, -X, Y,-Y} and add to a list
                strpsname.append(''.join([pulse for pulse in pulse_list[psname]]))
            pulses = ','.join([idx for idx in strpsname])   # Join all pulses together as a string
            print(pulses)

        # # New: this is to replicate Figure 4.1 in Will's thesis (5 lines)
        # if ps_count.value > 0:
        #
        #     # reward = out_info[1]
        #     try: total_reward = average_rewards[-1][1]*(ps_count.value-1) + reward
        #     except IndexError: total_reward = reward
        #     average_rewards.append((global_step.value, total_reward/ps_count.value))

        with lock:
            queue.put(output)
            ps_count.value += 1

            # dict_entry = 0      # Create a numerical dictionary entry for each pulse sequence
            # for pulse in out_info[0]:   # For each pulse
            #     dict_entry *= len(pulse_list)   # Multiple entry by number of actions
            #     dict_entry += pulse             # Add the value of the pulse
            # dict_entry = str(dict_entry)        # Convert to a string so that dictionary can be written to file

            dict_entry = str(out_info[0])       # No encoding, so key is str(pulse sequence)

            # Add elements to the dictionary here
            if dict_entry in reward_dict:           # If in the dictionaries
                count_dict[dict_entry] += 1         # Update the number of times this pulse sequence has been found
            else:
                count_dict.update({dict_entry: 1})  # Create an entry
                reward_dict.update({dict_entry: reward})    # Create an entry

    # # New (7 lines): Reproducing figure 4.1 from Will's thesis
    # x_vals, y_vals = [idx[0] for idx in average_rewards if idx[0] != 0], [idx[1] for idx in average_rewards if idx[0] != 0]
    # plt.plot(x_vals, y_vals)
    # plt.savefig('fig_4.1.png')
    #
    # vals = [x_vals, y_vals]
    # # Three different ways to do this
    # with open('csv.csv', 'wb') as f:
    #     pickle.dump(vals, f)     # This should convert it to the correct data type (do this if needed in the future)


def train_process(queue, net, global_step, ps_count, lock, pulse_list,
                  c_value=1e0, c_l2=1e-6):
    """
    Args:
        queue (Queue): A queue to add the statistics gathered
            from the MCTS rollouts.
        global_step (mp.managers.Value): Counter to keep track
            of training iterations
        writer (SummaryWriter): Write losses to log
    """
    start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    # create directory to store results in
    if not os.path.exists(start_time):
        os.makedirs(start_time)
    writer = SummaryWriter(start_time)
    net_optimizer = optim.Adam(net.parameters(),)

    buffer = []
    index = 0
    i = 0

    # Moved out of the while loop
    if not os.path.exists(f'{start_time}/network/'):    # Make the directory to save the network information
        os.makedirs(f'{start_time}/network/')

    if not os.path.exists(f'{start_time}/dict_info/'):    # Make the directory to save the dictionaries
        os.makedirs(f'{start_time}/dict_info/')

    if not os.path.exists(f'{start_time}/histograms/'):    # Make the directory to save the histograms
        os.makedirs(f'{start_time}/histograms/')
    
    # write network structure to tensorboard file
    tmp = torch.zeros((1, 10, len(pulse_list) + 1))
    writer.add_graph(net, tmp)
    del tmp
    
    while global_step.value < num_iters:
        # get stats from queue
        with lock:
            while not queue.empty():
                new_stats = queue.get()
                new_stats = az.convert_stats_to_tensors(new_stats, num_classes=len(pulse_list) + 1)
                for stat in new_stats:
                    if len(buffer) < buffer_size:
                        buffer.append(stat)
                    else:
                        buffer[index] = stat
                    index = index + 1 if index < buffer_size - 1 else 0

        # check if there's enough stats to start training
        if len(buffer) < batch_size:
            print(datetime.now(), 'not enough data yet, sleeping...')
            sleep(5)
            continue

        if i % save_every == 0:
            print(datetime.now(), 'saving network...')
            torch.save(net.state_dict(),
                       f'{start_time}/network/{i:07.0f}')

            if i % write_every == 0:

                # Writing dictionaries (and appropriate action space to decode keys) to output files
                with open(f'{start_time}/dict_info/pulse_counts_' + str(i) + '.csv', 'w') as f:
                    if i == 0:  # Only write the pulse list once (have to manually delete now when downloading data
                        f.write(json.dumps(pulse_list))
                    f.write(json.dumps(count_dict.copy()))
                with open(f'{start_time}/dict_info/pulse_rewards_' + str(i) + '.csv', 'w') as f:
                    if i == 0:  # Only write the pulse list once (have to manually delete now when downloading data
                        f.write(json.dumps(pulse_list))
                    f.write(json.dumps(reward_dict.copy()))

                # Making and saving a plot of frequency per pulse sequence
                # Scatter plot
                try:
                    plt.scatter([reward_dict[ps] for ps in reward_dict.keys()], [count_dict[ps] for ps in count_dict.keys()])
                    plt.title("Pulse Sequence Frequencies")     # Title
                    plt.ylabel("Count")     # Y axis is number of times each pulse sequence has appeared
                    plt.xlabel("Reward")    # X axis is reward associated with each pulse sequence
                    # Save the plot
                    plt.savefig(f'{start_time}/histograms/Rewards_by_sequence_Histogram_' + str(i) + '_Iterations')
                    plt.close()
                except ValueError:
                    print('ValueError: x and y must be the same size')
                    print('Size of x axis: ' + str(len([reward_dict[ps] for ps in reward_dict.keys()])))
                    print('Size of y axis: ' + str(len([count_dict[ps] for ps in count_dict.keys()])))

        net_optimizer.zero_grad()

        minibatch = random.sample(buffer, batch_size)
        states, probabilities, values = zip(*minibatch)
        try:probabilities = torch.stack(probabilities)
        except RuntimeError: print(probabilities)
        values = torch.stack(values)
        packed_states = az.pad_and_pack(states)

        policy_outputs, value_outputs, _ = net(packed_states)
        policy_loss = -1 / \
            len(states) * torch.sum(probabilities * torch.log(policy_outputs))
        value_loss = F.mse_loss(value_outputs, values)
        l2_reg = torch.tensor(0.)
        for param in net.parameters():
            l2_reg += torch.norm(param)
        loss = policy_loss + c_value * value_loss + c_l2 * l2_reg
        loss.backward()
        net_optimizer.step()

        writer.add_scalar('training_policy_loss',
                          policy_loss, global_step=global_step.value)
        writer.add_scalar('training_value_loss',
                          c_value * value_loss, global_step=global_step.value)
        writer.add_scalar('training_l2_reg',
                          c_l2 * l2_reg, global_step=global_step.value)

        if i % print_every == 0:
            print(datetime.now(), f'updated network (iteration {i})',
                  f'pulse_sequence_count: {ps_count.value}')
            _, _, values = zip(*list(buffer))
            values = torch.stack(values).squeeze()
            writer.add_histogram('buffer_values', values,
                                 global_step=global_step.value)
            writer.add_scalar('pulse_sequence_count', ps_count.value,
                              global_step.value)

        with lock:
            global_step.value += 1
        i += 1
        sleep(.1)


if __name__ == '__main__':
    with mp.Manager() as manager:
        queue = manager.Queue()
        global_step = manager.Value('i', 0)
        ps_count = manager.Value('i', 0)
        lock = manager.Lock()

        # # New: This is to reproduce figure 4.1 from Will's thesis. If needed, add average_rewards as a parameter
        # # to collect data and uncomment relevant lines in that function
        # average_rewards = manager.list()

        # For keeping track of pulse sequences found and for making the histogram of pulse sequences found
        reward_dict = manager.dict()
        count_dict = manager.dict()

        net = az.Network(input_size=len(pulse_list) + 1, policy_output_size=len(pulse_list))    # Correct action space
        # optionally load state dict
        # change global_step above too...
        # net.load_state_dict(torch.load('0026000-network'))
        net.share_memory()
        collectors = []
        for i in range(collect_procs):
            c = mp.Process(target=collect_data,
                           args=(i, queue, net, ps_count, global_step, lock, pulse_list))
            c.start()
            collectors.append(c)
        trainer = mp.Process(target=train_process,
                             args=(queue, net,
                                   global_step, ps_count, lock, pulse_list))
        trainer.start()
        
        for c in collectors:
            c.join()
        collectors.clear()
        
        print('all collectors are joined')
        trainer.join()
        print('trainer is joined')
        print('done!')
