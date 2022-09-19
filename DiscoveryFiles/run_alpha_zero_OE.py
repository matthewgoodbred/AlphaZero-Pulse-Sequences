from datetime import datetime
import random
from time import sleep
import qutip as qt
import sys
import os
import torch
# Same as importing multiprocessing from python
import torch.multiprocessing as mp      # Allowing for multiple threads to exist and run in parallel
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath('..'))

import alpha_zero as az      # TODO: Toggle between alpha_zero & alpha_zero_OE
import pulse_sequences as ps

"""
When in the correct directory, to run Tensorboard: tensorboard --logdir .
Then go to
http://localhost:6006/
"""

# TODO: I wonder if there's an easy way to make this file runnable in a loop with different parameters
# TODO: use sys
# Otherwise I have to update the params each time I want to run it and have to do that manually

collect_no_net_procs = 0
collect_no_net_count = 0
collect_procs = 3       # The number of parallel processors used to collect data (and run MCTS)
"14 --> 1-3 for running on laptop"

# To prevent computer from crashing (I don't want to see what would happen if number was too high)
# if collect_procs > 3:
#     input("Um might want to take a look at this unless you're on Discovery")

buffer_size = int(1e6)  # Size of the list of (state, child probs, and value) to select from when creating a batch
"1e6 --> 5e2 (for testing purposes)"
batch_size = 96         # Size of the batch (from buffer) to use to update gradients on the Neural Network net
"2048 --> 48 (trains faster)"
# The number of training iterations, not the total number of pulse sequences generated (that is ps_count)
num_iters = int(1e6)
"1e6 --> 1e4"

max_sequence_length = 12    # Target length of the pulse sequence

# TODO: maybe just print pulse sequences
print_every = 100           # Print information about the run to the console every x global steps
"100 --> 10"
save_every = 250            # Save information about the run to a file every x global steps
"250 --> 25"

reward_threshold = 3        # Minimum reward threshold to print a specific pulse sequence to the console

dipolar_strength = 1e2      # Standard deviation of dipolar coupling strengths.
pulse_width = 2e-5          # Amount of time (in seconds) that the pulse sequence is applied
delay = 1e-4                # Delay (in seconds) between pulses
N = 3                       # Number of spin 1/2s to use in each system
# Ensemble_size gives the size of the different systems you're using (they differ by 'offset' which affects H_cs)
ensemble_size = 50
"Less accurate fidelity 50 --> 10"

rot_error = 1e-2                # Standard deviation of rotation error to randomly sample from.
# Normalized magnitude of phase transient for pulses (1 is a full pi/2 pulse). Default is 0 (no error).
phase_transient_error = 1e-4
offset_error = 1e1              # Offset occurs when rf field not matched to Larmor frequency of spins

# Target propagator. We want this to be equal to I_2^( (x) N) to satisfy the cyclic condition of AHT
Utarget = qt.identity([2] * N)


# Find pulse sequences using a Monte Carlo tree search
# TODO: Still need more info about the relationship between the two things: MCTS and NN
def collect_data(proc_num, queue, net, ps_count, global_step, lock, info=True):
    """
    Args:
        proc_num (int): Number of the processor used (collect_procs gives total processors in parallel)
        queue (Queue): A queue to add the statistics gathered from the MCTS rollouts.
        net (neural network module): this is the neural network designed in the
            AZ code and graphically shown in Will's thesis
        ps_count (Value): A shared count of how many pulse sequences have been
            constructed so far
        global_step (mp.managers.Value): Counter to keep track of training iterations
            (only incremented in train_process())
        lock: Holds a lock on certain shared values such as queue, ps_count, and global_step. These shared values will
            not update unless 'with lock:' is called
        info (bool): True if you want information printed to the console. Defaults to True


    """
    if info:    # Print that you're collecting data from a specific thread
        print(datetime.now(), f'collecting data ({proc_num})')

    config = az.Config()        # Sets up basic information for the the Alpha Zero process
    # This includes parameters for self-play, MCTS, UCB (Upper Confidence Bound) formula, and training information

    # Sets up basic info about the Pulse Sequence and how it's used/implemented
    # TODO: Need to understand a little more of the quantum physics behind this before understanding it
    ps_config = ps.PulseSequenceConfig(Utarget=Utarget, N=N,
                                       ensemble_size=ensemble_size,
                                       max_sequence_length=max_sequence_length,
                                       dipolar_strength=dipolar_strength,
                                       pulse_width=pulse_width, delay=delay,
                                       rot_error=rot_error,
                                       phase_transient_error=phase_transient_error,
                                       offset_error=offset_error)

    # This thread runs until the global number of steps reaches the num_iters
    # This method continuously finds pulse sequences until training is complete (same loop in train_process())
    while global_step.value < num_iters:
        ps_config.reset()       # Reset the pulse sequence to [] (starting a new pulse sequence)

        # Since enforce_aht_0 = True, requires that equal time is spent on each axis to satisfy lowest order average H
        # The output is of the form: search_statistics = list of tuples of form (current state = list of pulse sequence,
        # child probabilities, and value up to and including target length)
        # So output[-1] = last pulse sequence generated (should have length = max_sequence_length)
        output, out_info = az.make_sequence(config, ps_config, network=net,
                                  rng=ps_config.rng, enforce_aht_0=True,
                                  max_difference=2) #, refocus_every=6

        # If the pulse sequence of the target length has a reward higher than the threshold, print this out to console
        if output[-1][2] > reward_threshold and info:
            print(datetime.now(),
                  f'candidate pulse sequence from {proc_num}',
                  out_info)       # Output[-1] contains three things: (list of pulse sequence, array, reward)
        with lock:
            # Note: each item in the queue has up to max_sequence_length items that will be put in the buffer
            queue.put(output)       # Add output to the end of the queue (of the form [(ps, child probs, reward), ...])
            ps_count.value += 1     # Increment the global pulse sequence count by one


# Train the neural network net and update the weights in each layer
def train_process(queue, net, global_step, ps_count, lock, info=True, save_info=True,
                  c_value=1e0, c_l2=1e-6):
    """
    Args:
        queue (Queue): A queue to add the statistics gathered from the MCTS rollouts.
        net (neural network module): this is the neural network designed in the
            AZ code and graphically shown in Will's thesis
        global_step (mp.managers.Value): Counter to keep track of training iterations
        ps_count (Value): A shared count of how many pulse sequences have been constructed so far
            (only incremented in collect_data())
        lock: Holds a lock on certain shared values such as queue, ps_count, and global_step. These shared values will
            not update unless 'with lock:' is called
        info (bool): True if you want information printed to the console. Defaults to True
        save_info (bool): True if you want to save information about the run to a file. Defaults to True
        c_value (float): TODO: Weighting of the value term in the loss function?
        c_l2 (float): TODO: Weighting of the L2 term in the loss function?
    """

    if save_info:
        start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        # create directory to store results in
        if not os.path.exists(start_time):
            os.makedirs(start_time)
        writer = SummaryWriter(start_time)      # writer (SummaryWriter): Write losses to log in directory start_time

    # TODO: This could be a section where I could implement step decay learning rate to try to improve loss
    net_optimizer = optim.Adam(net.parameters(),)       # Create the Adam optimizer using the network (net)
    # https://arxiv.org/pdf/1412.6980.pdf for paper on Adam (this is a form of gradient descent)

    buffer = []     # Empty list to be filled with search_statistics from the queue
    index = 0       # Index for replacing items in the buffer with ones from the end of the queue
    i = 0           # TODO: What is the purpose of having this i separate from global_step?

    if save_info:
        # write network structure to tensorboard file
        tmp = torch.zeros((1, 10, 6))   # TODO: Not sure why this is necessary
        writer.add_graph(net, tmp)      # This adds a graphical description of the model network (net) to Tensorboard
        del tmp
    
    while global_step.value < num_iters:        # Runs until the training iterations reaches num_iters
        # get stats from queue (stats are of the form [(ps, child probs, reward), ...] until len(ps) = target length)
        with lock:
            while not queue.empty():
                new_stats = queue.get()     # Blocks (waits) until an item is available, then grabs the one at the front
                new_stats = az.convert_stats_to_tensors(new_stats)      # Additionally converts ps to one hot encoding
                for stat in new_stats:
                    if len(buffer) < buffer_size:       # Note: buffer_size large (on order of 1e6)
                        buffer.append(stat)
                    else:
                        # If buffer list is full (len = buffer_size), replace items starting with the first index
                        buffer[index] = stat

                    # TODO: couldn't this be indented to save some time? Figure this out via testing
                    # (idx wouldn't need to be indexed for the first buffer_size iterations)
                    index = index + 1 if index < buffer_size - 1 else 0

        # check if there's enough stats to start training
        if len(buffer) < batch_size:        # TODO: When to start training?? 10 * batch_size? 100 * batch_size?
            if info:
                print(datetime.now(), 'not enough data yet, sleeping...')
            sleep(5)        # Sleep train_process() for 5 seconds to see if collect_data() can add enough to the queue
            continue

        if i % save_every == 0:
            if info and save_info:
                print(datetime.now(), 'saving network...')

            if save_info:
                if not os.path.exists(f'{start_time}/network/'):
                    os.makedirs(f'{start_time}/network/')
                torch.save(net.state_dict(),
                           f'{start_time}/network/{i:07.0f}')

        # TODO: more info here: https://pytorch.org/docs/stable/autograd.html#default-grad-layouts
        net_optimizer.zero_grad()       # Sets optimized gradients to zero TODO: why aren't they already zero?

        # Mini-batch [Stochastic Gradient Descent (SGD)?]
        # Elements of minibatch of type search_statistics[i] =
        # (ps in one hot encoding, tensor of child probs, tensor of value)
        minibatch = random.sample(buffer, batch_size)       # Sample a random list of batch_size from buffer
        states, probabilities, values = zip(*minibatch)     # Unzips minibatch items TODO: when did these get zipped?
        probabilities = torch.stack(probabilities)
        values = torch.stack(values)
        # This is a tensor of one-hot-encoded states, formed as if the sequences were lined from top to bottom and
        # read left to right, line by line (longest on the left-most column)
        # Ex: [0, 0, 0, 1, 2, 5, 3, 4, 1] is formed from [0, 2, 4], [0, 5], [0, 1, 3, 1] and sorted by [2, 0, 1]
        # 0 0 0
        # 1 2 5
        # 3 4
        # 1
        packed_states = az.pad_and_pack(states)     # Output is type PackedSequence (RNN modules accept this as input)

        # TODO: Key line
        # Compare output to search_stats = (ps, policy, value)
        # Note: policy_outputs already had softmax applied to it
        policy_outputs, value_outputs, _ = net(packed_states)       # Evaluate the network for the given packed_states

        # print(probabilities)
        # print(policy_outputs)
        # print(values)
        # print(value_outputs)

        # len(states) should equal batch_size
        # TODO: need to figure out what the policy_loss is to get a sense of the formula (PEMDAS)
        # TODO: need to figure out this block of code
        policy_loss = -1 / \
            len(states) * torch.sum(probabilities * torch.log(policy_outputs))  # Match this to the softmax
        # TODO: need to determine what the two inputs mean (and then refer to AZ paper equation (1))
        value_loss = F.mse_loss(value_outputs, values)      # Where do these differ? Is one always -1, 0, or 1?
        l2_reg = torch.tensor(0.)       # Initialize the L2 regularization scalar to zero (R(W))
        for param in net.parameters():       # TODO: What is this
            l2_reg += torch.norm(param)
        loss = policy_loss + c_value * value_loss + c_l2 * l2_reg       # Total loss
        loss.backward()     # Computes gradient of tensor wrt leaves
        net_optimizer.step()        # Performs one optimization step; Update all parameters after computing gradients

        if save_info:       # Saves values to the three scalars below every step
            writer.add_scalar('training_policy_loss',
                              policy_loss, global_step=global_step.value)
            writer.add_scalar('training_value_loss',
                              c_value * value_loss, global_step=global_step.value)
            writer.add_scalar('training_l2_reg',
                              c_l2 * l2_reg, global_step=global_step.value)

        if i % print_every == 0:
            if info:
                print(datetime.now(), f'updated network (iteration {i})', f'pulse_sequence_count: {ps_count.value}')
            if save_info:
                _, _, values = zip(*list(buffer))
                values = torch.stack(values).squeeze()
                writer.add_histogram('buffer_values', values,
                                     global_step=global_step.value)
                writer.add_scalar('pulse_sequence_count', ps_count.value,
                                  global_step.value)
        with lock:
            global_step.value += 1
        i += 1

        # Sleep so that the queue refills with new information
        # After len(buffer) >= batch_size, this loop will continuously run, and would go way too fast
        # (not a large enough sample size from the queue) to work well. This would effectively be running on the same
        # data over and over again
        sleep(.1)       # TODO: Why sleep for 0.1? Was this optimized?


if __name__ == '__main__':

    # Holds python objects and can control them
    with mp.Manager() as manager:
        queue = manager.Queue()                 # A queue (first in, first out) for keeping track of 'search_statistics'
        global_step = manager.Value('i', 0)     # A counter to keep track of training iterations
        ps_count = manager.Value('i', 0)        # A counter to keep track of the number of pulse sequences found
        # A lock on the above two objects; they can only be changed when this is called
        lock = manager.Lock()

        net = az.Network()      # Create the neural network according to the Network class in alpha_zero.py
        # optionally load state dict
        # change global_step above too...
        # net.load_state_dict(torch.load('0026000-network'))
        net.share_memory()
        collectors = []

        # These threads take up much more computing power than the trainer
        for i in range(collect_procs):
            # Process starts a thread running the 'target' method with 'args' arguments for that method
            # Note: this just creates the thread to be run, but does not actually start it
            c = mp.Process(target=collect_data,
                           args=(i, queue, net, ps_count, global_step, lock, True))

            # Starts the thread, and thus starts running the method targeted by the Process
            c.start()
            collectors.append(c)        # Add the thread to a list of collectors

        # Create thread for running the method train_process() to train the data
        # Much less computationally expensive than the other threads
        trainer = mp.Process(target=train_process,
                             args=(queue, net,
                                   global_step, ps_count, lock, True, False))
        trainer.start()
        
        for c in collectors:
            # TODO: I don't fully understand join()
            c.join()
        collectors.clear()
        
        print('all collectors are joined')
        trainer.join()
        print('trainer is joined')
        print('done!')
