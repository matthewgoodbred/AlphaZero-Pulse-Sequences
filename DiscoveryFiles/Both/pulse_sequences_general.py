# This file allows the user to input a custom action space in run_alpha_zero
# All parts that need to be changed are highlighted
'''
This action space is inputted in the form [action, action, ...] where action = ['pulse','pulse',...]
Example action spaces:
[['D'], ['X'], ['-X'], ['Y'], ['-Y']]
[['D'], ['X', 'Y'], ['X', '-Y'], ['-X', 'Y'], ['-X', '-Y'], ['Y', 'X'], ['Y', '-X'], ['-Y', 'X'], ['-Y', '-X']]
'''


import qutip as qt
import numpy as np
from scipy.spatial.transform import Rotation

# Notation: @ is used for matrix multiplication


# define system

def get_Hsys(N, cs_strength=1, offset=0, dipolar_strength=1e2,
             rng=None, return_all=False):
    """
    Get system Hamiltonian, defaults to strongly-coupled spin system. Units
    are normalized by the CS standard deviation (line width).
    
    Args:
        N (int): number of spins in the system
        cs_strength: Standard deviation of chemical shift strengths.
        offset (float): TODO:
        dipolar_strength (float): Standard deviation of dipolar coupling strengths.
    """
    if rng is None:
        rng = np.random.default_rng()
    # List of random chemical shifts with std = cs_strength. Each element is for one spin
    chemical_shifts = rng.normal(scale=cs_strength, size=(N,))
    # offset = rng.normal(scale=offset)

    # (x) = tensor product
    # Hcs =  sum to N of (I_2^( (x) i) (x) A*sigmaz (x) I_2^( (x) (N-i-1))); A = offset + chemical shift for ith spin
    # TODO: Does this include the resonance offset Hamiltonian H_o?
    Hcs = sum(
        [qt.tensor(
            [qt.identity(2)] * i
            + [(offset + chemical_shifts[i]) * qt.sigmaz()]
            + [qt.identity(2)] * (N - i - 1)
        ) for i in range(N)]
    )
    # dipolar interactions
    # N x N matrix of random dipolar interactions with std = dipolar_strength.
    # Each element is for one spin's interaction with another spin TODO: i's interaction with j or vice versa?
    dipolar_matrix = rng.normal(scale=dipolar_strength, size=(N, N))

    # Follows equation (21) from Linta's overview of AHT:
    # sum over i,j of dipolar strength * (3Iz^iIz^j - I^i•I^j)
    # TODO: Since I^i•I^j is frame invariant, why is it still in Hdip?
    # TODO: Is this so we don't have to add it back in at the end?
    Hdip = sum([
        dipolar_matrix[i, j] * (
            2 * qt.tensor(
                [qt.identity(2)] * i
                + [qt.sigmaz()]
                + [qt.identity(2)] * (j - i - 1)
                + [qt.sigmaz()]
                + [qt.identity(2)] * (N - j - 1)
            )
            - qt.tensor(
                [qt.identity(2)] * i
                + [qt.sigmax()]
                + [qt.identity(2)] * (j - i - 1)
                + [qt.sigmax()]
                + [qt.identity(2)] * (N - j - 1)
            )
            - qt.tensor(
                [qt.identity(2)] * i
                + [qt.sigmay()]
                + [qt.identity(2)] * (j - i - 1)
                + [qt.sigmay()]
                + [qt.identity(2)] * (N - j - 1)
            )
        )
        for i in range(N) for j in range(i + 1, N)
    ])
    if return_all:
        return Hcs + Hdip, (chemical_shifts, offset, dipolar_matrix)
    else:
        return Hcs + Hdip


def get_collective_spin(N):
    # Note: for spin_Ji(s), s = spin. Here s = 1/2; J_i(1/2) = (hbar/2)sigma_i
    # Equation: sum over i of (I^( (x) i) (x) J_axis (x) I^( (x) (N-i-1)))
    X = sum(
        [qt.tensor(
            [qt.identity(2)] * i
            + [qt.spin_Jx(1 / 2)]
            + [qt.identity(2)] * (N - i - 1)
        ) for i in range(N)]
    )
    Y = sum(
        [qt.tensor(
            [qt.identity(2)] * i
            + [qt.spin_Jy(1 / 2)]
            + [qt.identity(2)] * (N - i - 1)
        ) for i in range(N)]
    )
    Z = sum(
        [qt.tensor(
            [qt.identity(2)] * i
            + [qt.spin_Jz(1 / 2)]
            + [qt.identity(2)] * (N - i - 1)
        ) for i in range(N)]
    )
    return X, Y, Z

# pulses, pulse names, and corresponding rotations


def get_pulses(Hsys, X, Y, Z, pulse_list, pulse_width=1e-4, delay=1e-3,
               rot_error=0, phase_transient=0):
    """
    Args:
        Hsys (Qobj): Hamiltonian of the system in the toggling frame (Hsys = Hcs + Hdip)
        X (Qobj): Collective spin on X axis for N spin 1/2s
        Y (Qobj): Collective spin on Y axis for N spin 1/2s
        Z (Qobj): Collective spin on Z axis for N spin 1/2s
        pulse_list (list): List of lists of the pulses in the action space. Each element of the list is a pulse of the
            action space.
        pulse_width (float): the width of the pulse applied, in seconds
            (pulse_width = 0 is a delta\instantaneous pulses, which isn't physically possible but can serve as an
            approximation)
        delay (float): the delay in between pulses, in seconds
        rot_error: Percent error for rotations (consistent errors
            for each pulse).
        phase_transient (float): Normalized magnitude of phase transient for pulses
            (1 is a full pi/2 pulse). Default is 0 (no error).

        :return pulses (Dictionary): Returns a dictionary of pulses. Key is the string name of the pulse, and value is
            a list with two elements: the qutip representation of the pulse and the length of the pulse (units = tau)
    """
    # TODO: Is this slightly off? As in, is the total time spent on each pulse > tau?
    # TODO: Need to subtract phase transient time from the main propagator?
    Delay = qt.propagator(Hsys, pulse_width)
    X_pulse = qt.propagator(Y, np.pi / 2 * phase_transient) * \
              qt.propagator(X * (1 + rot_error) + Hsys * pulse_width / (np.pi / 2), np.pi / 2) * \
              qt.propagator(Y, np.pi / 2 * phase_transient)
    Xbar_pulse = qt.propagator(-X, np.pi / 2 * phase_transient) * \
                 qt.propagator(-X * (1 + rot_error) + Hsys * pulse_width / (np.pi / 2), np.pi / 2) * \
                 qt.propagator(-X, np.pi / 2 * phase_transient)
    Y_pulse = qt.propagator(-Y, np.pi / 2 * phase_transient) * \
              qt.propagator(Y * (1 + rot_error) + Hsys * pulse_width / (np.pi / 2), np.pi / 2) * \
              qt.propagator(-Y, np.pi / 2 * phase_transient)
    Ybar_pulse = qt.propagator(X, np.pi / 2 * phase_transient) * \
                 qt.propagator(-Y * (1 + rot_error) + Hsys * pulse_width / (np.pi / 2), np.pi / 2) * \
                 qt.propagator(X, np.pi / 2 * phase_transient)
    delay_propagator = qt.propagator(Hsys, delay)  # Equivalent to tau

    pulses = []  # List of action space. Elements are PulseInfo objects (pulse = qutip representation, length in tau)
    for pulse_name in pulse_list:
        if len(pulse_name) == 0:
            return IndexError

        string_name = ''
        pulse_length = 0
        pulse = qt.propagator(Hsys, 0)
        for subpulse in pulse_name:
            pulse_length += 1
            if subpulse == 'D':
                string_name += 'D'
                pulse = delay_propagator * Delay * pulse
            if subpulse == 'X':
                string_name += 'X'
                pulse = delay_propagator * X_pulse * pulse
            if subpulse == '-X':
                string_name += '-X'
                pulse = delay_propagator * Xbar_pulse * pulse
            if subpulse == 'Y':
                string_name += 'Y'
                pulse = delay_propagator * Y_pulse * pulse
            if subpulse == '-Y':
                string_name += '-Y'
                pulse = delay_propagator * Ybar_pulse * pulse
        pulses.append(PulseInfo(name=string_name, length=pulse_length, pulse=pulse))

    return pulses       # Return a list of PulseInfo objects


# These could maybe be put into a larger function which has the action space info
# def pulse_sequence_string(pulse_sequence):
#     """Return a string that correspond to pulse sequence
#     """
#     pulse_list = ','.join([pulse_names[i] for i in pulse_sequence])
#     return pulse_list
#
#
# def get_pulse_sequence(string):
#     """Returns a list of integers for the pulse sequence
#     """
#     chars = string.split(',')
#     pulse_sequence = [pulse_names.index(c) for c in chars]
#     return pulse_sequence
#
#
# def get_propagator(pulse_sequence, pulses):
#     propagator = qt.identity(pulses[0].dims[0])
#     for p in pulse_sequence:
#         propagator = pulses[p] * propagator
#     return propagator


def get_rotations(pulse_list):
    # General rotation matrices R_i(theta); i in {x, y, z}; 3x3 matrix
    Delay = np.eye(3)
    X = np.round(Rotation.from_euler('x', 90, degrees=True).as_matrix())
    Xbar = np.round(Rotation.from_euler('x', -90, degrees=True).as_matrix())
    Y = np.round(Rotation.from_euler('y', 90, degrees=True).as_matrix())
    Ybar = np.round(Rotation.from_euler('y', -90, degrees=True).as_matrix())

    rotations = []      # Each rotation corresponds to a pulse in the action space

    for pulse_name in pulse_list:
        if len(pulse_name) == 0:
            return IndexError
        rot = Delay
        for subpulse in pulse_name:
            if subpulse == 'D':
                rot = Delay @ rot
            if subpulse == 'X':
                rot = X @ rot
            if subpulse == '-X':
                rot = Xbar @ rot
            if subpulse == 'Y':
                rot = Y @ rot
            if subpulse == '-Y':
                rot = Ybar @ rot
        rotations.append(rot)
    return rotations


# # The following functions are never used, but very similar ones are used in alpha_zero.py's make_sequence()
# # TODO: Maybe I can use these to help generalize alpha_zero.py()
# def get_rotation(pulse_sequence):
#     # This is the non-recursive version of get_frame()
#     # Check to make sure this is the case
#     frame = np.eye(3)
#     for p in pulse_sequence:
#         frame = rotations[p] @ frame
#     return frame
#
#
# def is_cyclic(pulse_sequence):
#     frame = get_rotation(pulse_sequence)
#     return (frame == np.eye(3)).all()
#
#
# def count_axes(pulse_sequence):
#     """Count time spent on (x, y, z, -x, -y, -z) axes
#     """
#     axes_counts = [0] * 6
#     frame = np.eye(3)
#     for p in pulse_sequence:
#         frame = rotations[p] @ frame
#         axis = np.where(frame[-1, :])[0][0]
#         is_negative = np.sum(frame[-1, :]) < 0
#         axes_counts[axis + 3 * is_negative] += 1
#     return axes_counts
#
#
# def is_valid_dd(subsequence, sequence_length):
#     """Checks if the pulse subsequence allows for dynamical decoupling of
#         dipolar interactions (i.e. equal time spent on each axis)
#     """
#     axes_counts = count_axes(subsequence)
#     (x, y, z) = [axes_counts[i] + axes_counts[i + 3] for i in range(3)]
#     # time on each axis isn't more than is allowed for dd
#     return (np.array([x, y, z]) <= sequence_length / 3).all()
#
#
# def is_valid_time_suspension(subsequence, sequence_length):
#     """Checks if the pulse subsequence allows for dynamical decoupling of
#         all interactions (i.e. equal time spent on each ± axis)
#     """
#     axes_counts = count_axes(subsequence)
#     # time on each axis isn't more than is allowed for dd
#     return (np.array(axes_counts) <= sequence_length / 6).all()
#
#
# def get_valid_time_suspension_pulses(subsequence,
#                                      num_pulses,
#                                      sequence_length):
#     valid_pulses = []
#     for p in range(num_pulses):
#         if is_valid_time_suspension(subsequence + [p], sequence_length):
#             valid_pulses.append(p)
#     return valid_pulses
#
#
# # fidelity calculations
#
# def get_fidelity(pulse_sequence, Utarget, pulses):
#     Uexp = qt.identity(Utarget.dims[0])
#     for p in pulse_sequence:
#         Uexp = pulses[p] * Uexp
#     return qt.metrics.average_gate_fidelity(Uexp, Utarget)
#
#
# def get_mean_fidelity(pulse_sequence, Utarget, pulses_ensemble):
#     fidelity = 0
#     for pulses in pulses_ensemble:
#         fidelity += get_fidelity(pulse_sequence, Utarget, pulses)
#     return fidelity / len(pulses_ensemble)


# New pulses
# Try out a new class
class PulseInfo(object):
    def __init__(self, name, length, pulse, ensemble_number=None, ensemble_size=None):
        self.name = name
        self.length = length
        self.pulse = pulse
        self.ensemble_number = ensemble_number
        self.ensemble_size = ensemble_size

    def get_name(self):
        return self.name

    def get_length(self):
        return self.length

    def get_pulse(self):
        return self.pulse

    def get_ensemble(self):
        return self.ensemble_number

    def update_ensemble_number(self, new_number):
        self.ensemble_number = new_number

    def update_ensemble_size(self, new_size):
        self.ensemble_size = new_size

    def __str__(self):
        return str(self.ensemble_number) + '/' + str(self.ensemble_size) + ' ' + str(self.name) + ' pulse (' + \
               str(self.length) + ' tau).'


class PulseSequenceConfig(object):
    
    def __init__(self,
                 Utarget,
                 N=3,
                 ensemble_size=3,
                 max_sequence_length=48,
                 dipolar_strength=1e2,
                 pulse_width=1e-5,
                 delay=1e-4,
                 rot_error=1e-2,
                 phase_transient_error=1e-2,
                 offset_error=1e0,
                 sequence_length=None,
                 pulse_list=None,
                 rotations=None,
                 Hsys_ensemble=None,
                 pulses_ensemble=None,
                 sequence=None,
                 rng=None,
                 save_name=None,
                 ):
        """Create a new pulse sequence config object. Basically a collection
        of everything on the physics side of things that is relevant for
        pulse sequences.
        
        Args:
            Utarget (Qobj data): the target propagator
            N (int): number of spins in the system.
            ensemble_size (int): The number of systems of N spin 1/2s to work with
            max_sequence_length (int): length of the pulse sequence you want to generate
            dipolar_strength (float): Standard deviation of dipolar coupling strengths.
            pulse_width (float): amount of time each pulse is applied for (in seconds)
            delay (float): amount of time in between each pulse (in seconds)
            rot_error (float): Standard deviation of rotation error to randomly
                sample from.
            phase_transient_error (float): Normalized magnitude of phase transient for pulses
            (1 is a full pi/2 pulse). Default is 0 (no error).
            offset_error (float): TODO
            sequence_length (int): Length of the sequence. This depends on the length of each pulse in the sequence.
            pulse_list (list of lists): A list of lists of pulses which make up the action space. Defaults to the most
                basic action space used originally in Will's thesis
            rotations (list): A list of the rotations corresponding to each pulse in the action space. Defaults to none
            Hsys_ensemble: The Hamiltonian of the ensemble. Defaults to None.
            pulses_ensemble (list of lists of PulseInfo objects): Each element is a list of PulseInfo objects describing
                the pulse in the action space for the specific group in the ensemble. There are ensemble number of
                elements in pulses_ensemble.
            sequence: A list of the pulses applied to get to the current state. If no pulses have been applied,
                assumed value of None. Defaults to None
            rng: the random number generator used. Defaults to None. If None, rng=np.random.default_rng()
            save_name (str): Filename to save the ensemble parameters (chemical shift,
                offset, and dipolar matrices). Defaults to None.
        """
        self.N = N
        self.ensemble_size = ensemble_size
        self.max_sequence_length = max_sequence_length
        self.Utarget = Utarget
        self.dipolar_strength = dipolar_strength
        self.pulse_width = pulse_width
        self.delay = delay
        self.rot_error = rot_error
        self.phase_transient_error = phase_transient_error
        self.offset_error = offset_error
        self.sequence_length = 0 if sequence_length is None else sequence_length
        self.pulse_list = [['D'], ['X'], ['-X'], ['Y'], ['-Y']] if pulse_list is None else pulse_list
        self.save_name = save_name
        # create a unique rng for multiprocessing purposes
        self.rng = rng if rng is not None else np.random.default_rng()
        if Hsys_ensemble is None:
            self.Hsys_ensemble = []
            if save_name is not None:
                chemical_shifts = []
                offsets = []
                dipolar_matrices = []

            # Ensemble size usually between 10-50
            for _ in range(ensemble_size):
                o = 0
                if offset_error > 0:
                    # Samples from a Gaussian distribution with standard deviation = scale = offset_error (default is 1)
                    # So o is a random sample from a Gaussian distribution of standard deviation 1
                    o = self.rng.normal(scale=offset_error)
                if save_name is not None:
                    H, (cs, offset, dip) = get_Hsys(
                        N=N,
                        dipolar_strength=dipolar_strength,
                        offset=o,
                        rng=self.rng, return_all=True)
                    chemical_shifts.append(cs)
                    offsets.append(offset)
                    dipolar_matrices.append(dip)
                else:
                    H = get_Hsys(N=N,
                                 dipolar_strength=dipolar_strength,
                                 offset=o,
                                 rng=self.rng)
                self.Hsys_ensemble.append(H)        # Append the Hamiltonian for each system in the ensemble to a list
        else:
            self.Hsys_ensemble = Hsys_ensemble
        if pulses_ensemble is None:
            if save_name is not None:
                rots = []
            X, Y, Z = get_collective_spin(N)
            self.pulses_ensemble = []
            for idx, H in enumerate(self.Hsys_ensemble):        # Finds the pulses for each ensemble
                rot = self.rng.normal(scale=rot_error)
                pt = np.abs(self.rng.normal(scale=phase_transient_error))
                if save_name is not None:
                    rots.append(rot)
                pulse_ensemble = get_pulses(H, X, Y, Z, pulse_list=pulse_list, pulse_width=pulse_width, delay=delay,
                                            rot_error=rot, phase_transient=pt)
                for pulse in pulse_ensemble:
                    pulse.update_ensemble_number(idx+1)
                    pulse.update_ensemble_size(len(self.Hsys_ensemble))

                self.pulses_ensemble.append(pulse_ensemble)
        else:
            self.pulses_ensemble = pulses_ensemble
        if save_name is not None:
            chemical_shifts = np.stack(chemical_shifts)
            offsets = np.stack(offsets)
            dipolar_matrices = np.stack(dipolar_matrices)
            rots = np.stack(rots)
            np.savez_compressed(save_name, chemical_shifts=chemical_shifts,
                                offsets=offsets, dipolar_matrices=dipolar_matrices,
                                rots=rots)
        self.num_pulses = len(self.pulses_ensemble[0])      # Gives size of action space
        self.sequence = [] if sequence is None else sequence
        self.rotations = get_rotations(pulse_list=pulse_list) if rotations is None else rotations

    def reset(self):
        """Reset the pulse sequence config to an empty pulse sequence
        """
        self.sequence = []
        self.sequence_length = 0
    
    def is_done(self):
        """Return whether the pulse sequence is at or beyond its
        maximum sequence length.
        """
        return self.sequence_length >= self.max_sequence_length
    
    def apply(self, pulse):
        """Apply a pulse to the current pulse sequence.
        """
        self.sequence.append(pulse)
        self.sequence_length += self.pulses_ensemble[0][pulse].get_length()

    def clone(self):
        """Clone the pulse sequence config. Objects
        that aren't modified are simply returned as-is.
        """
        return PulseSequenceConfig(
            N=self.N,
            ensemble_size=self.ensemble_size,
            max_sequence_length=self.max_sequence_length,
            Utarget=self.Utarget,
            dipolar_strength=self.dipolar_strength,
            pulse_width=self.pulse_width,
            delay=self.delay,
            rot_error=self.rot_error,
            phase_transient_error=self.phase_transient_error,
            offset_error=self.offset_error,
            pulse_list=self.pulse_list,
            sequence_length=self.sequence_length,
            Hsys_ensemble=self.Hsys_ensemble,
            pulses_ensemble=self.pulses_ensemble,
            sequence=self.sequence.copy(),
            rng=self.rng,
            save_name=self.save_name,
            rotations=self.rotations
        )
