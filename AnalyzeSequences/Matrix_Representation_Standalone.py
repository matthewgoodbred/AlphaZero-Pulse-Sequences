# 2/4/22
# Owen Eskandari
# This file is to work through implementing the visual interface described in detail in Choi et al.
# https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.031002

# Two main parts to this:
# 1) Making sure that the PS correctly correspond to the right representations (the physics)
# 2) Visualization given physics

# TODO: Note that this follows the format of Choi et al. and shows the matrix representation BEFORE the pulse is applied

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import string
from Convert import convert


def visual_representation(pulse_sequence, action_space, dpi=500, name='Matrix_representation',
                          plot=False, folder=None):
    '''

    :param pulse_sequence: List. List of pulses (numbers)
    :param action_space: Str. O, SE, SED: A string for the action space representation of the pulse sequence.
        O = original action space [D, X, -X, Y, -Y}; SE = solid echo; SED = solid echo plus delay
    :param dpi: Int. Resolution of the output image. Default is 500
    :param name: Str. Name of output file, which will be a png.
    :return: saves a png of the table
    '''

    # Helper functions

    # From pulse_sequences_general.py
    def get_rotations(pulse_list):
        # General rotation matrices R_i(theta); i in {x, y, z}; 3x3 matrix
        Delay = np.eye(3)
        X = np.round(Rotation.from_euler('x', 90, degrees=True).as_matrix())
        Xbar = np.round(Rotation.from_euler('x', -90, degrees=True).as_matrix())
        Y = np.round(Rotation.from_euler('y', 90, degrees=True).as_matrix())
        Ybar = np.round(Rotation.from_euler('y', -90, degrees=True).as_matrix())

        rotations = []  # Each rotation corresponds to a pulse in the action space

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

    # Only need rotations for the basic pulse as pulse sequences have already been converted
    # It is needed in this order as we want to see what axis the system is on after every Pi/2 pulse (not composite)
    rotations = get_rotations(pulse_list=[['D'], ['X'], ['-X'], ['Y'], ['-Y']])  # This way I only have to call it once

    # From alpha_zero_OE.py
    def get_frame(sequence):
        if len(sequence) == 0:
            return np.eye(3)
        else:
            # Rotation is from the tested pulse_index in get_valid_pulses() (and then perform matrix multiplication)
            return rotations[sequence[-1]] @ get_frame(sequence[:-1])

    # Would need to account for pulse length weights under same conditions as get_valid_pulses()
    def get_matrix(sequence):
        if len(sequence) == 0:
            return []  # Return blank list
        else:
            axes = get_matrix(sequence[:-1]).copy()  # This is where caching comes in handy
            # 3x3 matrix with 0, ±1 as entries; all rows/columns linearly independent and span
            frame = get_frame(sequence)
            # Determine which axis the pulse sequence is on (±x, ±y, ±z) and increment by 1
            axis = np.where(frame[-1, :])[0][0]  # The axis is the index of the nonzero element in the third row
            is_negative = np.sum(frame[-1, :]) < 0  # Determines whether the nonzero element (axis) is +1 or -1
            # is_negative acts as piecewise function: False = 0, True = 1
            axes.append(2 * axis + 1 + 1 * is_negative)  # Add axis to axes list (key below)
            # 1: X; 2: -X; 3: Y; 4: -Y; 5: Z; 6: -Z
            return axes

    pulse_sequence = convert(pulse_sequence, action_space, 'O')

    axes = get_matrix(pulse_sequence)        # List of what axis the system ended on after each pulse

    pulses = ['Pulses']

    for pulse in pulse_sequence:
        if pulse == 0:
            pulses.append('D')
        elif pulse == 1:
            pulses.append('X')
        elif pulse == 2:
            pulses.append(r'$\mathregular{\overline{X}}$')
        elif pulse == 3:
            pulses.append('Y')
        elif pulse == 4:
            pulses.append(r'$\mathregular{\overline{Y}}$')

    pulses.append('End')

    data, green, red = [pulses], [], []     # Data is array for data, red and green highlight the correct cells (-1, +1)
    green.append((3, 1))

    T_J_x, T_J_y, T_J_z = 0, 0, 1
    T_W_x, T_W_y, T_W_z = 0, 0, 1
    refocus_count = 0
    T_J_light, T_J_dark, T_W_light, T_W_dark = [], [], [], []    # Symmetrization of interactions; decoupling disorder
    light_J, light_W = True, True        # Boolean to determine cycles for T_J and T_W
    T_J_light.append((4, 1))
    T_W_light.append((5, 1))
    yellow = []      # To see if cycles completed

    # Add data row by row
    # Add data column by column
    row1 = ['$\mathregular{F_x}$', 0]
    row2 = ['$\mathregular{F_y}$', 0]
    row3 = ['$\mathregular{F_z}$', 1]
    row4 = ['$\mathregular{T_J}$', '']
    row5 = ['$\mathregular{T_W}$', '']

    end_J, end_W = False, False        # To mark at the end if a cycle has been completed

    for j, axis in enumerate(axes):
        row4.append('')
        row5.append('')
        if axis == 1:
            row1.append(1)
            T_J_x += 1
            T_W_x += 1
            green.append((1, j + 2))
        elif axis == 2:
            row1.append(-1)
            T_J_x += 1
            T_W_x -= 1
            red.append((1, j + 2))
        else:
            row1.append(0)

        if axis == 3:
            row2.append(1)
            T_J_y += 1
            T_W_y += 1
            green.append((2, j + 2))
        elif axis == 4:
            row2.append(-1)
            T_J_y += 1
            T_W_y -= 1
            red.append((2, j + 2))
        else:
            row2.append(0)

        if axis == 5:
            row3.append(1)
            T_J_z += 1
            T_W_z += 1
            green.append((3, j + 2))
        elif axis == 6:
            row3.append(-1)
            T_J_z += 1
            T_W_z -= 1
            red.append((3, j + 2))
        else:
            row3.append(0)

        if j != len(axes) - 1:
            if not light_J:
                T_J_dark.append((4, j + 2))
            else:
                T_J_light.append((4, j + 2))
            if not light_W:
                T_W_dark.append((5, j + 2))
            else:
                T_W_light.append((5, j + 2))

            if T_J_x == T_J_y == T_J_z:
                light_J = not light_J
                if j == len(axes) - 2:
                    end_J = True
            if T_W_x == 0 and T_W_y == 0 and T_W_z == 0:
                refocus_count += 1
                light_W = not light_W
                if j == len(axes) - 2:
                    end_W = True
        else:
            if end_J:
                yellow.append((4, j + 2))
            if end_W:
                yellow.append((5, j + 2))

    if not plot:        # Don't make the plots/save them, and just show what the sum is on each axis
        return T_W_x, T_W_y, T_W_z-1, refocus_count

    data = [pulses, row1, row2, row3, row4, row5]  # Add row to data

    # Create the table
    the_table = plt.table(cellText=data,
                          loc='center',
                          cellLoc='center',
                          colLoc='right',
                          )

    for cell in green:  # Color the appropriate cells green
        the_table[cell[0], cell[1]].set_facecolor('yellowgreen')
    for cell in red:    # Color the appropriate cells red
        the_table[cell[0], cell[1]].set_facecolor('tab:red')
    for cell in T_W_light:    # Color the appropriate cycles
        the_table[cell[0], cell[1]].set_facecolor('lightgray')
    for cell in T_J_light:    # Color the appropriate cycles
        the_table[cell[0], cell[1]].set_facecolor('violet')
    for cell in T_W_dark:    # Color the appropriate cycles
        the_table[cell[0], cell[1]].set_facecolor('darkgray')
    for cell in T_J_dark:    # Color the appropriate cycles
        the_table[cell[0], cell[1]].set_facecolor('darkviolet')
    for cell in yellow:
        the_table[cell[0], cell[1]].set_facecolor('yellow')

    the_table.scale(len(pulse_sequence)*0.1, 2)     # Scale the table accordingly

    # Remove plot-like parts of the figure
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)
    if folder is None:
        plt.savefig('Matrix_Representation/' + name + '.png', bbox_inches='tight', dpi=dpi)  # Save a high resolution figure
    else:
        plt.savefig(folder + name + '.png', bbox_inches='tight', dpi=dpi)  # Save a high resolution figure
