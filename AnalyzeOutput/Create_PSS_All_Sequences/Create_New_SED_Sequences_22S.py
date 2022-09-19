# Created 2/17/22
# Updated 4/29/22 to get the names of all of the pulse sequences found here
# Owen Eskandari

# This is to take a number of SE (solid echo) sequences that perform well, look at their representation as SED
# (solid echo plus delay) sequences, and pair together sequences that zero out time spent on each axis, and then plot
# their error against standard sequences to try to find pulse sequences that outperform CORY48

# Additional goal: determine patterns that begin to emerge and see what pulse sequences are high fidelity and robust

# Imports
from Matrix_Representation_Reference import *
from Error_Plots_Reference import *
import string       # For making a list of names
import pprint


def create_New_SED_Sequences(ps_list):
    '''
    :param ps_list: Make sure this list has tuples ('name', pulse_sequence). Make sure (for now) that the action
        space is SE (solid echo)
    :return:
    '''

    # Create the dictionary with the keys being the axis counts and the pulse sequences associated with each axis count
    single_SED_dictionary = {}
    for ps in ps_list:
        # x, y, z = 0, 0, -2
        x, y, z, refocus_count = visual_representation(ps[1], 'SED', plot=False)

        try:
            single_SED_dictionary[(x, y, z)]
        except KeyError:
            single_SED_dictionary.update({(x, y, z): []})
        single_SED_dictionary[(x, y, z)].append(ps)

    pprint.pprint(single_SED_dictionary)

    # Create list of combined pulse sequences (for now, only a + b (2 pulse sequences))
    refocused_list = []
    for key in single_SED_dictionary:
        # TODO: for now, only consider adding pulses where only one axis is nonzero
        nonzero_count = 0
        for idx, axis in enumerate(key):
            if axis != 0:
                nonzero_count += 1
                nonzero_axis = idx
        if nonzero_count == 1:
            key_a = key
            key_b = list(key)
            key_b[nonzero_axis] = -2*key[nonzero_axis]      # TODO Twice now
            key_b = tuple(key_b)
            print(key_a, key_b)     # TODO: why is this not working??
            for ps_a in single_SED_dictionary[key_a]:
                try:
                    for ps_b in single_SED_dictionary[key_b]:
                        x, y, z, refocus_count = visual_representation(2*ps_a[1]+ps_b[1], 'SED', plot=False)    # Todo
                        if x == y == z == 0:
                            refocused_list.append(('2*' + ps_a[0]+'+'+ps_b[0], 2*ps_a[1]+ps_b[1], refocus_count))   # Todo
                        else:
                            print("What happened?")
                except KeyError:
                    pass

    return refocused_list


# Above reward of 5 from SE All Error run
items1 = [[6, 1, 1, 3, 8, 7, 8, 7, 3, 4, 4, 5], [5, 5, 7, 3, 4, 4, 3, 7, 8, 8, 1, 1], [6, 6, 7, 8, 8, 5, 2, 3, 3, 1, 4, 4], [6, 5, 6, 7, 5, 7, 2, 4, 3, 4, 3, 1], [2, 2, 6, 6, 5, 4, 3, 3, 7, 7, 8, 1], [1, 5, 5, 7, 3, 4, 4, 3, 7, 8, 8, 1], [7, 5, 5, 7, 3, 4, 4, 8, 8, 1, 1, 3], [5, 5, 7, 3, 4, 4, 8, 8, 1, 1, 3, 7], [7, 5, 8, 8, 6, 3, 4, 3, 1, 2, 1, 7], [3, 7, 5, 6, 2, 4, 8, 7, 3, 1, 2, 6], [5, 5, 7, 3, 4, 4, 8, 8, 6, 2, 1, 1], [5, 5, 2, 4, 2, 8, 6, 6, 3, 1, 3, 7], [5, 5, 2, 2, 4, 7, 8, 8, 3, 3, 1, 6], [5, 6, 2, 4, 3, 7, 8, 7, 3, 1, 2, 6], [5, 5, 7, 3, 1, 1, 3, 7, 8, 8, 4, 4], [3, 7, 5, 5, 4, 4, 2, 6, 8, 8, 1, 1], [2, 6, 5, 5, 4, 4, 3, 7, 8, 8, 1, 1], [5, 7, 3, 4, 4, 3, 7, 8, 8, 1, 1, 5], [5, 5, 3, 3, 1, 6, 7, 7, 4, 4, 2, 8], [2, 6, 5, 6, 2, 4, 3, 7, 8, 7, 3, 1], [5, 5, 7, 2, 4, 4, 8, 8, 6, 3, 1, 1], [2, 6, 5, 5, 4, 4, 8, 8, 1, 1, 3, 7], [5, 5, 4, 4, 3, 7, 8, 8, 1, 1, 2, 6], [2, 6, 5, 5, 4, 4, 8, 8, 6, 2, 1, 1], [5, 5, 7, 3, 1, 1, 8, 8, 6, 2, 4, 4], [1, 5, 5, 7, 3, 4, 4, 8, 8, 6, 2, 1]]

# ps_list = [('seae24'+names[i], item) for i, item in enumerate(items)]

items2 = [[5, 2, 1, 2, 8, 7, 8, 7, 4, 3, 4, 6], [8, 2, 4, 7, 8, 2, 4, 3, 5, 3, 5, 7],[2, 2, 1, 8, 7, 8, 4, 4, 3, 6, 5, 6],[1, 3, 8, 7, 8, 2, 4, 2, 1, 5, 6, 5],[5, 2, 1, 2, 8, 7, 8, 7, 4, 3, 4, 6],[5, 7, 1, 2, 1, 7, 8, 6, 8, 3, 4, 3],[5, 2, 1, 2, 8, 7, 8, 7, 4, 3, 4, 6],[1, 3, 1, 4, 2, 7, 5, 7, 6, 8, 6, 4],[3, 4, 3, 6, 6, 8, 2, 1, 2, 7, 7, 5],[1, 1, 3, 5, 6, 8, 6, 5, 7, 2, 2, 4],[3, 3, 6, 5, 7, 8, 7, 5, 2, 1, 1, 4],[4, 3, 3, 8, 6, 8, 1, 2, 2, 5, 7, 5],[3, 1, 2, 4, 6, 6, 8, 5, 5, 7, 2, 1],[3, 3, 6, 6, 8, 1, 2, 2, 7, 7, 5, 4]]


items3 = [[4, 8, 7, 8, 3, 8, 7, 5, 1, 3, 1, 3, 1, 2, 5, 6, 7, 2, 6, 5, 6, 4, 4, 2], [8, 5, 8, 1, 3, 8, 4, 2, 5, 7, 6, 3, 2, 2, 3, 6, 7, 1, 6, 1, 4, 4, 7, 5], [6, 7, 8, 8, 5, 6, 3, 8, 1, 6, 7, 7, 1, 1, 5, 3, 4, 3, 5, 2, 4, 2, 4, 2], [6, 7, 3, 8, 8, 2, 4, 1, 5, 5, 6, 5, 6, 7, 1, 2, 4, 2, 1, 3, 7, 3, 4, 8], [4, 4, 7, 1, 2, 1, 3, 4, 8, 5, 6, 3, 8, 8, 6, 5, 6, 2, 2, 3, 5, 7, 7, 1], [4, 2, 4, 7, 5, 6, 2, 4, 2, 6, 6, 3, 1, 3, 1, 3, 1, 5, 8, 7, 8, 5, 8, 7], [2, 2, 4, 5, 7, 5, 7, 5, 2, 4, 1, 1, 3, 3, 1, 8, 6, 8, 6, 6, 8, 4, 3, 7], [4, 2, 4, 7, 5, 7, 7, 5, 1, 2, 3, 3, 1, 1, 3, 8, 8, 6, 6, 6, 8, 4, 2, 5], [4, 2, 7, 7, 5, 8, 4, 2, 2, 1, 1, 1, 5, 7, 5, 8, 6, 3, 4, 3, 3, 6, 6, 8], [8, 8, 4, 2, 3, 3, 1, 4, 8, 5, 5, 5, 6, 6, 6, 2, 2, 4, 1, 1, 3, 7, 7, 7], [7, 5, 3, 4, 3, 3, 2, 7, 4, 2, 6, 8, 5, 4, 6, 8, 6, 1, 2, 1, 7, 8, 1, 5], [7, 8, 2, 4, 2, 5, 7, 5, 1, 5, 7, 6, 4, 3, 3, 1, 3, 6, 6, 8, 4, 1, 2, 8], [8, 2, 2, 2, 5, 5, 5, 1, 1, 1, 6, 6, 6, 7, 7, 7, 3, 3, 3, 4, 4, 4, 8, 8], [7, 8, 2, 6, 5, 5, 7, 5, 1, 1, 3, 3, 1, 2, 4, 4, 3, 7, 6, 8, 6, 4, 2, 8], [8, 8, 2, 1, 4, 8, 5, 6, 5, 1, 2, 1, 6, 7, 7, 5, 7, 4, 3, 3, 3, 4, 2, 6], [7, 8, 2, 6, 8, 5, 5, 5, 7, 1, 2, 1, 4, 2, 3, 3, 3, 4, 7, 6, 6, 4, 1, 8], [7, 8, 2, 8, 7, 1, 2, 2, 5, 8, 6, 7, 3, 3, 6, 1, 3, 6, 4, 4, 4, 5, 5, 1], [7, 8, 2, 5, 6, 6, 5, 5, 1, 2, 3, 4, 4, 4, 7, 1, 3, 3, 6, 7, 8, 1, 2, 8], [3, 4, 3, 4, 3, 3, 5, 8, 8, 8, 6, 6, 6, 5, 5, 1, 2, 1, 1, 2, 2, 7, 7, 7], [3, 4, 4, 3, 3, 5, 7, 5, 7, 5, 7, 6, 8, 6, 1, 2, 2, 1, 2, 1, 8, 6, 8, 4], [3, 4, 4, 3, 3, 5, 5, 8, 1, 2, 1, 6, 7, 8, 1, 2, 2, 6, 6, 7, 4, 7, 5, 8], [3, 4, 5, 3, 4, 3, 6, 8, 6, 7, 1, 2, 1, 2, 1, 2, 5, 7, 5, 7, 8, 6, 8, 4], [3, 4, 5, 3, 4, 3, 6, 6, 8, 5, 7, 5, 7, 7, 2, 1, 2, 1, 2, 1, 8, 6, 8, 4], [3, 4, 4, 3, 3, 5, 6, 5, 6, 8, 8, 6, 8, 4, 7, 7, 5, 2, 1, 7, 2, 1, 1, 2], [3, 4, 4, 3, 3, 5, 6, 5, 6, 7, 8, 8, 6, 8, 4, 1, 1, 2, 1, 2, 7, 7, 5, 2], [3, 4, 4, 3, 3, 5, 6, 5, 6, 6, 8, 8, 1, 2, 8, 7, 7, 1, 2, 1, 2, 5, 7, 4]]

# From most recent se24 run (after 2/23)
items4 = [[6, 4, 1, 1, 3, 2, 2, 5, 6, 7, 8, 7], [1, 1, 3, 2, 2, 5, 6, 8, 7, 8, 6, 4], [2, 4, 2, 1, 3, 8, 8, 5, 6, 6, 7, 1], [3, 4, 2, 4, 3, 1, 5, 6, 7, 8, 7, 6], [3, 4, 4, 1, 2, 2, 5, 6, 8, 7, 8, 6], [2, 6, 8, 8, 6, 2, 1, 1, 5, 5, 4, 4], [7, 5, 7, 8, 6, 8, 3, 2, 2, 4, 1, 1], [6, 3, 4, 2, 1, 2, 4, 7, 8, 8, 5, 6], [5, 1, 3, 1, 5, 6, 8, 2, 4, 2, 8, 7], [6, 8, 7, 5, 1, 1, 3, 2, 2, 4, 7, 8], [5, 8, 8, 6, 4, 3, 4, 2, 1, 2, 7, 7], [5, 6, 8, 4, 2, 4, 8, 6, 5, 1, 2, 1], [6, 8, 3, 4, 2, 5, 7, 5, 2, 1, 3, 8], [5, 2, 1, 3, 1, 2, 4, 8, 8, 6, 7, 7], [6, 6, 8, 3, 4, 2, 4, 3, 1, 5, 5, 7], [5, 6, 3, 4, 2, 1, 2, 4, 8, 7, 8, 6], [5, 6, 5, 3, 4, 2, 1, 2, 4, 8, 7, 8], [5, 6, 3, 4, 1, 5, 8, 7, 2, 1, 4, 8], [5, 6, 5, 3, 4, 2, 1, 2, 4, 7, 8, 7], [5, 6, 3, 4, 3, 7, 7, 8, 2, 1, 2, 5], [5, 6, 8, 4, 2, 4, 8, 7, 5, 1, 3, 1], [5, 8, 8, 6, 7, 7, 2, 2, 4, 1, 1, 3], [5, 6, 5, 3, 4, 2, 8, 7, 8, 2, 1, 3], [5, 6, 1, 3, 1, 6, 8, 7, 4, 2, 4, 7], [5, 6, 8, 4, 2, 4, 8, 7, 5, 3, 1, 3], [5, 8, 8, 6, 7, 7, 4, 2, 4, 1, 3, 1], [5, 6, 5, 8, 7, 8, 2, 4, 2, 1, 3, 1], [6, 3, 4, 2, 1, 2, 4, 8, 7, 8, 6, 5], [5, 8, 8, 6, 7, 7, 2, 2, 3, 4, 4, 1]]


alphabet = list(string.ascii_lowercase)
alphabet += ['a' + i for i in list(string.ascii_lowercase)]
alphabet += ['b' + i for i in list(string.ascii_lowercase)]
names1 = ['seae24' + name for name in alphabet[:len(items1)]]
names11 = ['2x seae24' + name for name in alphabet[:len(items1)]]
names2 = [name + '24' for name in alphabet[:len(items2)]]
names21 = ['2x ' + name + '24' for name in alphabet[:len(items2)]]
names3 = [name + '48' for name in alphabet[:len(items3)]]
names4 = [name + '24new' for name in alphabet[:len(items4)]]
names41 = ['2x' + name + '24new' for name in alphabet[:len(items4)]]
names = names1 + names11 + names2 + names21 + names3 + names4 + names41
items = items1 + [2*item for item in items1] + items2 + [2*item for item in items2] + items3 + items4 + [2*item for item in items4]
print(len(items), len(names))

ps_list = [(names[i], item) for i, item in enumerate(items)]
print(ps_list)


l = create_New_SED_Sequences(ps_list=ps_list)

print(l)
