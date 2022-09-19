'''
4/25/22
Owen Eskandari

This file converts a given pulse sequence into a specified action space, if possible

Manually convert for now
'''


def convert(ps, orig, new):
    '''
    This file converts a given pulse sequence into a specified action space, if possible
    :param ps: List. List of pulses, numbered according to orig action space
    :param orig: Str. Original action space ('O', 'SE', 'SED', 'SEDD', 'B')
    :param new: Str. New action space ('O', 'SE', 'SED', 'SEDD', 'B')
    :return: List. List of pulses, numbered according to new action space. Return of 0 means input was incompatible
    with orig action space, 1 means input was incompatible with output action space
    '''

    # TODO: For now, until I can fix the conversion back
    if new != 'O':
        return 'Come back later'

    # Convert from original action space to the original action space
    if orig == 'O':
        output = []
        for i in ps:
            if i == 0:
                output.append(0)
            elif i == 1:
                output.append(1)
            elif i == 2:
                output.append(2)
            elif i == 3:
                output.append(3)
            elif i == 4:
                output.append(4)
            else:
                return 0

    elif orig == 'SE':
        output = []
        for i in ps:
            if i == 0:
                output.append(0)
            elif i == 1:
                output.append(1)
                output.append(3)
            elif i == 2:
                output.append(1)
                output.append(4)
            elif i == 3:
                output.append(2)
                output.append(3)
            elif i == 4:
                output.append(2)
                output.append(4)
            elif i == 5:
                output.append(3)
                output.append(1)
            elif i == 6:
                output.append(3)
                output.append(2)
            elif i == 7:
                output.append(4)
                output.append(1)
            elif i == 8:
                output.append(4)
                output.append(2)
            else:
                return 0

    elif orig == 'SEDD':     # Note: DELAY in SEDD action space
        output = []
        for i in ps:
            if i == 0:
                output.append(0)
            elif i == 1:
                output.append(1)
                output.append(3)
                output.append(0)
            elif i == 2:
                output.append(1)
                output.append(4)
                output.append(0)
            elif i == 3:
                output.append(2)
                output.append(3)
                output.append(0)
            elif i == 4:
                output.append(2)
                output.append(4)
                output.append(0)
            elif i == 5:
                output.append(3)
                output.append(1)
                output.append(0)
            elif i == 6:
                output.append(3)
                output.append(2)
                output.append(0)
            elif i == 7:
                output.append(4)
                output.append(1)
                output.append(0)
            elif i == 8:
                output.append(4)
                output.append(2)
                output.append(0)
            else:
                return 0

    elif orig == 'SED':     # Note: NO DELAY in SED action space
        output = []
        for i in ps:
            if i == 0:
                output.append(1)
                output.append(3)
                output.append(0)
            elif i == 1:
                output.append(1)
                output.append(4)
                output.append(0)
            elif i == 2:
                output.append(2)
                output.append(3)
                output.append(0)
            elif i == 3:
                output.append(2)
                output.append(4)
                output.append(0)
            elif i == 4:
                output.append(3)
                output.append(1)
                output.append(0)
            elif i == 5:
                output.append(3)
                output.append(2)
                output.append(0)
            elif i == 6:
                output.append(4)
                output.append(1)
                output.append(0)
            elif i == 7:
                output.append(4)
                output.append(2)
                output.append(0)
            else:
                return 0

    elif orig == 'B':
        output = []
        for i in ps:
            if i == 0:
                output.append(0)
            elif i == 1:
                output.append(1)
                output.append(3)
            elif i == 2:
                output.append(1)
                output.append(4)
            elif i == 3:
                output.append(2)
                output.append(3)
            elif i == 4:
                output.append(2)
                output.append(4)
            elif i == 5:
                output.append(3)
                output.append(1)
            elif i == 6:
                output.append(3)
                output.append(2)
            elif i == 7:
                output.append(4)
                output.append(1)
            elif i == 8:
                output.append(4)
                output.append(2)
            elif i == 9:
                output.append(1)
                output.append(3)
                output.append(0)
            elif i == 10:
                output.append(1)
                output.append(4)
                output.append(0)
            elif i == 11:
                output.append(2)
                output.append(3)
                output.append(0)
            elif i == 12:
                output.append(2)
                output.append(4)
                output.append(0)
            elif i == 13:
                output.append(3)
                output.append(1)
                output.append(0)
            elif i == 14:
                output.append(3)
                output.append(2)
                output.append(0)
            elif i == 15:
                output.append(4)
                output.append(1)
                output.append(0)
            elif i == 16:
                output.append(4)
                output.append(2)
                output.append(0)
            else:
                return 0
    else:
        return "Not a valid original action space"

    # Now given the output in the original action space, convert to desired action space
    if new == 'O':
        final = output
        return final
    elif new == 'SE':
        final = []
        for idx, i in enumerate(output):
            try:
                if i == 0:
                    final.append(0)
                elif i == 1 and output[idx+1] == 3:
                    final.append(1)
                elif i == 1 and output[idx+1] == 4:
                    final.append(2)
                elif i == 2 and output[idx+1] == 3:
                    final.append(3)
                elif i == 2 and output[idx+1] == 4:
                    final.append(4)
                elif i == 3 and output[idx+1] == 1:
                    final.append(5)
                elif i == 3 and output[idx+1] == 2:
                    final.append(6)
                elif i == 4 and output[idx+1] == 1:
                    final.append(7)
                elif i == 4 and output[idx+1] == 2:
                    final.append(8)
                else:
                    return 1
            except IndexError:
                pass

    elif new == 'SED':
        final = []
        for idx, i in enumerate(output):
            print(i)
            try:
                if i == 0:
                    final.append(0)
                elif i == 1 and output[idx+1] == 3 and output[idx+2] == 0:
                    final.append(1)
                elif i == 1 and output[idx+1] == 4 and output[idx+2] == 0:
                    final.append(2)
                elif i == 2 and output[idx+1] == 3 and output[idx+2] == 0:
                    final.append(3)
                elif i == 2 and output[idx+1] == 4 and output[idx+2] == 0:
                    final.append(4)
                elif i == 3 and output[idx+1] == 1 and output[idx+2] == 0:
                    final.append(5)
                elif i == 3 and output[idx+1] == 2 and output[idx+2] == 0:
                    final.append(6)
                elif i == 4 and output[idx+1] == 1 and output[idx+2] == 0:
                    final.append(7)
                elif i == 4 and output[idx+1] == 2 and output[idx+2] == 0:
                    final.append(8)
                else:
                    return 1
            except IndexError:
                print("HERE")
                break
                pass

    elif new == 'B':
        final = output

    else:
        return "Not a valid action space"

    return final
