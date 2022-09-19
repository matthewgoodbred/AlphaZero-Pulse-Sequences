'''
6/20/22
Owen Eskandari

Takes a sequence in the original action space and outputs it in the format Wynter uses

Two components: sequence and delays
'''

from Convert import convert

def WynterInput(seq):
    sequence = '{'
    delays = [1]

    for idx,ps in enumerate(seq):
        if ps == 1:
            sequence += 'X, '
            delays.append(1)
        if ps == 2:
            sequence += '-X, '
            delays.append(1)
        if ps == 3:
            sequence += 'Y, '
            delays.append(1)
        if ps == 4:
            sequence += '-Y, '
            delays.append(1)
        if ps == 0 and idx != len(seq)-1:
            delays[-1] += 1

    sequence = sequence[:len(sequence)-2]
    sequence += '}'

    return sequence, delays

# seq1 = convert([4, 3, 3, 8, 6, 8, 1, 2, 2, 5, 7, 5], 'SEDD', 'O')
# print(seq1)
# seq1 = [2, 4, 0, 2, 3, 0, 2, 3, 0, 4, 2, 0, 3, 2, 0, 4, 2, 0, 1, 3, 0, 1, 4, 0, 1, 4, 0, 3, 1, 0, 4, 1, 0, 3, 1, 0, 1, 3, 0, 3, 1, 0, 3, 1, 0, 4, 1, 0, 2, 3, 0, 2, 4, 0, 2, 4, 0, 2, 3, 0, 4, 1, 0, 4, 2, 0, 4, 2, 0, 1, 3, 0]

SED24 = [1, 3, 0, 2, 4, 0, 3, 2, 0, 3, 2, 0, 3, 1, 0, 2, 3, 0, 1, 3, 0, 2, 4, 0, 3, 1, 0, 4, 2, 0, 4, 2, 0, 1, 4, 0]
SED48 = [2, 4, 0, 2, 3, 0, 2, 3, 0, 4, 2, 0, 3, 2, 0, 4, 2, 0, 1, 3, 0, 1, 4, 0, 1, 4, 0, 3, 1, 0, 4, 1, 0, 3, 1, 0,
         1, 3, 0, 3, 1, 0, 3, 1, 0, 4, 1, 0, 2, 3, 0, 2, 4, 0, 2, 4, 0, 2, 3, 0, 4, 1, 0, 4, 2, 0, 4, 2, 0, 1, 3, 0]
SED96 = [2, 3, 0, 4, 2, 0, 3, 2, 0, 1, 3, 0, 4, 1, 0, 2, 4, 0, 1, 4, 0, 1, 4, 0, 3, 2, 0, 2, 4, 0, 3, 1, 0, 3, 1, 0,
         3, 1, 0, 3, 2, 0, 2, 3, 0, 4, 2, 0, 4, 1, 0, 1, 4, 0, 1, 3, 0, 2, 4, 0, 1, 4, 0, 3, 2, 0, 4, 2, 0, 1, 3, 0,
         3, 1, 0, 1, 3, 0, 2, 3, 0, 1, 3, 0, 3, 1, 0, 3, 2, 0, 4, 2, 0, 1, 4, 0, 2, 4, 0, 1, 4, 0, 4, 2, 0, 4, 1, 0,
         3, 2, 0, 1, 3, 0, 1, 3, 0, 2, 3, 0, 4, 2, 0, 4, 1, 0, 4, 2, 0, 4, 1, 0, 2, 3, 0, 2, 4, 0, 2, 4, 0, 3, 1, 0]

print(WynterInput(SED24))
print(WynterInput(SED48))
print(WynterInput(SED96))


