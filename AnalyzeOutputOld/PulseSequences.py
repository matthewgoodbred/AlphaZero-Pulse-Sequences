from PulseSequence import *


class PulseSequences:

    def __init__(self):
        self.ps_dict = {}

        self.ps_obj_list = []

        self.ps_obj_repeated = []

    def get_ps_dict(self):
        return self.ps_dict

    def add_ps_object(self, ps_obj):
        key = str(ps_obj.get_ps())
        if key in self.ps_dict:
            self.ps_dict[key].append(ps_obj)
        else:
            self.ps_dict[key] = [ps_obj]

    def get_repeated_sequence_names(self):
        repeated = []
        for key in self.ps_dict:
            if len(self.ps_dict[key]) > 1:
                repeated.append(self.ps_dict[key][0].get_orig_name())

        return repeated

    def make_entries_singular(self):
        self.ps_obj_repeated = []           # Reset this to make sure it's empty
        self.ps_obj_list = []
        for key in self.ps_dict:

            if len(self.ps_dict[key]) > 1:
                for idx, value in enumerate(self.ps_dict[key]):
                    if idx != 0:
                        row_data = self.ps_dict[key][idx].get_orig_row_data()
                        num = self.ps_dict[key][idx].get_orig_num()
                        counts = self.ps_dict[key][idx].get_orig_count()
                        if row_data[0] is not None:
                            self.ps_dict[key][0].add_run(row_data, num, counts)      # row data, num, counts
                        else:
                            name = self.ps_dict[key][idx].get_orig_name()
                            self.ps_dict[key][0].add_run(row_data, num, counts, actionspace='O', name=name)

                self.ps_dict[key] = [self.ps_dict[key][0]]
                self.ps_obj_repeated.append(self.ps_dict[key][0])

            elif self.ps_dict[key][0].get_timesfound() > 1:     # If make_entries_singular() has already been called
                if self.ps_dict[key][0] not in self.ps_obj_repeated:
                    self.ps_obj_repeated.append(self.ps_dict[key][0])

            self.ps_obj_list.append(self.ps_dict[key][0])       # Add regardless

    def get_ps_obj_list(self):
        '''
        List is only populated after make_entries_singular() is called
        :return: List of PS objects
        '''
        return self.ps_obj_list

    def get_repeated_objects(self):
        '''
        List is only populated after make_entries_singular() is called
        :return: List of PS objects
        '''
        return self.ps_obj_repeated

    def get_length(self):
        '''
        Gets the length of the number of independent objects
        :return: Int. Number of independent pulse sequences
        '''
        return len(self.ps_obj_list)

    def find_name(self, name):
        '''
        Find a PS object with the name given as the parameter
        :param name: Str. Name of sequence to find
        :return: PS object if found, None if not found
        '''
        for key in self.ps_dict:
            for idx, value in enumerate(self.ps_dict[key]):
                if name in self.ps_dict[key][idx].get_names():
                    return self.ps_dict[key][idx]
        return None

    def __str__(self):
        return 'Pulse Sequences Object'
