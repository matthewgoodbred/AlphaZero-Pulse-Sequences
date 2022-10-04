'''
4/25/22
Owen Eskandari
Thesis work 22S

This is a class which (hopefully) contains all of the necessary information associated with each pulse sequence.

The main constituents are as follows (- = uncompleted, + = completed):

-Naming system
+Access to pulse sequence (the pulses themselves)
+Length (in tau)
+Originally found action space
-Possible action space representations TODO: This will be hard to implement
+When originally found
    +Line of ThesisResults_OE.xlsx spreadsheet
    +and subsequent times found with relevant lines
+How many times found for a given run
    -Make run specific
-Results
    -Fidelity for one regime
        -need a definition for this (possible options below)
            -3 measurements: no error, some, a lot
            -granular plot information
    -Markers for exceptional sequences
        -Within a specific parameter regime (low coupling strength, etc.)
            -Robust
            -Not robust
            -Outperforms CORY48 or yxx24
                -Small error regime
                -Large error regime
                -Both
            -For which error does it outperform CORY48 or yxx24
+Matrix representation function
-Error plotting function
    -Should this be able to make one plot for multiple sequences?
    -Or compare to CORY48 and yxx24
'''

from Convert import convert
from Matrix_Representation_Standalone import visual_representation
# from ActionSpace import ActionSpace

pulse_list_dict = {}
pulse_list_dict['O'] = [['D'], ['X'], ['-X'], ['Y'], ['-Y']]
pulse_list_dict['SE'] = [['D'], ['X', 'Y'], ['X', '-Y'], ['-X', 'Y'], ['-X', '-Y'],
                         ['Y', 'X'], ['Y', '-X'], ['-Y', 'X'], ['-Y', '-X']]
pulse_list_dict['SED'] = [['X', 'Y', 'D'], ['X', '-Y', 'D'], ['-X', 'Y', 'D'], ['-X', '-Y', 'D'],
                          ['Y', 'X', 'D'], ['Y', '-X', 'D'], ['-Y', 'X', 'D'], ['-Y', '-X', 'D']]
pulse_list_dict['SEDD'] = [['D'], ['X', 'Y', 'D'], ['X', '-Y', 'D'], ['-X', 'Y', 'D'], ['-X', '-Y', 'D'],
                          ['Y', 'X', 'D'], ['Y', '-X', 'D'], ['-Y', 'X', 'D'], ['-Y', '-X', 'D']]
pulse_list_dict['B'] = [['D'], ['X', 'Y'], ['X', '-Y'], ['-X', 'Y'], ['-X', '-Y'],
                        ['Y', 'X'], ['Y', '-X'], ['-Y', 'X'], ['-Y', '-X'],
                        ['X', 'Y', 'D'], ['X', '-Y', 'D'], ['-X', 'Y', 'D'], ['-X', '-Y', 'D'],
                        ['Y', 'X', 'D'], ['Y', '-X', 'D'], ['-Y', 'X', 'D'], ['-Y', '-X', 'D']]


class PulseSequence:

    def __init__(self, ps, row_info, number, count, actionspace=None, name=None):
        '''

        :param ps: List. List of pulses in the sequence.
        :param row_info: List. List of a row from a Pandas datafile with information regarding the runs
        :param number: Int. Number of the pulse sequence found during the run
        :param counts: Int. Number of times the sequence was found during the run
        :param actionspace: Str or None. This overrides the row_info[12] command if not None.
        :param actionspace: Str or None. This overrides the self.name command if not None.
        '''

        self.timesfound = 1     # This is the number of distinct runs in which the pulse sequence was found

        # Name = jobid + target length + action space + number
        if name is None:
            self.names = [str(row_info[1]) + '_' + str(row_info[11]) + '_' + str(row_info[12]) + '_' + str(number)]
        else:
            self.names = [name]

        self.numbers = [number]

        if actionspace is None:
            self.pulse_sequence = convert(ps, row_info[12], 'O')
        else:
            self.pulse_sequence = convert(ps, actionspace, 'O')
        if self.pulse_sequence == 0 or self.pulse_sequence == 1:
            print(ps)
            raise ValueError        # One or more pulses cannot be represented in the original action space
        self.length = len(self.pulse_sequence)

        self.counts = [count]      # Number of counts for a given run (TODO need to make this run specific

        if row_info is None:
            self.runvals = [[None for i in range(25)]]
        else:
            self.runvals = [row_info]

        # Some initial output data storage for now
        self.no_error_fid = None
        self.rot_error_fids = None
        self.pte_fids = None
        self.offset_fids = None

        # self.jobids = [self.runvals[0][1]]        # Not needed now, but coded if needed in the future
        # self.names = [self.runvals[0][2]]
        # self.start_dates = [self.runvals[0][3]]
        # self.end_dates = [self.runvals[0][5]]
        # self.start_times = [self.runvals[0][4]]
        #
        # # Get the total time run in seconds
        # time = 0
        # times = self.runvals[0][6].split(':')
        # for idx, i in enumerate(times):
        #     time += 60 ** (len(times) - 2 - idx) * int(i)
        # self.runtimes = [int(time)]
        #
        # maxtime = 0
        # times = self.runvals[0][7].split(':')
        # for idx, i in enumerate(times):
        #     maxtime += 60 ** (len(times) - 2 - idx) * int(i)
        # self.maxtimes = [int(maxtime)]
        #
        # self.cpus = [self.runvals[0][8]]
        # self.terminate_reasons = [self.runvals[0][9]]
        # self.runfiles = [self.runvals[0][10]]
        # self.target_lengths = [self.runvals[0][11]]
        # self.actionspaces = [ActionSpace(pulse_sequence=self.pulse_sequence, orig=[self.runvals[0][12]])]
        # # Now find all possible action space representations TODO
        #
        # # Physical system parameters
        # self.dipolar_strengths = [self.runvals[0][13]]
        # self.tau_lengths = [self.runvals[0][14]]
        # self.pulse_widths = [self.runvals[0][15]]
        # self.rot_errs = [[self.runvals[0][16]]]
        # self.ptes = [self.runvals[0][17]]
        # self.offset_errs = [self.runvals[0][18]]
        # self.Ns = [self.runvals[0][19]]
        # self.ensemble_sizes = [self.runvals[0][20]]
        #
        # # ML Parameter
        # self.pbcs = [self.runvals[0][21]]
        #
        # self.runfolders = [self.runvals[0][22]]
        # self.histograms = [self.runvals[0][23]]
        # self.notes = [self.runvals[0][24]]

    def __str__(self):      # What prints when the object is printed
        strpsname = []  # This is a string of the pulse sequence
        for psname in self.pulse_sequence:  # For each pulse in the sequence
            # For each pulse in the sequence, find its basic pulses {D, X, -X, Y,-Y} and add to a list
            strpsname.append(''.join([pulse for pulse in pulse_list_dict['O'][psname]]))
        pulses = ','.join([idx for idx in strpsname])  # Join all pulses together as a string
        return pulses

    # def tick(self):
    #     '''
    #     Increase count by 1
    #     :return: None
    #     '''
    #     self.count += 1

    def matrix_representation(self, name='Matrix_representation', folder=None, plot=True):
        '''
        Create a matrix form of the pulse sequence. If plot=False, then axis counts are returned.
        :param name: Str. Name of the saved figure file.
        :param folder: Str. Path to the folder to use to save the figure
        :param plot: Bool. True --> plot. False --> axis counts are returned
        :return: Plot or axis counts
        '''
        return visual_representation(pulse_sequence=self.pulse_sequence, action_space='O', plot=plot,
                                     name=name, folder=folder)

    # Below are the functions for the relevant information from self.runvals
    def get_orig_jobid(self):
        return self.runvals[0][1]

    def get_jobids(self):
        jobids = []
        for i in self.runvals:
            jobids.append(i[1])
        return jobids

    def get_orig_runname(self):
        return self.runvals[0][2]

    def get_runnames(self):
        names = []
        for i in self.runvals:
            names.append(i[2])
        return names

    def get_orig_start_date(self):
        return self.runvals[0][3]

    def get_start_dates(self):
        dates = []
        for i in self.runvals:
            dates.append(i[3])
        return dates

    def get_orig_end_date(self):
        return self.runvals[0][5]

    def get_end_dates(self):
        dates = []
        for i in self.runvals:
            dates.append(i[5])
        return dates

    def get_orig_start_time(self):
        return self.runvals[0][4]

    def get_start_times(self):
        time = []
        for i in self.runvals:
            time.append(i[4])
        return time

    def get_orig_runtime(self):
        time = 0
        times = self.runvals[0][6].split(':')
        for idx, i in enumerate(times):
            time += 60 ** (len(times) - 2 - idx) * int(i)
        self.runtimes = [int(time)]
        return int(time)

    def get_runtimes(self):
        runtimes = []
        for runval in self.runvals:
            time = 0
            times = runval[6].split(':')
            for idx, i in enumerate(times):
                time += 60 ** (len(times) - 2 - idx) * int(i)
            runtimes.append(int(time))
        return runtimes

    def get_orig_maxtime(self):
        time = 0
        times = self.runvals[0][7].split(':')
        for idx, i in enumerate(times):
            time += 60 ** (len(times) - 2 - idx) * int(i)
        self.runtimes = [int(time)]
        return int(time)

    def get_maxtimes(self):
        maxtimes = []
        for runval in self.runvals:
            time = 0
            times = runval[7].split(':')
            for idx, i in enumerate(times):
                time += 60 ** (len(times) - 2 - idx) * int(i)
            maxtimes.append(int(time))
        return maxtimes

    def get_orig_cpus(self):
        return self.runvals[0][8]

    def get_cpus(self):
        cpus = []
        for i in self.runvals:
            cpus.append(i[8])
        return cpus

    def get_orig_terminate_reason(self):
        return self.runvals[0][9]

    def get_terminate_reasons(self):
        reasons = []
        for i in self.runvals:
            reasons.append(i[9])
        return reasons

    def get_orig_runfile(self):
        return self.runvals[0][10]

    def get_runfiles(self):
        files = []
        for i in self.runvals:
            files.append(i[10])
        return files

    def get_orig_target_length(self):
        return self.runvals[0][11]

    def get_target_lengths(self):
        lengths = []
        for i in self.runvals:
            lengths.append(i[11])
        return lengths

    def get_orig_action_space(self):
        return self.runvals[0][12]

    def get_action_spaces(self):
        acts = []
        for i in self.runvals:
            acts.append(i[12])
        return acts

    # Physical system parameters
    def get_orig_dip_strength(self):
        return self.runvals[0][13]

    def get_dip_strengths(self):
        dips = []
        for i in self.runvals:
            dips.append(i[13])
        return dips

    def get_orig_tau_length(self):
        return self.runvals[0][14]

    def get_tau_lengths(self):
        taus = []
        for i in self.runvals:
            taus.append(i[14])
        return taus

    def get_orig_pulse_width(self):
        return self.runvals[0][15]

    def get_pulse_widths(self):
        ws = []
        for i in self.runvals:
            ws.append(i[15])
        return ws

    def get_orig_rot_error(self):
        return self.runvals[0][16]

    def get_rot_errors(self):
        res = []
        for i in self.runvals:
            res.append(i[16])
        return res

    def get_orig_pte(self):
        return self.runvals[0][17]

    def get_ptes(self):
        ptes = []
        for i in self.runvals:
            ptes.append(i[17])
        return ptes

    def get_orig_offset_error(self):
        return self.runvals[0][18]

    def get_offset_errors(self):
        oers = []
        for i in self.runvals:
            oers.append(i[18])
        return oers

    def get_orig_N(self):
        return self.runvals[0][19]

    def get_Ns(self):
        Ns = []
        for i in self.runvals:
            Ns.append(i[19])
        return Ns

    def get_orig_ensemble_size(self):
        return self.runvals[0][20]

    def get_ensemble_sizes(self):
        ess = []
        for i in self.runvals:
            ess.append(i[20])
        return ess

    # ML Parameter
    def get_orig_pbc(self):
        return self.runvals[0][21]

    def get_pbcs(self):
        pbcs = []
        for i in self.runvals:
            pbcs.append(i[21])
        return pbcs

    def get_orig_runfolder(self):
        return self.runvals[0][22]

    def get_runfolders(self):
        folders = []
        for i in self.runvals:
            folders.append(i[22])
        return folders

    def get_orig_hist(self):
        return self.runvals[0][23]

    def get_hists(self):
        hists = []
        for i in self.runvals:
            hists.append(i[23])
        return hists

    def get_orig_note(self):
        return self.runvals[0][24]

    def get_notes(self):
        notes = []
        for i in self.runvals:
            notes.append(i[24])
        return notes

    # Other important callable functions
    def get_ps(self):
        return self.pulse_sequence

    def get_length(self):
        return self.length

    def get_orig_count(self):
        return self.counts[0]

    def get_counts(self):
        return self.counts

    def get_orig_num(self):
        return self.numbers[0]

    def get_nums(self):
        return self.numbers

    def get_timesfound(self):
        return self.timesfound

    def get_orig_name(self):
        return self.names[0]

    def get_names(self):
        return self.names

    def get_orig_row_data(self):
        return self.runvals[0]

    def get_row_datas(self):
        return self.runvals

    def help(self):
        '''
        Give a list of the possible options of stuff to look for
        :return:
        '''

        print('jobid, runname, start date, end date, start time, runtime, maxtime, cpus, terminate reason, ' 
              'runfile, target length, action space, dip strength, tau length, pulse width, rot error, pte, '
              'offset error, N, ensemble size, pbc, runfolder, hist, note')
        print('ps, length, count, num, timesfound, name, row data')

    # def copy(self):
    #     return PulseSequence(
    #         ps=1,
    #         row_info=self.runvals,
    #
    #     )

    # Adding another run into the object (the pulse sequence has been found during another run)
    def add_run(self, row_data, num, cts, actionspace=None, name=None):
        self.runvals.append(row_data)
        self.timesfound += 1
        if name is None:
            self.names.append(str(row_data[1]) + '_' + str(row_data[11]) + '_' + str(row_data[12]) + '_' + str(num))
        else:
            self.names.append(name)
        self.counts.append(cts)
        self.numbers.append(num)

    # Editing fidelity information
    def set_no_error_fid(self, fid):
        self.no_error_fid = fid

    def set_rot_error_fids(self, fids):
        '''

        :param fid: List of 5 elements
        :return:
        '''
        self.rot_error_fids = fids

    def set_pte_fids(self, fids):
        '''

        :param fid: List of 5 elements
        :return:
        '''
        self.pte_fids = fids

    def set_offset_fids(self, fids):
        '''

        :param fids: List of 9 elements
        :return:
        '''
        self.offset_fids = fids

    def get_no_error_fid(self):
        return self.no_error_fid

    def get_rot_error_fids(self):
        return self.rot_error_fids

    def get_pte_fids(self):
        return self.pte_fids

    def get_offset_fids(self):
        return self.offset_fids


# # Results from the Excel spreadsheet of runs
# df = pd.read_csv('/Users/oweneskandari/Desktop/ThesisResults_OE.csv')
# row_info = df.loc[df['Folder name'] == '20211128-113439'].values.tolist()[0]
#
# # Testing
# row1 = '20211128-113439'
# row2 = '20210930-231845'
# csv = '/Users/oweneskandari/Desktop/ThesisResults_OE.csv'
#
# df = pd.read_csv(csv)
# row_info1 = pd.read_csv(csv).loc[df['Folder name'] == row1].values.tolist()[0]
# row_info2 = pd.read_csv(csv).loc[df['Folder name'] == row2].values.tolist()[0]
#
#
# a = PulseSequence([0, 1, 3, 2], row_info1, 1, 100)
#
# print(a.get_ps())
# print(a)
# print(a.get_maxtimes())
#
# a.add_run(row_info2, 1, 12)
# print(a.get_ps())
# print(a)
# print(a.get_maxtimes())
#
# print(a.get_orig_name())




