from pyFTS.common import fts
import numpy as np
import matplotlib.pyplot as mplt
import time
import skfuzzy.defuzzify as defuzz


class IncAnyOrderFTS(fts.FTS):

    def __init__(self, **kwargs):
        """ Class constructor

                Args:
                    fs_params:             fuzzy sets parameters
                    ftype:                 fuzzy set type (FOR NOW IT ONLY IMPLEMENTS TRIANGULAR FUZZY SETSD)
                    order:                 FTS order
                    nsets:                 number of fuzzy sets
                    sigma_multiplier:      used to define the universe of discourse U = [mu - sigma_multiplier * sigma,mu + sigma_multiplier * sigma]
                    do_plots:              plots the time series, forcasts, fuzzy sets and prints the rules to the console

        """

        # Handling super class constructor
        if 'name' not in kwargs:
            kwargs = dict(kwargs, name='AnyOrderIncrementalFTS')

        if 'shortname' not in kwargs:
            kwargs = dict(kwargs, shortname='AOI-FTS')

        if 'order' not in kwargs:
            kwargs = dict(kwargs, order=1)

        if 'max_lag' not in kwargs:
            kwargs = dict(kwargs, max_lag=1)

        super(IncAnyOrderFTS, self).__init__(**kwargs)

        ''' Initialization of the incremental part

               Args:
                   fs_params:             fuzzy sets paramenters
                   ftype:                 fuzzy set type (FOR NOW IT ONLY IMPLEMENTS TRIANGULAR FUZZY SETSD)
                   order:                 FTS order
                   nsets:                 number of fuzzy sets
                   sigma_multiplier:      used to define the universe of discourse U = [mu - sigma_multiplier * sigma,mu + sigma_multiplier * sigma] 
                   do_plots:              plots the time series, forcasts, fuzzy sets and prints the rules to the console
        '''

        self.do_plots = kwargs.get('do_plots', False)
        self.fs_params = kwargs.get('fs_params', []) # Fuzzy set parameters
        self.ftype = kwargs.get('ftype', 'triang') # Type of fuzzy set (For nor now it only implements triangular fuzzy sets )
        self.order = kwargs.get('order', 1) # FTS order (For now it only implements first order FTSs)
        self.sigma_multiplier = kwargs.get('sigma_multiplier', 2.326) # Defines the universe of discourse extent
        self.nsets = kwargs.get('nsets', 7)  # number of fuzzy sets

        self.centers = []  # Fuzzy sets centers
        self.rules = []  # Fuzzy logic relationships

        # Stores data stats
        self.lastx = []  # Last n seen samples
        self.data_mu = 0  # Data mean
        self.data_sigma = 0  # Data standard deviation
        self.data_n = 0  # Total number of samples
        self.data_max = 0 # Data minimum
        self.data_min = 0 # Data maximum

    def build_fuzzy_sets(self):
        pass

    def update_rules(self):
        pass

    def train(self):
        pass

    def forecast(self, data, **kwargs):
        pass

    def forecast_distribution(self, data, **kwargs):
        pass

    def forecast_ahead_interval(self, data, steps, **kwargs):
        pass

    def forecast_ahead_distribution(self, data, steps, **kwargs):
        pass
