'''
Created on Jul 9, 2018

@author: rcpsi
'''

#from pyFTS.common import fts
import numpy as np
import time
from SilvaIncrementalFTS import SilvaIncrementalFTS as sIncFTS
import matplotlib.pyplot as mplt


class SilvaIncDistributionRestartFTS(sIncFTS):

    '''
    classdocs
    '''

    def __init__(self, **kwargs):
        ''' Class constructor
    
        Args:
            name:
            shortname:
            fs_params:             fuzzy sets paramenters
            ftype:                 fuzzy set type (FOR NOW IT ONLY IMPLEMENTS TRIANGULAR FUZZY SETSD)
            order:                 FTS order
            nsets:                 number of fuzzy sets
            sigma_multiplier:      used to define the universe of discourse U = [mu - sigma_multiplier * sigma,mu + sigma_multiplier * sigma] 
            do_plots:              plots the time series, forcasts, fuzzy sets and prints the rules to the console
        
        '''
        
        if 'name' not in kwargs:
            kwargs = dict(kwargs, name='SilvaIncDistributionRestartFTS')
            
        if 'shortname' not in kwargs:
            kwargs = dict(kwargs, shortname = 'SIncDResFTS')
            
            
        super(SilvaIncDistributionRestartFTS, self).__init__(**kwargs)
        
    def forecast(self, data, **kwargs):
        
        forecasts = []
        if self.do_plots:
            times = []
            samples = []
            t = 0
        
        for x in data:
            
            if self.do_plots:
                times.append(t)
                samples.append(x)
                mplt.cla()
                
            # 1) update fuzzy sets
            old_centers = self.centers.copy()
            # Update data stats
            
            n = self.data_n + 1 
            newmean = self.data_mu + (x - self.data_mu)/n
            var = self.data_sigma**2
            newstd =  np.sqrt( (n-2)/(n-1) * var + (1/n) * (x - self.data_mu)**2)
            
            
            # if there is a significant change in the distribution \alpha = 0.05 restart the learning process
            if (x > (self.data_mu + self.sigma_multiplier * self.data_sigma)) or (x < (self.data_mu - self.sigma_multiplier * self.data_sigma)): 
                # Reset
                self.train([self.lastx,x])
            else:
                self.data_mu = newmean;
                self.data_sigma = newstd;       
                self.data_max = np.maximum(self.data_max,x)
                self.data_min = np.minimum(self.data_min,x)
                self.data_n += 1
            
            
            # Update sets
            
            bounds = self.update_bounds()
            lb = bounds[0]
            ub = bounds[1]
            self.generate_sets(lb,ub,self.nsets)
            
            
            # 2) Update rules
            self.update_rules(old_centers)
            
            if self.do_plots:
                print('====================')
                self.print_rules()
            
            #3) Add latest rule
            # Fuzzify
            
            # Update
            ## Update rules with the new point
            antecendent = self.fuzzify([self.lastx])
            consequent = self.fuzzify([x])
                       
            self.rules[antecendent[0]].update(consequent)
            
            ## Update current state
            ### Convert back to lists 
            self.rules = [list(r) for r in self.rules]
            
            self.lastx = x.copy()
            
            # 3) Forecast
            forecasts.append(self.forecast_weighted_average([x]))
            
            # plots
            if self.do_plots:
                self.plot_fuzzy_sets(500,30000,
                                 begin = -500, scale = 400, nsteps = 1000)
                
                mplt.plot(np.array(times)+1,forecasts,'b')
                mplt.draw()
                mplt.plot(times,samples,'r')
                mplt.draw()
                mplt.pause(1e-17)
                time.sleep(1e-8)
                t += 1 
        
        if self.do_plots:
            mplt.show()
        return forecasts 
        