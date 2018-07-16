'''
Created on Jul 9, 2018

@author: rcpsi
'''

#from pyFTS.common import fts
import numpy as np
import time
from SilvaIncrementalFTS import SilvaIncrementalFTS as sIncFTS
import matplotlib.pyplot as mplt


class SilvaIncKmeansFTS(sIncFTS):

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
            
            
        super(SilvaIncKmeansFTS, self).__init__(**kwargs)
        
        self.cluster_counts = np.ones(self.nsets) #number of points at each cluster
    
    def update_sets(self,x):
        ''' Incremental kmeans update
    
        Args:
            x: current values
            n: cluster count
            
        '''
        # Compute the distances from the current centers
        dists = [np.linalg.norm(x-c) for c in self.centers]
        
        # Update the current centers
        closest_center = np.argmin(dists)
        
        
        n = self.cluster_counts[closest_center]
        self.centers[closest_center] = (self.centers[closest_center]*n + x)/(n+1) 
        self.cluster_counts[closest_center] += 1
        
        # Re-generate set paramaters  
        self.fs_params = []
        
        self.fs_params.append([-np.inf,self.centers[0], self.centers[1]])
        
        for i in np.arange(1,(len(self.centers)-1)):
            self.fs_params.append([self.centers[i-1],self.centers[i],self.centers[i+1]])
        
        self.fs_params.append([self.centers[len(self.centers)-2], self.centers[len(self.centers)-1],np.inf])
        
        
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
                
            if x < self.data_max and x > self.data_min:   
                
                # Update sets
                self.update_sets(x)
                
                # 2) Update rules
                self.update_rules(old_centers)
            
            else: # Reset fuxxy sets
                self.data_max = np.maximum(self.data_max,x)
                self.data_min = np.minimum(self.data_min,x)
                
                self.cluster_counts = np.ones(self.nsets)
                self.generate_sets(self.data_min, self.data_max, self.nsets)
                
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
            forecasts.append(self.forecast_weighted_average_method([x]))
            
            # plots
            if self.do_plots:
                self.plot_fuzzy_sets(500,40000,begin = -500, scale = 400, nsteps = 1000)
                
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
        