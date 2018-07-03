'''
Created on Jun 30, 2018

@author: rcpsi
'''
from SilvaIncrementalFTS2 import SilvaIncrementalFTS as sFTS
from pyFTS.data import TAIEX
import numpy as np

def main():
    print('Hello')
    # Fuzzy set type
    

    fts = sFTS(nsets = 7, do_plots = True)
    #fts.generate_sets(0,6,7)
    #fts.plot_fuzzy_sets(-4,10)
    
    #f_vals = fts.fuzzify(data)
    
    
    #data = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3]
    data = TAIEX.get_data()
    data = list(data[0:1500]) +  list(np.mean(data[0:1500])+data[0:1500]*0.2) + list(np.mean(data[0:1500])+data[0:1500]*0.2) 
    
    fts.train(data[0:2])
    fts.forecast(data[2:len(data)])
    
    
    
    #print(data)
    #print(f_vals)
    fts.print_rules()
    
    

if __name__ == '__main__':
    main()