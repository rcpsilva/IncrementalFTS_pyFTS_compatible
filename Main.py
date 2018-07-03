'''
Created on Jun 30, 2018

@author: rcpsi
'''
from SilvaIncrementalFTS import SilvaIncrementalFTS as sFTS
from pyFTS.data import TAIEX
import numpy as np

def main():
    # Fuzzy set type

    #fts = sFTS(nsets = 7, do_plots = True)

    fts = sFTS(fs_params = [], ftype = 'triang', order = 1, nsets = 7,
                         do_plots = False, par1 = 'test1', par2 = 'test2')

    data = TAIEX.get_data()
    data = list(data[0:1500]) +  list(np.mean(data[0:1500])+data[0:1500]*0.2) + list(np.mean(data[0:1500])+data[0:1500]*0.2) 
    
    fts.train(data[0:2])
    fts.forecast(data[2:len(data)])
    

    fts.print_rules()
        

if __name__ == '__main__':
    main()