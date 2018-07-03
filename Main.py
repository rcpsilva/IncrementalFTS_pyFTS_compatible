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
    order = 1
    fts = sFTS(do_plots = False, order = 1)

    data = TAIEX.get_data()
    data = list(data[0:1500]) + list(data[0:1500]*3 - 5000) + list(data[0:1500])  
    
    fts.train(data[0:(order+1)])
    fts.forecast(data[2:len(data)])
    

    fts.print_rules()
        

if __name__ == '__main__':
    main()