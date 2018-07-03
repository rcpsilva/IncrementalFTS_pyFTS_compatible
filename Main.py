'''
Created on Jun 30, 2018

@author: rcpsi
'''
from SilvaIncrementalFTS import SilvaIncrementalFTS as sFTS
from pyFTS.benchmarks import benchmarks as bc 
from pyFTS.data import TAIEX


def main():
    # Fuzzy set type

    #fts = sFTS(do_plots = False)

    data = TAIEX.get_data()

    
    fts.train(data[0:2])
    fts.forecast(data[2:len(data)])
    

    fts.print_rules()
        

if __name__ == '__main__':
    main()