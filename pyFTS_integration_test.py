'''
Created on Jul 10, 2018

@author: rcpsi
'''

import numpy as np
import pandas as pd
from SilvaIncrementalFTS import SilvaIncrementalFTS as sFTS
from SilvaIncDistributionRestartFTS import SilvaIncDistributionRestartFTS as rFTS
from SilvaIncKmeansFTS import SilvaIncKmeansFTS as kFTS
from pyFTS.benchmarks import benchmarks as bchmk, arima, naive, quantreg, knn
from pyFTS.data import TAIEX, NASDAQ, SP500, artificial
from matplotlib import pyplot as mplt


dataset_names = ["TAIEX", "SP500", "NASDAQ", "IMIV", "IMIV0","CMIV", "IMCV"]

benchmark_methods= [arima.ARIMA for k in range(3)] + [naive.Naive] + [quantreg.QuantileRegression for k in range(2)]

benchmark_methods_parameters= [
        {'order': (1, 0, 0)},
        {'order': (1, 0, 1)},
        {'order': (2, 0, 1)},
        #{'order': (2, 0, 2)},
        {},
        {'order': 1, 'alpha': .5},
        {'order': 2, 'alpha': .5},
    ]

def get_dataset(dataset_name):
    if dataset_name == "TAIEX":
        return TAIEX.get_data()
    elif dataset_name == "NASDAQ":
        return NASDAQ.get_data()
    elif dataset_name == 'IMIV': # Incremental Mean and Incremental Variance
        return artificial.generate_gaussian_linear(1,0.2,0.2,0.05,it=100, num=50)
    elif dataset_name == 'IMIV0': # Incremental Mean and Incremental Variance, lower bound equals to 0
        return artificial.generate_gaussian_linear(1,0.2,0.,0.05, vmin=0,it=100, num=50)
    elif dataset_name == 'CMIV': # Constant Mean and Incremental Variance
        return artificial.generate_gaussian_linear(5,0.1,0,0.02,it=100, num=50)
    elif dataset_name == 'IMCV': # Incremental Mean and Constant Variance
        return artificial.generate_gaussian_linear(1,0.6,0.1,0,it=100, num=50)

def main():

    fts = sFTS(do_plots=False)
 
    for dataset_name in dataset_names:
        dataset = get_dataset(dataset_name)
        bchmk.sliding_window_benchmarks(dataset, 1000, train=0.5, inc=0.2,
                                         benchmark_models=True,
                                         benchmark_methods=benchmark_methods,
                                         benchmark_methods_parameters=benchmark_methods_parameters,
                                         models = [fts],
                                         build_methods = False,
                                         transformations=[None],
                                         orders=[1,2,3],
                                         partitions=[35], # np.arange(10,100,2),
                                         progress=False, 
                                         type='point',
                                         steps_ahead=[1],
                                         distributed=True, 
                                         nodes = ['192.168.0.110', '192.168.0.107','192.168.0.106'],
                                         file="benchmarks.db",
                                         dataset=dataset_name,
                                         tag='incremental')

if __name__ == '__main__':
    main()