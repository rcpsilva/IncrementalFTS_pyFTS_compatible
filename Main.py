'''
Created on Jun 30, 2018

@author: rcpsi
'''
import numpy as np
import pandas as pd
from SilvaIncrementalFTS import SilvaIncrementalFTS as sFTS
<<<<<<< HEAD
from pyFTS.benchmarks import benchmarks as bchmk, arima, naive, quantreg, knn
from pyFTS.data import TAIEX, NASDAQ, SP500, artificial

dataset_names = ["TAIEX", "SP500", "NASDAQ", "IMIV", "IMIV0","CMIV", "IMCV"]
=======
from SilvaIncDistributionRestartFTS import SilvaIncDistributionRestartFTS as rFTS
from pyFTS.data import TAIEX
from matplotlib import pyplot as mplt
import numpy as np
>>>>>>> 33a802d18577660efccd05a241e47ee5a1ec6eb5

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

<<<<<<< HEAD
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
                                        progress=False, type='point',
                                        steps_ahead=[1],
                                        #distributed=True, nodes=['192.168.0.110', '192.168.0.107','192.168.0.106'],\n",
                                        file="benchmarks.db",
                                        dataset=dataset_name,
                                        tag='incremental')
=======
    print('Testing  SilvaIncDistributionRestartFTS')
    
    fts = rFTS(do_plots = True)

    data = TAIEX.get_data()
    #data = data - data[0]
    data = list(data) + list(data*10 - np.mean(data)) + list(data)
    

    fts.train(data[0:2])
    forecasts = fts.forecast(data[2:len(data)])
    
    mplt.plot(np.arange(2,len(data))+1,forecasts,'b')            
    mplt.plot(np.arange(2,len(data)),data[2:len(data)],'r')
    mplt.show()
    
    fts.print_rules()
>>>>>>> 33a802d18577660efccd05a241e47ee5a1ec6eb5
        

if __name__ == '__main__':
    main()