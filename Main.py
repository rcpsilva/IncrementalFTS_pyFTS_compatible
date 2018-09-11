'''
Created on Jun 30, 2018

@author: rcpsi
'''
import numpy as np
import pandas as pd
from SilvaIncrementalFTS import SilvaIncrementalFTS as sFTS
from SilvaIncDistributionRestartFTS import SilvaIncDistributionRestartFTS as rFTS
from SilvaIncKmeansFTS import SilvaIncKmeansFTS as kFTS
# from pyFTS.benchmarks import benchmarks as bchmk, arima, naive, quantreg, knn
from pyFTS.data import TAIEX, NASDAQ, SP500, artificial
from matplotlib import pyplot as mplt

dataset_names = ["TAIEX", "SP500", "NASDAQ", "IMIV", "IMIV0", "CMIV", "IMCV"]


def get_dataset(dataset_name):
    if dataset_name == "TAIEX":
        return TAIEX.get_data()
    elif dataset_name == "NASDAQ":
        return NASDAQ.get_data()
    elif dataset_name == 'IMIV':  # Incremental Mean and Incremental Variance
        return artificial.generate_gaussian_linear(1, 0.2, 0.2, 0.05, it=100, num=50)
    elif dataset_name == 'IMIV0':  # Incremental Mean and Incremental Variance, lower bound equals to 0
        return artificial.generate_gaussian_linear(1, 0.2, 0., 0.05, vmin=0, it=100, num=50)
    elif dataset_name == 'CMIV':  # Constant Mean and Incremental Variance
        return artificial.generate_gaussian_linear(5, 0.1, 0, 0.02, it=100, num=50)
    elif dataset_name == 'IMCV':  # Incremental Mean and Constant Variance
        return artificial.generate_gaussian_linear(1, 0.6, 0.1, 0, it=100, num=50)


def main():
    # Fuzzy set type

    print('Testing  SilvaIncDistributionRestartFTS')

    # fts = kFTS(do_plots = False)
    fts = sFTS(do_plots=True, nsets=9)
    # fts = rFTS(do_plots = False)

    data = get_dataset('TAIEX')
    data = list(data)
    # data = list(data[0:1000]) + list(np.array(data[0:1000]) * 4) + list(data[0:1000]) + list(np.array(data[0:1000])
    #  * 4) 

    # data = data - data[0]
    # data = list(data) + list(data*10 - np.mean(data)) + list(data)

    cut = 5
    print(len(data))
    fts.train(data[0:cut])
    print(fts.centers)
    forecasts = fts.forecast(data[cut:len(data)])

    mplt.plot(np.arange(cut, len(data)) + 1, forecasts, 'b')
    mplt.plot(np.arange(cut, len(data)), data[cut:len(data)], 'r')
    # mplt.plot(np.arange(2,len(data))+1,data[2:len(data)],'g')
    fts.plot_fuzzy_sets(2000, 12000,
                        begin=-500, scale=400, nsteps=1000)
    mplt.show()

    fts.print_rules()


if __name__ == '__main__':
    main()
