import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as mpp

data = np.genfromtxt('Time_sample.csv', delimiter=',')
col_mean = stats.nanmean(data, axis = 0)
print col_mean
idxs = np.where(np.isnan(data))

data[idxs] = np.take(col_mean, idxs[1])


tmp = np.where(np.isnan(data))

col_nums = np.random()
len = 90
for i in [1, 24, 37, 54, 88]:
    test = data[:, i]
    mpp.plot(test[1:len])