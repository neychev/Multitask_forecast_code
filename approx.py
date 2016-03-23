import numpy as np
from __future__ import division

tmp1 = np.genfromtxt('data/GermanSpotPrice.csv', delimiter=',')
tmp2 = np.genfromtxt('data/one.csv', delimiter=',')
tmp3 = np.genfromtxt('data/GermanWeather.csv', delimiter=',')

ts1 = tmp1[0:len(tmp2), 1:]
ts2 = tmp2
ts3 = tmp3[0:len(tmp2), 1:]

ts1 = ts1.flatten()
ts2 = ts2.flatten()
ts3 = ts2.flatten()

time_array = np.arange(len(ts1))
time_array_short = np.arange(0, len(ts2), 24)
dict_ts1 = {'time_series' : ts1, 'time_mask' : time_array, 'time_step' : 1}
dict_ts3 = {'time_series' : ts3, 'time_mask' : time_array, 'time_step' : 1}
dict_ts2 = {'time_series' : ts2, 'time_mask' : time_array_short, 'time_step' : 24}

def demo_sampler(time_spot, dict_ts):
    point = np.min(np.where(time_spot < dict_ts['time_mask']))
    if (time_spot - point < dict_ts['time_step']/2):
        val = dict_ts['time_series'][point]
    else:
        val = dict_ts['time_series'][point + 1]
    return val

check = demo_sampler(time_spot=75, dict_ts=dict_ts2)

def create_sample(Tp, Tr, Ti, tau, dict_ts):
    if (Tp + Tr) % tau <> 0:
        return -1
    begin_point = Ti - (Tp + Tr);
    sample_p = np.array([])
    sample_r = np.array([])
    for i in range(begin_point, begin_point + Tp, tau):
        sample_p = np.append(sample_p, demo_sampler(i, dict_ts))
    for i in range(begin_point + Tp, Ti, tau):
        sample_r = np.append(sample_r, demo_sampler(i, dict_ts))
    return {'prehistory' : sample_p, 'horizon' : sample_r}

tmp = np.array([])
for i in range(620, 840, 1):
    temp2 = create_sample(6*24, 24, i, 3, dict_ts2)
    tmp = np.append(tmp, len(temp2['horizon']))

print(np.max(tmp) - np.min(tmp))

# TODO: Check with bad lengths!
