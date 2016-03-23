#here will be draft of multiscale time series predictions
#Electicity consumption + weather in Germany

import numpy as np
import pandas as pd
from __future__ import division
import pickle
import matplotlib.pyplot as mpl
import sklearn.linear_model as skl

path = "data/"

#Loading time series. For now, length and period are hardcoded.
ElectricityTS = np.genfromtxt(path+'GermanSpotPrice.csv', delimiter=',')
WeatherTS = np.genfromtxt(path+'GermanWeather.csv', delimiter=',')

#Reshaping and dividing to X and Y.
tmp_el = ElectricityTS[0:250*7, 1:]
tmp_wth = WeatherTS[0:250*7, 1:]
matrix_el = np.reshape(tmp_el, newshape=[tmp_el.size/(24*7), 24*7])

dates = range(0, 250*7)
dates_drop = range(6, 250*7, 7)
idxs_X = np.setxor1d(dates, dates_drop)
idxs_Y = dates_drop

tmp2_el = np.array([tmp_el[i, :] for i in idxs_X])
tmp2_wth = np.array([tmp_wth[i, :] for i in idxs_X])

tmp2_el = np.reshape(tmp2_el, [tmp2_el.shape[0]/6, tmp2_el.shape[1] * 6])
tmp2_wth = np.reshape(tmp2_wth, [tmp2_wth.shape[0]/6, tmp2_wth.shape[1] * 6])

#myX - object-features matrix, myY - answer matrix, W - weights
myX = np.concatenate([tmp2_el, tmp2_wth], axis = 1)
myY = np.array([tmp_el[i,:] for i in idxs_Y])


import prediction_model_check

check_idxs = np.random.random_integers(0, 250, 10)
std_dev_arr = np.zeros(check_idxs.shape)

list_of_models = ['lin_reg', 'Elastic_net', 'Lasso']
result = {}

for _mode in list_of_models:
    my_iterator = 0
    result_dict = {}
    for idx in check_idxs:
        result_dict[my_iterator] = []
        result_dict[my_iterator].append(prediction_model_check(myX, myY, idx, _mode))
        my_iterator += 1
    result[_mode] = []
    result[_mode].append(result_dict)


def get_mean_W(result, list_of_models):
    model_mean_W = {}
    for _mode in list_of_models:
        model_mean_W[_mode] = []
        temp = [result[_mode][0][i][0]['W'] for i in range(0,10)]
        temp_W = sum(temp)/len(temp)
        model_mean_W[_mode].append(temp_W)
    return model_mean_W

model_mean_W = get_mean_W(result, list_of_models)



checkX = myX[-1, :]
checkY = myY[-1, :]

list_of_colors = ['red', 'blue', 'green', 'cyan']
list_of_linestyles = ['dashed', 'dashdot', 'dotted', ':']
my_iterator = 0;
mpl.cla
mpl.grid(True)
for _mode in list_of_models:
    W = model_mean_W[_mode][0]
    if _mode != 'lin_reg':
        W = np.transpose(W)
    predY = np.dot(checkX, W)
    mpl.plot(predY, label = _mode, color = list_of_colors[my_iterator], hold=True, linestyle = list_of_linestyles[my_iterator], lw = 1.5)
    my_iterator += 1

mpl.plot(checkY, label = 'Real', linewidth = 1.75)
mpl.legend()
print('Done!')

print(std_dev_arr)
print(np.mean(std_dev_arr))




# Saving the objects:
with open('objs.pickle', 'w') as f:
    pickle.dump([result], f)

#Load back results:
with open('objs.pickle') as f:
    result = pickle.load(f)
