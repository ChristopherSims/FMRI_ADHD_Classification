import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import csv

mpl.rcParams["mathtext.fontset"] = 'dejavusans'
font = {
        'size'   : 22}

mpl.rc('font', **font)


# data = np.genfromtxt('adhd-fmri-history_cv1_15-06-34_03302022.csv',
#                      skip_header=1,
#                      skip_footer=0,
#                      dtype=float,
#                      delimiter=',')

# data2 = np.genfromtxt('adhd-fmri-history_cv1_14-22-06_GRU.csv',
#                      skip_header=1,
#                      skip_footer=0,
#                      dtype=float,
#                      delimiter=',')

data = np.genfromtxt('adhd-fmri-history_cv1_Multi_LSTM.csv',
                     skip_header=1,
                     skip_footer=0,
                     dtype=float,
                     delimiter=',')

data2 = np.genfromtxt('adhd-fmri-history_cv1_Mutli_GRU.csv',
                     skip_header=1,
                     skip_footer=0,
                     dtype=float,
                     delimiter=',')


## 0 : Epoch
## 1 : Training accuracy
## 2 : Training loss
## 3 : Validation accuracy
## 4 : Validation loss

plt.figure(1)

params = {'linewidth': 4, 'color': 'black'}
params2 = {'linewidth': 4, 'color': 'red'}
plt.plot(data[:,1]*100, **params, linestyle = '-.', label = "LSTM Training")
plt.plot(data[:,3]*100, **params, label = "LSTM Val")
plt.plot(data2[:,1]*100, **params2, linestyle = '-.', label = "GRU Training")
plt.plot(data2[:,3]*100, **params2,label = "GRU Val")
#plt.ylim([0.5,1])
plt.ylim([50,100])
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.show()