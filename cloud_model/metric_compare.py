#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:15:44 2018

@author: chenhx1992
"""

from keras.utils import plot_model
import keras.backend as K
import keras
from keras import backend
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras import metrics
import tensorflow as tf
import pydot
import h5py
from numpy import genfromtxt
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
import scipy.fftpack
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import os
import glob
import csv

dataDir = "../training2017/"
folder1 = "./smooth_eval/"
folder2 = "./l2_eval/"
folder3 = "./l2smooth_0_01_eval/"
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

#### Funtion definition
def preprocess(x, maxlen):
    x =  np.nan_to_num(x)
#    x =  x[0, 0:min(maxlen,len(x))]
    x =  x[0, 0:maxlen]
    x = x - np.mean(x)
    x = x / np.std(x)
    
    tmp = np.zeros((1, maxlen))
    tmp[0, :len(x)] = x.T  # padding sequence
    x = tmp
    x = np.expand_dims(x, axis=2)  # required by Keras
    del tmp
    
    return x


idx = 3863
TRUTH = '0'
g_label = classes[int(TRUTH)]
TARGET = '1'
t_label = classes[int(TARGET)]

record = "A{:05d}".format(idx)
local_filename = dataDir+record
print('Loading record {}'.format(record))    
mat_data = scipy.io.loadmat(local_filename)
data = mat_data['val']
data = preprocess(data, WINDOW_SIZE)  
sample = np.reshape(data, (9000,1))


file1 = glob.glob(folder1 + "R" +  str(idx) + "_" + TRUTH + "_" + TARGET + "_?.csv")[0]
adv_sample_1 = genfromtxt(file1, delimiter=',')
adv_sample_1 = np.reshape(adv_sample_1, (9000,1))
res_1 = file1[-5]
r_label_1 = classes[int(res_1)]

file2 = glob.glob(folder2 + "R" +  str(idx) + "_" + TRUTH + "_" + TARGET + "_?.csv")[0]
adv_sample_2 = genfromtxt(file2, delimiter=',')
adv_sample_2 = np.reshape(adv_sample_2, (9000,1))
res_2 = file2[-5]
r_label_2 = classes[int(res_2)]

file3 = glob.glob(folder3 + "R" +  str(idx) + "_" + TRUTH + "_" + TARGET + "_?.csv")[0]
adv_sample_3 = genfromtxt(file3, delimiter=',')
adv_sample_3 = np.reshape(adv_sample_3, (9000,1))
res_3 = file3[-5]
r_label_3 = classes[int(res_3)]


ymax = np.max([sample, adv_sample_1, adv_sample_2, adv_sample_3])+1
ymin = np.min([sample, adv_sample_1, adv_sample_2, adv_sample_3])-1


fig, axs = plt.subplots(4, 1, figsize=(160,40), sharex=True)

axs[0].plot(sample[0:4000,:], color='black', label='Original signal')
axs[0].set_title('Original ECG signal, Class {}'.format(g_label))
axs[0].set_ylim([ymin, ymax])
axs[0].set_ylabel('Amplitude')

axs[1].plot(adv_sample_1[0:4000,:], color='forestgreen', label='Adv signal_diff')
axs[1].set_title('Adversarial ECG signal, Class {}, Metric: Smoothness'.format(r_label_1))
axs[1].set_ylim([ymin, ymax])
axs[1].set_ylabel('Amplitude')

axs[2].plot(adv_sample_2[0:4000,:], color='forestgreen', label='Adv signal_l2')
axs[2].set_title('Adversarial ECG signal, Class {}, Metric: L2-norm'.format(r_label_1))
axs[2].set_ylim([ymin, ymax])
axs[2].set_ylabel('Amplitude')

axs[3].plot(adv_sample_3[0:4000,:], color='forestgreen', label='Adv signal_l2_diff')
axs[3].set_title('Adversarial ECG signal, Class {}, Metric: Smoothness+L2-norm'.format(r_label_1))
axs[3].set_ylim([ymin, ymax])
axs[3].set_xlabel('Sample Index')
axs[3].set_ylabel('Amplitude')
#fig.tight_layout()
     
