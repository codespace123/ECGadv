#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras import metrics

import tensorflow as tf

# Parameters
dataDir = './training2017/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

## funtion 
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

def predict_data(model, x):
    prob = model.predict(x)
    ann = np.argmax(prob)
    return prob, ann
    
## Loading time serie signals
files = sorted(glob.glob(dataDir+"*.mat"))

# Load and apply model
print("Loading model")    
model = load_model('ResNet_30s_34lay_16conv.hdf5')

# load groundTruth
print("Loading ground truth file")   
csvfile = list(csv.reader(open('REFERENCE-v3.csv')))

# Main loop 
prediction = np.zeros((len(files),5))
count = 0
correct = 0

for f in files:
    record = f[:-4]
    record = record[-6:]
    # Loading
    mat_data = scipy.io.loadmat(f[:-4] + ".mat")
    print('Loading record {}'.format(record))    
#    data = mat_data['val'].squeeze()
    data = mat_data['val']
    
    x = preprocess(data, WINDOW_SIZE)
    
    print("Applying model ..") 
    ground_truth_label = csvfile[count][1]
    ground_truth = classes.index(ground_truth_label)
    prob_x, ann_x = predict_data(model, x)
    
    print("Record {} ground truth: {}".format(record, ground_truth_label))
    print("Record {} classified as {} with {:3.1f}% certainty".format(record, classes[ann_x], 100*prob_x[0,ann_x]))
    
    prediction[count,0] = ground_truth
    prediction[count,1] = ann_x
    prediction[count,2] = prob_x[0,ann_x]
    prediction[count,3] = len(data[0,:])/300.0
    prediction[count,4] = count+1
    
    if (ground_truth == ann_x):
        correct += 1
        
    count += 1


print("Correct:{}, total:{}, percent:{}".format(correct, count, correct/(count)))

#Select correct prediction
cond_x_gt = np.equal(prediction[:,1], prediction[:,0]) 
correct_prediction = prediction[cond_x_gt]

# save prediction to csv files
format = '%i,%i,%.5f,%.2f,%i'
np.savetxt("prediction_correct.csv", correct_prediction, fmt= format, delimiter=",")


type_A = correct_prediction[(correct_prediction[:,0] == 0)]
type_A_select = type_A[:,[0,2,3,4]]

type_N = correct_prediction[(correct_prediction[:,0] == 1)]
type_N_select = type_N[:,[0,2,3,4]]

type_O = correct_prediction[(correct_prediction[:,0] == 2)]
type_O_select = type_O[:,[0,2,3,4]]

type_i = correct_prediction[(correct_prediction[:,0] == 3)]
type_i_select = type_i[:,[0,2,3,4]]

np.savetxt('data_select_A.csv', type_A_select, delimiter=",")
np.savetxt('data_select_N.csv', type_N_select, delimiter=",")
np.savetxt('data_select_O.csv', type_O_select, delimiter=",")
np.savetxt('data_select_i.csv', type_i_select, delimiter=",")
