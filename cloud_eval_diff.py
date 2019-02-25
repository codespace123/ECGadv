#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Module import
import keras.backend as K
import keras
from keras import backend
from keras.models import load_model
import tensorflow as tf

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans import utils
import csv
import scipy.io
import glob
import numpy as np
import sys
from numpy import genfromtxt
import time

#### Funtion definition
def preprocess(x, maxlen):
    x =  np.nan_to_num(x)
#    x =  x[0, 0:min(maxlen,len(x))]
    x =  x[0, 0:maxlen]
    x = x - np.mean(x)
    x = x / np.std(x)
    
    tmp = np.zeros((1, maxlen))
#    print(x.shape)
    tmp[0, :len(x)] = x.T  # padding sequence
    x = tmp
#    print(x.shape)
    x = np.expand_dims(x, axis=2)  # required by Keras
#    print(x.shape)
    del tmp
    
    return x

def zero_mean(x):
    x = x - np.mean(x)
    x = x / np.std(x)
    return x

#### Main Program

#--- parameters
dataDir = './training2017/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

file = sys.argv[1]
fid_from = int(sys.argv[2])
fid_to = int(sys.argv[3])
data_select = genfromtxt(file, delimiter=',')

#--- loading model and prepare wrapper
keras.layers.core.K.set_learning_phase(0)
sess = tf.Session()
K.set_session(sess)
print("Loading model")    
model = load_model('ResNet_30s_34lay_16conv.hdf5')

#wrap = KerasModelWrapper(model, nb_classes=4)
wrap = KerasModelWrapper(model)

x = tf.placeholder(tf.float32, shape=(None, 9000, 1))
y = tf.placeholder(tf.float32, shape=(None, 4))
preds = model(x)

#--- load groundTruth File
print("Loading ground truth file")   
csvfile = list(csv.reader(open('REFERENCE-v3.csv')))
files = sorted(glob.glob(dataDir+"*.mat"))

#--- Attacker
from myattacks_diff import ECGadvDiff
ecgadvDiff = ECGadvDiff(wrap, sess=sess)
print('Attack diff is running...')

#--- loop on file including data_select[:,3] from fid_from-th row to fid_to-th row

eval_result = np.zeros((4*(fid_to-fid_from), 4)) # fid, ground_truth, target, adv_result

num = fid_from
while (num < fid_to):
    
    #--- Loading
    fid = int(data_select[num, 3]) 
    record = "A{:05d}".format(fid)
    local_filename = dataDir+record
    print('Loading record {}'.format(record))    
    mat_data = scipy.io.loadmat(local_filename)
    #data = mat_data['val'].squeeze()
    data = mat_data['val']
    print(data.shape)
    
    #--- Processing data
    data = preprocess(data, WINDOW_SIZE)
    X_test=np.float32(data)
    
    #--- Read the ground truth label, Change it to one-shot form
    ground_truth_label = csvfile[fid-1][1]
    ground_truth = classes.index(ground_truth_label)
    print('Ground truth:{}'.format(ground_truth))
    
    Y_test = np.zeros((1, 1))
    Y_test[0,0] = ground_truth
    Y_test = utils.to_categorical(Y_test, num_classes=4)
    
    #--- Prepare the target labels for targeted attack
    for i in range(4):
        if (i == ground_truth):
            continue
        
        target = np.zeros((1, 1))
        target[0,0] = i
        target = utils.to_categorical(target, num_classes=4)
        target = np.float32(target)
        
        #--- Attacking...
        ecgadvDiff_params = {'y_target': target}
        start_time = time.time() 
        adv_x = ecgadvDiff.generate(x, **ecgadvDiff_params)
        adv_x = tf.stop_gradient(adv_x) # Consider the attack to be constant
        feed_dict = {x: X_test}
        adv_sample = adv_x.eval(feed_dict=feed_dict, session=sess)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        #--- Attack result
        adv_sample = zero_mean(adv_sample)
        prob = model.predict(adv_sample)
        ann = np.argmax(prob)
#        ann_label = classes[ann]
        print('Adv result:{}'.format(ann))
        
        idx = num - fid_from
        eval_result[4*idx+i, 0] = fid
        eval_result[4*idx+i, 1] = ground_truth
        eval_result[4*idx+i, 2] = i
        eval_result[4*idx+i, 3] = ann
        
        #--- Save adv_sample to file
        file_sample = './cloud_model/smooth_eval/R' + str(fid)+ '_' + str(ground_truth) + '_' + str(i) + '_' + str(ann) + '.csv'
        np.savetxt(file_sample, adv_sample[0,:], delimiter=",")
        
    num = num+1
        
file_result = './cloud_model/smooth_eval/res'+ '_' + str(fid_from) + '_' + str(fid_to) + '.csv'
np.savetxt(file_result, eval_result, delimiter=",")  
        
    
    
