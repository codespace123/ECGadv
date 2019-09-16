import numpy as np
from keras.models import load_model
from cleverhans import utils
from random import randrange
from os import walk
import re
import scipy.io
from numpy import genfromtxt
import sys
import csv
def preprocess(x, maxlen):
    x = np.nan_to_num(x)
    x = x[0, 0:maxlen]
    x = x - np.mean(x)
    x = x / np.std(x)
    tmp = np.zeros((1, maxlen))
    tmp[0, :len(x)] = x.T  # padding sequence
    x = tmp
    x = np.expand_dims(x, axis=2)  # required by Keras
    del tmp
    return x

def filter(x):
    fs = 300

    #butterworth
    b, a = signal.butter(3, 0.05, btype='hp', fs = 300)
    bandpss_x = signal.lfilter(b, a, x)

    #notch filter
    f0 = 60
    b, a = signal.iirnotch(f0, 30, fs)
    y = signal.lfilter(b, a, bandpss_x)
    f0 = 50
    b, a = signal.iirnotch(f0, 30, fs)
    y = signal.lfilter(b, a, y)
    return y

def zero_mean(x):
    x = x - np.mean(x)
    x = x / np.std(x)
    return x

def op_concate(x,w,p):
    if w != 9000:
        x_tile = np.tile(x, (1, 1, 1))
        new_x = np.zeros((1,9000,1))
        new_x[0,p:p+w,0] = x_tile[0,:,0]
    else:
        x_tile = np.tile(x, (1, 1, 1))
        x1 = x_tile[:, 0:p, :]
        x2 = x_tile[:, p:9000, :]
        new_x = np.append(x2, x1, axis=1)
    return new_x

# parameters
dataDir = './training2017/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

print("Loading model")
model = load_model('./ResNet_30s_34lay_16conv.hdf5')


# loading data
select_data_id = int(sys.argv[1])
count = select_data_id - 1
record = "A{:05d}".format(select_data_id)

# select attack target
target = int(sys.argv[2])
# perturb window size
perturb_window = int(sys.argv[3])


print("Loading ground truth file")
csvfile = list(csv.reader(open('REFERENCE-v3.csv')))
ground_truth_label = csvfile[count][1]
ground_truth = classes.index(ground_truth_label)
ground_truth_a = utils.to_categorical(ground_truth, num_classes=4)
print("Record {} ground truth: {}".format(record, ground_truth_label))

# loading perturbation
filename = './output/' + str(ground_truth) + '/LDM_Attack_w' + str(perturb_window) + '_l2_A' + record + 'T' + str(target) + '.out'
perturb = genfromtxt(filename, delimiter=',')
perturb = filter(perturb)
perturb = np.expand_dims(perturb, axis=0)
perturb = np.expand_dims(perturb, axis=2)

# max possible shifting transformation
if perturb_window == 9000:
    maxpos = 9000
else:
    maxpos = 9000-perturb_window

# select target samples
if ground_truth == 0:
    target_file = np.genfromtxt('./data_select_A.csv', delimiter=',')
    target_id = target_file[:,3]
if ground_truth == 1:
    target_file = np.genfromtxt('./data_select_N.csv', delimiter=',')
    target_id = target_file[:,3]
if ground_truth == 2:
    target_file = np.genfromtxt('./data_select_O.csv', delimiter=',')
    target_id = target_file[:,3]
if ground_truth == 3:
    target_file = np.genfromtxt('./data_select_i.csv', delimiter=',')
    target_id = target_file[:,3]

target_len = target_file[:,2]

attack_success = np.zeros((4), dtype=int)
print("input file: ", filename)

#k = 0
for i, id_float in enumerate(target_id):
    # select sample length larger than 30s
    if int(target_len[i]) < 30:
        continue

    #if k >= 3:
    #   break
    #k = k + 1
    id_1 = int(id_float)
    count = id_1 - 1
    record_1 = "A{:05d}".format(id_1)

    # Loading victim sample
    local_filename = dataDir + record_1
    mat_data = scipy.io.loadmat(local_filename)
    data = mat_data['val']
    data = preprocess(data, WINDOW_SIZE)
    X_test_1 = np.float32(data)
    print("Victim sample: "+record_1)

    # Generate test data
    for p in range(100):
        # randomly select shifting position
        pos = randrange(0, maxpos)
        if p == 0:
           test_all = zero_mean(op_concate(perturb, perturb_window, pos) + X_test_1)
        else:
           test_all = np.append(test_all, zero_mean(op_concate(perturb, perturb_window, pos) + X_test_1), axis=0)

    # Predict
    prob = model.predict(test_all)
    ind = np.argmax(prob, axis=1)
    attack_success_current = np.zeros((4),dtype=int)
    for _, it in enumerate(ind):
        attack_success_current[it] = attack_success_current[it] + 1
        attack_success[it] = attack_success[it] + 1
    print("attack_success_current:",attack_success_current)
print("attack success:", attack_success)






