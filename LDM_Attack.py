import keras.backend as K
import keras
from keras.models import load_model
import tensorflow as tf
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans import utils

import csv
import scipy.io
import numpy as np
import sys
from LDM_EOT import LDM_EOT_ATTACK
import math


# parameters
dataDir = './training2017/'
FS = 300
WINDOW_SIZE = 30 * FS  # padding window for CNN
classes = ['A', 'N', 'O', '~']

keras.layers.core.K.set_learning_phase(0)

sess = tf.Session()
K.set_session(sess)

print("Loading model")
model = load_model('./ResNet_30s_34lay_16conv.hdf5')

wrap = KerasModelWrapper(model)
#wrap = KerasModelWrapper(model, nb_classes=4)

x = tf.placeholder(tf.float32, shape=(None, 9000, 1))
y = tf.placeholder(tf.float32, shape=(None, 4))


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


def zero_mean(x):
    x = x - np.mean(x)
    x = x / np.std(x)
    return x


def op_concate(x, w, i):
    data_len = 9000
    tile_times = math.ceil(data_len / w)
    x_tile = np.tile(x, (1, tile_times, 1))
    p = i
    x1 = x_tile[:, 0:p, :]
    x2 = x_tile[:, p:data_len, :]

    return np.append(x2, x1, axis=1)


# input
# select data
select_data_id = int(sys.argv[1])
# attack target
target = np.zeros((1, 1))
target[0, 0] = int(sys.argv[2])
target_a = utils.to_categorical(target, num_classes=4)
# perturb window size
perturb_window = int(sys.argv[3])

# l2 distance
dis_metric = 1

# load groundTruth
count = select_data_id - 1
record = "A{:05d}".format(select_data_id)
local_filename = dataDir + record
print("Loading ground truth file")
csvfile = list(csv.reader(open('REFERENCE-v3.csv')))
ground_truth_label = csvfile[count][1]
ground_truth = classes.index(ground_truth_label)
ground_truth_a = utils.to_categorical(ground_truth, num_classes=4)
print("Record {} ground truth: {}".format(record, ground_truth_label))


# Loading data
mat_data = scipy.io.loadmat(local_filename)
print('Loading record {}'.format(record))
data = mat_data['val']

# Preprocess
data = preprocess(data, WINDOW_SIZE)
X_test = np.float32(data)

#adjust the ensemble_size for a suitable learning time.
if perturb_window != 9000:
    ensemble_size = int((9000 - perturb_window) / 50)
else:
    ensemble_size = 9000 / 50

# Attack
LDM_EOT = LDM_EOT_ATTACK(wrap, sess=sess)
LDM_EOT_params = {'y_target': target_a, 'learning_rate': 1, 'max_iterations': 500, 'initial_const': 50000,
                'perturb_window': perturb_window, 'dis_metric': dis_metric, 'ensemble_size': ensemble_size,
                'ground_truth': ground_truth_a}
adv_x = LDM_EOT.generate(x, **LDM_EOT_params)
adv_x = tf.stop_gradient(adv_x)  # Consider the attack to be constant
feed_dict = {x: X_test}
adv_sample = adv_x.eval(feed_dict=feed_dict, session=sess)

# perturbation
perturb = adv_sample - X_test
perturb = perturb[:, 0:perturb_window, :]
perturb_squeeze = np.squeeze(perturb, axis=2)

# save perturbation
outputstr = './output/' + str(ground_truth) + '/LDM_Attack_w' + str(perturb_window) + '_l2_' + record + 'T' + str(int(target[0, 0])) + '.out'
np.savetxt(outputstr, perturb_squeeze, delimiter=",")

