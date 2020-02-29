from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from collections import OrderedDict as odict
from timeit import default_timer as timer
import ipdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.linalg as LA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tqdm import tqdm, trange
import math

import keras
from keras import backend as K
from keras import optimizers, regularizers, losses
from keras.layers import Dense, Input, Activation, Conv1D, Dropout, GlobalAveragePooling1D
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.engine.topology import Layer

# import complexnn
from .complexnn import ComplexDense, ComplexConv1D, utils
from .cxnn.train_network  import set_keras_backend

from .models_adsb import Modrelu
from ..preproc.fading_model import normalize, add_custom_fading_channel, add_freq_offset


# mpl.rc('text', usetex=True)
# mpl.rc('text.latex', preamble = r'\usepackage{amsmath}, \usepackage{sfmath}, \usepackage{amssymb}, \usepackage{bm}, \DeclareMathOperator*{\supp}{supp}, \DeclareMathOperator*{\proj}{\mathcal{P}_K}, \newcommand{\be}{{\bm e}}, \newcommand{\bx}{{\bm x}}, \DeclareMathOperator*{\support}{\mathcal{S}_K}')



set_keras_backend("theano")
np.set_printoptions(precision=2)

exp_dir = '/home/rfml/wifi/experiments/exp19'

preprocess_type = 1
sample_rate = 200
sample_duration = 16

# Offset for testing
df_test = 20e-6 # 20 ppm

df_train = 20e-6 
rand_train = 'unif'
# rand_train = 'ber'
# rand_train = 'False'

# aug_train_df = 0
aug_train_df = 5
keep_orig = False
# keep_orig = True

'''
channel type:
	1 - Extended Pedestrian A
	2 - Extended Vehicular A
	3 - Extended Typical Urban
'''
channel_type_train = 3
channel_type_test = 3

seed_train = 0
seed_test = 0
# seed_test = 0

channel_type_aug_train = 1
channel_type_aug_test = 1

channel_method = 'FFT' 
# channel_method = 'RC' # Raised-cosine

noise_method = 'reg' # Regular
# noise_method = 'bl' # Bandlimited via root raised-cosine

# Random seed for delay perturbation
# Set to False for fixed delays
delay_seed = False
# delay_seed = None

'''
aug_type:
	0 - usual channel aug
	1 - same channel for ith example in each class
'''
aug_type = 0

aug_train_ch = 0
keep_orig = False
# keep_orig = True

num_ch_train = -1
# num_ch_test = -1
num_ch_test = 0

snr_train = 500
snr_test = 500



data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

start = timer()

npz_filename = exp_dir + '/sym-' + data_format + '.npz'

np_dict = np.load(npz_filename)
dict_wifi = {}
dict_wifi['x_train'] = np_dict['arr_0']
dict_wifi['y_train'] = np_dict['arr_1']
dict_wifi['x_test'] = np_dict['arr_2']
dict_wifi['y_test'] = np_dict['arr_3']
dict_wifi['fc_train'] = np_dict['arr_4']
dict_wifi['fc_test'] = np_dict['arr_5']
dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]

end = timer()
print('Load time: {:} sec'.format(end - start))

# data_format = 'offset-{}-{}-'.format(aug_train, rand_train)
# data_format += '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

# Checkpoint path
# checkpoint_in = exp_dir + '/ckpt-' + data_format + '.h5'

x_train = dict_wifi['x_train']
y_train = dict_wifi['y_train']
x_test = dict_wifi['x_test']
y_test = dict_wifi['y_test']
fc_train = dict_wifi['fc_train']
fc_test = dict_wifi['fc_test']
num_classes = dict_wifi['num_classes']
num_train = x_train.shape[0]
num_test = x_test.shape[0]
num_features = x_train.shape[1]
fs = sample_rate * 1e+6


#--------------------------------------------------------------------------------------------
# Different day scenario simulation
#--------------------------------------------------------------------------------------------
sampling_rate = fs
signal_ch = dict_wifi['x_train'].copy()
for i in tqdm(range(num_train)):
	signal = dict_wifi['x_train'][i][:,0]+1j*dict_wifi['x_train'][i][:,1]
	signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
														seed=seed_train, 
														beta=0, 
														delay_seed=False, 
														channel_type=channel_type_train,
														channel_method=channel_method,
														noise_method=noise_method)
	signal_faded = normalize(signal_faded)
	signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
dict_wifi['x_train'] = signal_ch.copy()

signal_ch = dict_wifi['x_test'].copy()
for i in tqdm(range(num_test)):
	signal = dict_wifi['x_test'][i][:,0]+1j*dict_wifi['x_test'][i][:,1]
	signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
														seed=seed_test, 
														beta=0, 
														delay_seed=False,
														channel_type=channel_type_test,
														channel_method=channel_method,
														noise_method=noise_method)
	signal_faded = normalize(signal_faded)
	signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
dict_wifi['x_test'] = signal_ch.copy()


data_format = 'aug-{}-ty-{}-nch-{}-{}-snr-{:.0f}-{:.0f}-'.format(aug_train_ch, aug_type, num_ch_train, num_ch_test, snr_train, snr_test) 
if aug_train_df > 0:
	data_format = 'offset-{}-{}-'.format(aug_train_df, rand_train) + data_format
data_format += '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)


# Checkpoint path
checkpoint_in = exp_dir + '/ckpt-' + data_format +'.h5'

x_train = dict_wifi['x_train']
x_test = dict_wifi['x_test']

print('\nFc:')
print('Train: {}'.format(np.unique(fc_train)))
print('Test: {}\n'.format(np.unique(fc_test)))

batch_size = 100	
epochs = 100
weight_decay = 1e-3

# print("========================================") 
# print("MODEL HYPER-PARAMETERS") 
# print("BATCH SIZE: {:3d}".format(batch_size)) 
# print("WEIGHT DECAY: {:.4f}".format(weight_decay))
# print("EPOCHS: {:3d}".format(epochs))
print("========================================") 
print("== BUILDING MODEL... ==")

if checkpoint_in is None:
	raise ValueError('Cannot test without a checkpoint')
	# data_input = Input(batch_shape=(batch_size, num_features, 2))
	# output, model_name = network_20_2(data_input, num_classes, weight_decay)
	# densenet = Model(data_input, output)

checkpoint_in += '-modrelu-100-100-new--new.h5'
densenet = load_model(checkpoint_in, 
					  custom_objects={'ComplexConv1D':complexnn.ComplexConv1D,
					  				  'GetAbs': utils.GetAbs,
					  				  'Modrelu': Modrelu})

densenet.summary()
# for layer in densenet.layers:
# 	print(layer.name)
# densenet = ...  # create the original model


probs = densenet.predict(x=x_test, batch_size=batch_size, verbose=0)
label_pred = probs.argmax(axis=1) 
label_act = y_test.argmax(axis=1) 
ind_correct = np.where(label_pred==label_act)[0] 
ind_wrong = np.where(label_pred!=label_act)[0] 
assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
test_acc = 100.*ind_correct.size / num_test

acc_class = np.zeros([num_classes])
for class_idx in range(num_classes):
	idx_inclass = np.where(label_act==class_idx)[0]
	ind_correct = np.where(label_pred[idx_inclass]==label_act[idx_inclass])[0] 
	acc_class[class_idx] = 100*ind_correct.size / idx_inclass.size

print("\n========================================") 
print('Test accuracy: {:.2f}%'.format(test_acc))

rand_test = 'ber'
x_test_offset = add_freq_offset(x_test, rand = rand_test,
									   	df = df_test,
									   	fc = fc_test, 
									   	fs = fs)

probs = densenet.predict(x=x_test_offset, batch_size=batch_size, verbose=0)
label_pred = probs.argmax(axis=1) 
label_act = y_test.argmax(axis=1) 
ind_correct = np.where(label_pred==label_act)[0] 
ind_wrong = np.where(label_pred!=label_act)[0] 
assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
test_acc = 100.*ind_correct.size / num_test

print("\n========================================") 
print('Test accuracy with {} freq offset: {:.2f}%'.format(rand_test ,test_acc))


rand_test = 'unif'
x_test_offset = add_freq_offset(x_test, rand = rand_test,
									   	df = df_test,
									   	fc = fc_test, 
									   	fs = fs)

probs = densenet.predict(x=x_test_offset, batch_size=batch_size, verbose=0)
label_pred = probs.argmax(axis=1) 
label_act = y_test.argmax(axis=1) 
ind_correct = np.where(label_pred==label_act)[0] 
ind_wrong = np.where(label_pred!=label_act)[0] 
assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
test_acc = 100.*ind_correct.size / num_test

print("\n========================================") 
print('Test accuracy with {} freq offset: {:.2f}%'.format(rand_test ,test_acc))

rand_test = False
x_test_offset = add_freq_offset(x_test, rand = rand_test,
									   	df = df_test,
									   	fc = fc_test, 
									   	fs = fs)

probs = densenet.predict(x=x_test_offset, batch_size=batch_size, verbose=0)
label_pred = probs.argmax(axis=1) 
label_act = y_test.argmax(axis=1) 
ind_correct = np.where(label_pred==label_act)[0] 
ind_wrong = np.where(label_pred!=label_act)[0] 
assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
test_acc = 100.*ind_correct.size / num_test

print("\n========================================") 
print('Test accuracy with {} freq offset: {:.2f}%'.format(rand_test ,test_acc))


conf_matrix_test = metrics.confusion_matrix(label_act, label_pred)
conf_matrix_test = 100*conf_matrix_test/(num_test/num_classes)
# print('{}'.format(conf_matrix_test))

output_dict = odict(acc=odict(), comp=odict(), loss=odict())

output_dict['acc']['test'] = test_acc
# output_dict['acc']['val'] = 100.*history.history['val_acc'][-1]
# output_dict['acc']['train'] = 100.*history.history['acc'][-1]

# output_dict['loss']['val'] = history.history['val_loss'][-1]
# output_dict['loss']['train'] = history.history['loss'][-1]

stringlist = []
densenet.summary(print_fn=lambda x: stringlist.append(x))
summary = '\n' + \
		'Batch size: {:3d}\n'.format(batch_size) + \
		'Weight decay: {:.4f}\n'.format(weight_decay) + \
		'Epochs: {:3d}\n'.format(epochs) + \
		'Optimizer:' + str(densenet.optimizer) + '\n'
summary += '\n'.join(stringlist)