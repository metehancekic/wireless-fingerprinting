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
from tqdm import trange
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



exp_dirs = []
# exp_dirs += ['/home/rfml/wifi/experiments/exp19']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2']
exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3E']



preprocess_type = 1
sample_rate = 200
sample_duration = 16

# Offset for testing
df_test = 40e-6 # 20 ppm


# -------------------------------------------------------------------- #
# CHANNEL

# Augmetation method for channel augmentation
# CH_AUG_TYPE = 0  # simple channel augmentation
CH_AUG_TYPE = 1 # channels are different inside a class, but same for i'th packet in each class

# CHANNEL AUGMENTATION
NUM_CH_AUG_TRAIN = 20 # Number of channel augmentations that will be done on train set
NUM_CH_AUG_TEST = 5 # Number of channel augmentations that will be done on test set

# Number of channel filters per augmentation for train data used (-1 corresponds to use different channel for every packet)
NUM_CH_PER_AUG_TRAIN = -1
# Number of channel filters per augmentation for test data used (-1 corresponds to use different channel for every packet)
NUM_CH_PER_AUG_TEST = -1

'''
channel type:
	1 - Extended Pedestrian A (410 ns, 7 taps)
	2 - Extended Vehicular A (2510 ns, 9 taps)
	3 - Extended Typical Urban (5000 ns, 9 taps)
'''
CHANNEL_TYPE_TRAIN = 2
CHANNEL_TYPE_TEST = 1

CHANNEL_METHOD = 'FFT' 
# CHANNEL_METHOD = 'RC' # Raised-cosine

NOISE_METHOD = 'reg' # Regular
# NOISE_METHOD = 'bl' # Bandlimited via root raised-cosine

# Random seed for delay perturbation
# Set to False for fixed delays
# DELAY_SEED = None
DELAY_SEED = False
# -------------------------------------------------------------------- #



# -------------------------------------------------------------------- #
# CARRIER FREQUENCY OFFSET
# Number of CFO augmentations will be done on train set
NUM_CFO_AUG_TRAIN = 5 # aug_train = 5
NUM_CFO_AUG_TEST = 1 # aug_train = 5


df_train = 40e-6 
# rand_train = 'unif'
rand_train = 'ber'
# rand_train = False

# -------------------------------------------------------------------- #



# -------------------------------------------------------------------- #
# KEEP ORIGINAL DATA
# Whether you want to keep original set or not
KEEP_ORIG_TRAIN = False
KEEP_ORIG_TEST = False
# -------------------------------------------------------------------- #



# -------------------------------------------------------------------- #
# IN CASE OF NOISE INJECTION INSIDE CHANNEL AUGMENTATION
# SNR values for noise injection in channel augmentation 500>= corresponds to no noise injection
snr_train = 500
snr_test = 500
# -------------------------------------------------------------------- #



data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

start = timer()

npz_filename = exp_dirs[0] + '/sym-' + data_format + '.npz'

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

data_format = 'aug-{}-ty-{}-nch-{}-{}-snr-{:.0f}-{:.0f}-'.format(NUM_CH_AUG_TRAIN, CH_AUG_TYPE, NUM_CH_PER_AUG_TRAIN, NUM_CH_PER_AUG_TEST, snr_train, snr_test)
data_format += 'offset-{}-{}-ko-{}-'.format(NUM_CFO_AUG_TRAIN, rand_train, KEEP_ORIG_TRAIN)
data_format += '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

# Checkpoint path
checkpoint_in = exp_dirs[0] + '/ckpt-' + data_format +'.h5'


print('\n Channel Augmentation')
print('\t Channel type: Train: {}, Test: {} (EPA, EVA, ETU)'.format( CHANNEL_TYPE_TRAIN, CHANNEL_TYPE_TEST ))
print('\t Channel augmentation style (0: simple augmentation) {}'.format( CH_AUG_TYPE ))
print('\t Number of Augmentations (training): {}, (testing): {} '.format(NUM_CH_AUG_TRAIN, NUM_CH_AUG_TEST))
print('\t Number of channels per augmentation (training): {}, (testing): {} '.format(NUM_CH_PER_AUG_TRAIN, NUM_CH_PER_AUG_TEST))
print('\t Channel method : {}, noise method : {} '.format(CHANNEL_METHOD, NOISE_METHOD))
print('\t Delay seed for taps: {} \n'.format(DELAY_SEED))

print('Carrier Frequency Augmentation')
print('\t Randomness of Train CFO: {} (uniform, bernouili, False: (fixed) )'.format( rand_train))
print('\t PPM train : {}, PPM test : {}'.format( df_train, df_test ))
print('\t Number of Augmentations (training): {}, (testing): {} \n'.format(NUM_CFO_AUG_TRAIN, NUM_CFO_AUG_TEST))

print('Keep Original Dataset and Noise Addition')
print('\t Keep Original Data (Train) : {}, (Test) : {}'.format( KEEP_ORIG_TRAIN, KEEP_ORIG_TEST ))
# print('\t SNR values for (training): {}, (testing): {} \n'.format(snr_trains[0], snr_tests[0]))



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

checkpoint_in += '-modrelu-100-100-new-'+'-new.h5'
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