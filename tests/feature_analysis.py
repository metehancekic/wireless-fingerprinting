'''
Trains data for a WiFi experiment.

Data is read from npz files.
'''

import numpy as np
from timeit import default_timer as timer
import argparse
from tqdm import trange, tqdm
import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict as odict
import copy

import keras
from keras.models import Model, load_model
from keras import backend as K
from keras.utils import plot_model
from keras import optimizers, regularizers, losses
from keras.layers import Dense, Input, Activation, Conv1D, Dropout, GlobalAveragePooling1D, Lambda, Average
from keras.regularizers import l2

from ..cxnn.models_globecom import network_20_modrelu_short, network_20_reim, network_20_reim_2x, network_20_reim_sqrt2x, network_20_mag, network_200_modrelu_short, network_200_reim, network_200_reim_2x, network_200_reim_sqrt2x, network_200_mag, network_200_modrelu_short_shared

from ..cxnn.complexnn import ComplexDense, ComplexConv1D, utils

from ..cxnn.models_adsb import Modrelu

from ..cxnn.train_globecom import train_20, train_200

# from freq_offset import estimate_freq_offset -- TO BE REMOVED!!! ---

from ..preproc.preproc_wifi import basic_equalize_preamble, offset_compensate_preamble
from ..preproc.fading_model import normalize, add_custom_fading_channel, add_freq_offset
from ..simulators import augment_with_channel_test

with open('/home/rfml/wifi/scripts/config_cfo_channel.json') as config_file:
    config = json.load(config_file, encoding='utf-8')

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--arch", type=str, choices=['reim', 
													   'reim2x', 
													   'reimsqrt2x',
													   'magnitude',
													   'phase',
													   're',
													   'im',
													   'modrelu',
													   'crelu'],
													   default= 'modrelu', 
					help="Architecture") 
architecture = parser.parse_args().arch


#-------------------------------------------------
# Data configuration
#-------------------------------------------------

exp_dir = config['exp_dir']
sample_duration = config['sample_duration']
preprocess_type = config['preprocess_type']
sample_rate = config['sample_rate']

#-------------------------------------------------
# Training configuration
#-------------------------------------------------
epochs = config['epochs']

#-------------------------------------------------
# Physical Channel Parameters
#-------------------------------------------------
add_channel = True

phy_method = config['phy_method']
seed_phy_train = config['seed_phy_train']
seed_phy_test = config['seed_phy_test']
channel_type_phy_train = config['channel_type_phy_train']
channel_type_phy_test = config['channel_type_phy_test']
phy_noise = config['phy_noise']
snr_train_phy = config['snr_train_phy']
snr_test_phy = config['snr_test_phy']

#-------------------------------------------------
# Physical CFO parameters
#-------------------------------------------------

add_cfo = True
remove_cfo = False

phy_method_cfo = config["phy_method_cfo"]
df_phy_train = config['df_phy_train']
df_phy_test = config['df_phy_test']
seed_phy_train_cfo = config['seed_phy_train_cfo']
seed_phy_test_cfo = config['seed_phy_test_cfo']

#-------------------------------------------------
# Equalization params
#-------------------------------------------------
equalize_train = False
equalize_test = False
verbose_train = False
verbose_test = False

#-------------------------------------------------
# Augmentation channel parameters
#-------------------------------------------------
augment_channel = True

channel_type_aug_train = config['channel_type_aug_train']
channel_type_aug_test = config['channel_type_aug_test']
num_aug_train = config['num_aug_train']
num_aug_test = config['num_aug_test']
aug_type = config['aug_type']
num_ch_train = config['num_ch_train']
num_ch_test =  config['num_ch_test']
channel_method = config['channel_method']
noise_method = config['noise_method']
delay_seed_aug_train = False
delay_seed_aug_test = False
keep_orig_train = False
keep_orig_test = False
snr_train = config['snr_train']
snr_test = config['snr_test']
beta = config['beta']

#-------------------------------------------------
# Augmentation CFO parameters
#-------------------------------------------------
augment_cfo = True

df_aug_train = df_phy_train 
rand_aug_train = config['rand_aug_train']
num_aug_train_cfo = config['num_aug_train_cfo']
keep_orig_train_cfo = config['keep_orig_train_cfo']
aug_type_cfo = config['aug_type_cfo']

keep_orig_test_cfo = config['keep_orig_test_cfo']
num_aug_test_cfo = config['num_aug_test_cfo']
rand_aug_test = config['rand_aug_test']
df_aug_test = config['df_aug_test']

#-------------------------------------------------
# Loading Data
#-------------------------------------------------

data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)
outfile = exp_dir + '/sym-' + data_format + '.npz'

np_dict = np.load(outfile)
dict_wifi = {}
dict_wifi['x_train'] = np_dict['arr_0']
dict_wifi['y_train'] = np_dict['arr_1']
dict_wifi['x_test'] = np_dict['arr_2']
dict_wifi['y_test'] = np_dict['arr_3']
dict_wifi['fc_train'] = np_dict['arr_4']
dict_wifi['fc_test'] = np_dict['arr_5']
dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]

num_train = dict_wifi['x_train'].shape[0]
num_test = dict_wifi['x_test'].shape[0]

data_format += '_{}'.format(architecture)

num_train = dict_wifi['x_train'].shape[0]
num_test = dict_wifi['x_test'].shape[0]
num_classes = dict_wifi['y_train'].shape[1]

sampling_rate = sample_rate * 1e+6
fs = sample_rate * 1e+6

x_train_orig = dict_wifi['x_train'].copy()
y_train_orig = dict_wifi['y_train'].copy()

x_test_orig = dict_wifi['x_test'].copy()
y_test_orig = dict_wifi['y_test'].copy()

if add_channel:
	# data_format = data_format + '-phy-{}-m-{}'.format(channel_type_phy_train, phy_method)
	data_format = data_format + '-phy-{}-m-{}-s-{}'.format(channel_type_phy_train, phy_method, np.max(seed_phy_train))		
if add_cfo:
	data_format = data_format + '_cfo_{}'.format(np.int(df_phy_train*1000000))
if remove_cfo:
	data_format = data_format + '_comp'
if equalize_train or equalize_test:
	data_format = data_format + '-eq'
if augment_channel:
	data_format = data_format + 'aug-{}-art-{}-ty-{}-nch-{}-snr-{:.0f}'.format(num_aug_train, channel_type_aug_train, aug_type, num_ch_train, snr_train)
if augment_cfo:
	data_format = data_format + 'augcfo-{}-df-{}-rand-{}-ty-{}-{}-t-'.format(num_aug_train, df_aug_train*1e6, rand_aug_train, aug_type, keep_orig_train)

checkpoint_in = str(exp_dir + '/ckpt-' + data_format)


#--------------------------------------------------------------------------------------------
# Physical channel simulation (different days)
#--------------------------------------------------------------------------------------------
if add_channel:
	print('\nPhysical channel simulation (different days)')
	print('\tMethod: {}'.format(phy_method))
	print('\tChannel type: Train: {}, Test: {}'.format(channel_type_phy_train, channel_type_phy_test))
	print('\tSeed: Train: {}, Test: {}\n'.format(seed_phy_train, seed_phy_test))

	if phy_method == 0: # Same channel for all packets

		signal_ch = dict_wifi['x_test'].copy()
		for i in tqdm(range(num_test)):
			signal = dict_wifi['x_test'][i][:,0]+1j*dict_wifi['x_test'][i][:,1]
			signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
																seed=seed_phy_test, 
																beta=0, 
																delay_seed=False,
																channel_type=channel_type_phy_test,
																channel_method=channel_method,
																noise_method=noise_method)
			signal_faded = normalize(signal_faded)
			signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
		dict_wifi['x_test'] = signal_ch.copy()


	elif phy_method==1: # Different channel for each class, same channel for all packets in a class

		signal_ch = dict_wifi['x_test'].copy()
		for n in trange(num_classes):
			ind_n = np.where(y_test_orig.argmax(axis=1)==n)[0]
			seed_phy_test_n = seed_phy_test + n
			for i in ind_n:
				signal = dict_wifi['x_test'][i][:,0]+1j*dict_wifi['x_test'][i][:,1]
				signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
																	seed=seed_phy_test_n, 
																	beta=0, 
																	delay_seed=False,
																	channel_type=channel_type_phy_test,
																	channel_method=channel_method,
																	noise_method=noise_method)
				signal_faded = normalize(signal_faded)
				signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
		dict_wifi['x_test'] = signal_ch.copy()



	else: # train on multiple days, test on 1 day, with a diff channel for each class

		signal_ch = dict_wifi['x_test'].copy()
		for n in trange(num_classes):
			ind_n = np.where(y_test_orig.argmax(axis=1)==n)[0]
			seed_phy_test_n = seed_phy_test + n
			for i in ind_n:
				signal = dict_wifi['x_test'][i][:,0]+1j*dict_wifi['x_test'][i][:,1]
				signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
																	seed=seed_phy_test_n, 
																	beta=0, 
																	delay_seed=False,
																	channel_type=channel_type_phy_test,
																	channel_method=channel_method,
																	noise_method=noise_method)
				signal_faded = normalize(signal_faded)
				signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
		dict_wifi['x_test'] = signal_ch.copy()

	# data_format = data_format + '-phy-{}-m-{}'.format(channel_type_phy_train, phy_method)	

#--------------------------------------------------------------------------------------------
# Physical offset simulation (different days)
#--------------------------------------------------------------------------------------------
if add_cfo:

	print('\n---------------------------------------------')
	print('Physical offset simulation (different days)')
	print('---------------------------------------------')
	print('Physical offsets: Train: {}, Test: {} ppm'.format(df_phy_train*1e6, df_phy_test*1e6))
	print('Physical seeds: Train: {}, Test: {}\n'.format(seed_phy_train_cfo, seed_phy_test_cfo))

	num_classes = y_train_orig.shape[1]

	x_test_orig = dict_wifi['x_test'].copy()
	y_test_orig = dict_wifi['y_test'].copy()

	fc_test_orig = dict_wifi['fc_test']

	if phy_method_cfo == 0:
		
		signal_ch = x_test_orig.copy()
		for n in trange(num_classes):
			ind_n = np.where(y_test_orig.argmax(axis=1)==n)[0]
			seed_phy_test_n = seed_phy_test_cfo + n
			for i in ind_n:
				rv_n = np.random.RandomState(seed=seed_phy_test_n)
				signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																	 df = rv_n.binomial(n = 1, p=0.5) * 2 * df_phy_train - df_phy_train,
																	 fc = fc_test_orig[i:i+1], 
																	 fs = fs)
		dict_wifi['x_test'] = signal_ch.copy()
	else :

		signal_ch = x_test_orig.copy()
		for n in trange(num_classes):
			ind_n = np.where(y_test_orig.argmax(axis=1)==n)[0]
			seed_phy_test_n = seed_phy_test_cfo + n
			for i in ind_n:
				rv_n = np.random.RandomState(seed=seed_phy_test_n)
				signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																	 df = rv_n.uniform(low=-df_phy_test, high=df_phy_test),
																	 fc = fc_test_orig[i:i+1], 
																	 fs = fs)
		dict_wifi['x_test'] = signal_ch.copy()
	del signal_ch, x_test_orig, x_train_orig, y_train_orig, y_test_orig, fc_test_orig
	

	# data_format = data_format + '_cfo_{}'.format(np.int(df_phy_train*1000000))

dict_wifi_no_aug = copy.deepcopy(dict_wifi)

#--------------------------------------------------------------------------------------------
# Physical offset compensation 
#--------------------------------------------------------------------------------------------
if remove_cfo:

	x_test = dict_wifi['x_test'].copy()
	complex_test = x_test[..., 0] + 1j* x_test[..., 1]

	del x_test

	complex_test_removed_cfo = complex_test.copy()

	freq_test = np.zeros([num_test, 2])
	for i in trange(num_test):
		complex_test_removed_cfo[i], freq_test[i] = offset_compensate_preamble(complex_test[i], fs = fs, verbose=False, option=2)

	dict_wifi['x_test'] = np.concatenate((complex_test_removed_cfo.real[..., None], complex_test_removed_cfo.imag[..., None]), axis=-1)
	
	# data_format = data_format + '_comp'

#--------------------------------------------------------------------------------------------
# Equalization
#--------------------------------------------------------------------------------------------

if equalize_test is True:
	# print('\nEqualizing test preamble')
	complex_test = dict_wifi['x_test'][..., 0] + 1j* dict_wifi['x_test'][..., 1]

	for i in trange(num_test):
		complex_test[i] = basic_equalize_preamble(complex_test[i], 
													  fs=fs, 
													  verbose=verbose_test)

	dict_wifi['x_test'] = np.concatenate((complex_test.real[..., None], complex_test.imag[..., None]), axis=2)

	del complex_test

#--------------------------------------------------------------------------------------------
# Channel augmentation
#--------------------------------------------------------------------------------------------
if augment_channel is True:

	seed_aug = np.max(seed_phy_train) + seed_phy_test + num_classes + 1

	dict_wifi, data_format = augment_with_channel_test(dict_wifi = dict_wifi, 
													  aug_type = aug_type, 
													  channel_method = channel_method, 
													  num_aug_train = num_aug_train, 
													  num_aug_test = num_aug_test, 
													  keep_orig_train = keep_orig_train, 
													  keep_orig_test = keep_orig_test, 
													  num_ch_train = num_ch_train, 
													  num_ch_test = num_ch_test, 
													  channel_type_aug_train = channel_type_aug_train, 
													  channel_type_aug_test = channel_type_aug_test, 
													  delay_seed_aug_test = delay_seed_aug_test, 
													  snr_test = snr_test, 
													  noise_method = noise_method, 
													  seed_aug = seed_aug, 
													  sampling_rate = sampling_rate,
													  data_format = data_format)

#--------------------------------------------------------------------------------------------
# Carrier Frequency Offset augmentation
#--------------------------------------------------------------------------------------------
if augment_cfo is True:

	print('\nCFO augmentation')
	print('\tAugmentation type: {}'.format(aug_type_cfo))
	print('\tNo of augmentations: Test: {}, \n\tKeep originals: Test: {}'.format(num_aug_test_cfo, keep_orig_test_cfo))
	
	print('\tCFO aug type: {}\n'.format(aug_type_cfo))

	x_test_aug = dict_wifi['x_test'].copy()
	y_test_aug = dict_wifi['y_test'].copy()

	fc_test_orig = dict_wifi['fc_test']

	if aug_type_cfo == 0:
		for k in tqdm(range(num_aug_test_cfo)):
			signal_ch = dict_wifi['x_test'].copy()
			# import ipdb; ipdb.set_trace()
			signal_ch = add_freq_offset(signal_ch, rand = rand_aug_test,
												   df = df_aug_test,
												   fc = fc_test_orig, 
												   fs = fs)
			if keep_orig_test is False:
				if k==0:
					x_test_aug = signal_ch.copy()
					y_test_aug = dict_wifi['y_test'].copy()
				else:
					x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
					y_test_aug = np.concatenate((y_test_aug, dict_wifi['y_test']), axis=0)
			else:
				x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
				y_test_aug = np.concatenate((y_test_aug, dict_wifi['y_test']), axis=0)		
	
	elif aug_type_cfo == 1:
		offset_dict = {}
		for i in range(401):
			offset_dict[i] = seed_phy_test_cfo+seed_phy_test_cfo+num_classes+1			
		for k in tqdm(range(num_aug_test_cfo)):
			signal_ch = dict_wifi['x_test'].copy()
			for i in tqdm(range(num_test)):
				rv_n = np.random.RandomState(seed=offset_dict[np.argmax(dict_wifi['y_test'][i])])
				if rand_aug_test=='unif':
					df_n = rv_n.uniform(low=-df_aug_test, high=df_aug_test)
				elif rand_aug_test=='ber':
					df_n = rv_n.choice(a=np.array([-df_aug_test, df_aug_test]))
				signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																	 df = df_n,
																	 fc = fc_test_orig[i:i+1], 
																	 fs = fs)
				offset_dict[np.argmax(dict_wifi['y_test'][i])] += 1
			if keep_orig_test_cfo is False:
				if k==0:
					x_test_aug = signal_ch.copy()
					y_test_aug = dict_wifi['y_test'].copy()
				else:
					x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
					y_test_aug = np.concatenate((y_test_aug, dict_wifi['y_test']), axis=0)
			else:
				x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
				y_test_aug = np.concatenate((y_test_aug, dict_wifi['y_test']), axis=0)			


	dict_wifi['x_test'] = x_test_aug.copy()
	dict_wifi['y_test'] = y_test_aug.copy()

	del x_test_aug, y_test_aug, fc_test_orig


	# data_format = data_format + 'augcfo-{}-df-{}-rand-{}-ty-{}-{}-t-'.format(num_aug_train, df_aug_train*1e6, rand_aug_train, aug_type, keep_orig_train)


#####################################  Buranin Altina bak


print("========================================") 
print("== BUILDING MODEL... ==")

if checkpoint_in is None:
	raise ValueError('Cannot test without a checkpoint')
	# data_input = Input(batch_shape=(batch_size, num_features, 2))
	# output, model_name = network_20_2(data_input, num_classes, weight_decay)
	# densenet = Model(data_input, output)

checkpoint_in = checkpoint_in + '.h5-new.h5'
densenet = load_model(checkpoint_in, 
					  custom_objects={'ComplexConv1D':ComplexConv1D,
					  				  'GetAbs': utils.GetAbs,
					  				  'Modrelu': Modrelu})

batch_size = 100

# import ipdb; ipdb.set_trace()

x_test = dict_wifi['x_test']
y_test = dict_wifi['y_test']
num_test_aug = dict_wifi['x_test'].shape[0]
num_classes = dict_wifi['y_train'].shape[1]

probs = densenet.predict(x=x_test, batch_size=batch_size, verbose=0)
label_pred = probs.argmax(axis=1) 
label_act = y_test.argmax(axis=1) 
ind_correct = np.where(label_pred==label_act)[0] 
ind_wrong = np.where(label_pred!=label_act)[0] 
assert (num_test_aug == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
test_acc = 100.*ind_correct.size / num_test_aug
print("\n========================================") 
print('Test accuracy augmented big test data: {:.2f}%'.format(test_acc))

output_dict = odict(acc=odict(), comp=odict(), loss=odict())

if num_test_aug != num_test:
	
	num_test_per_aug = num_test_aug // num_test

	embeddings = densenet.layers[-2].output

	model2 = Model(densenet.input, embeddings)


	logits_test = model2.predict(x=dict_wifi['x_test'],
								 batch_size=batch_size,
							 	 verbose=0)		

	softmax_test = densenet.predict(x=dict_wifi['x_test'],
								 batch_size=batch_size,
							 	 verbose=0)	

	layer_name = densenet.layers[-1].name
	weight, bias = densenet.get_layer(layer_name).get_weights()

	logits_test = logits_test.dot(weight) + bias


	logits_test_new = np.zeros((num_test, num_classes))
	softmax_test_new = np.zeros((num_test, num_classes))
	for i in range(num_test_per_aug):
		# list_x_test.append(x_test_aug[i*num_test:(i+1)*num_test])

		logits_test_new += logits_test[i*num_test:(i+1)*num_test]
		softmax_test_new += softmax_test[i*num_test:(i+1)*num_test]

	# Adding LLRs for num_channel_aug_test test augmentations
	label_pred_llr = logits_test_new.argmax(axis=1)
	label_act = dict_wifi['y_test'][:num_test].argmax(axis=1) 
	ind_correct = np.where(label_pred_llr==label_act)[0] 
	ind_wrong = np.where(label_pred_llr!=label_act)[0] 
	assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
	test_acc_llr = 100.*ind_correct.size / num_test


	# Adding LLRs for num_channel_aug_test test augmentations
	label_pred_soft = softmax_test_new.argmax(axis=1)
	label_act = dict_wifi['y_test'][:num_test].argmax(axis=1) 
	ind_correct = np.where(label_pred_soft==label_act)[0] 
	ind_wrong = np.where(label_pred_soft!=label_act)[0] 
	assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
	test_acc_soft = 100.*ind_correct.size / num_test

	# 1 test augmentation
	probs = densenet.predict(x=dict_wifi['x_test'][:num_test],
							 batch_size=batch_size,
							 verbose=0)
	label_pred = probs.argmax(axis=1)
	ind_correct = np.where(label_pred==label_act)[0] 
	ind_wrong = np.where(label_pred!=label_act)[0] 
	assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
	test_acc = 100.*ind_correct.size / num_test

	# No test augmentations
	probs = densenet.predict(x=dict_wifi_no_aug['x_test'],
							 batch_size=batch_size,
							 verbose=0)
	label_pred = probs.argmax(axis=1)
	label_act = dict_wifi_no_aug['y_test'].argmax(axis=1) 
	ind_correct = np.where(label_pred==label_act)[0] 
	ind_wrong = np.where(label_pred!=label_act)[0] 
	assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
	test_acc_no_aug = 100.*ind_correct.size / num_test

	# print("\n========================================") 
	print('Test accuracy (0 aug): {:.2f}%'.format(test_acc_no_aug))
	print('Test accuracy (1 aug): {:.2f}%'.format(test_acc))
	print('Test accuracy ({} aug) llr: {:.2f}%'.format(num_test_per_aug, test_acc_llr))
	print('Test accuracy ({} aug) softmax avg: {:.2f}%'.format(num_test_per_aug, test_acc_soft))



	output_dict['acc']['test_zero_aug'] = test_acc_no_aug
	output_dict['acc']['test_one_aug'] = test_acc
	output_dict['acc']['test_many_aug'] = test_acc_llr
	output_dict['acc']['test_many_aug_soft_avg'] = test_acc_soft

else:
	probs = densenet.predict(x=dict_wifi['x_test'],
							 batch_size=batch_size,
							 verbose=0)
	label_pred = probs.argmax(axis=1)
	label_act = y_test_orig.argmax(axis=1) 
	ind_correct = np.where(label_pred==label_act)[0] 
	ind_wrong = np.where(label_pred!=label_act)[0] 
	assert (dict_wifi['x_test'].shape[0]== ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
	test_acc_no_aug = 100.*ind_correct.size / dict_wifi['x_test'].shape[0]

	# print("\n========================================") 
	print('Test accuracy (no aug): {:.2f}%'.format(test_acc_no_aug))
	output_dict['acc']['test'] = test_acc_no_aug



acc_class = np.zeros([num_classes])
for class_idx in range(num_classes):
	idx_inclass = np.where(label_act==class_idx)[0]
	ind_correct = np.where(label_pred[idx_inclass]==label_act[idx_inclass])[0] 
	acc_class[class_idx] = 100*ind_correct.size / idx_inclass.size


# print(densenet.summary())
# for layer in densenet.layers:
# 	print(layer.name)
# densenet = ...  # create the original model

######################################
# Mean and cov_train
######################################

x_test_classes = [None]*19

for n in range(19):
	ind_n = np.where(y_test.argmax(axis=1)==n)[0]
	x_test_classes[n] = x_test[ind_n]

x_test_classes = np.array(x_test_classes)


layer_name = densenet.layers[-1].name
print(layer_name)
model_2 = Model(inputs=densenet.input,
                outputs=densenet.get_layer(layer_name).input)
weight, bias = densenet.get_layer(layer_name).get_weights()





logits_test = model_2.predict(x=x_test, batch_size=batch_size, verbose=0)
logits_test = logits_test.dot(weight) + bias

# logits_train = model_2.predict(x=x_train, batch_size=batch_size, verbose=0)
# logits_train = logits_train.dot(weight) + bias


def get_layer_statistics(densenet, which_layer):
	'''

	Input:
		densenet: Neural Network model
		which_layer: Layer name whose output will be investigated
	Output:
		features:
		weight:
		bias:
	'''

	layer_name = densenet.layers[which_layer].name

	intermediate_layer_model = Model(inputs=densenet.input,
                                 outputs=densenet.get_layer(layer_name).output)

	features_test_classes = [None]*19
	for n in range(19):
		features_test_classes[n] = intermediate_layer_model.predict(x_test_classes[n], batch_size=batch_size)
	features_test_classes = np.array(features_test_classes)

	print(layer_name)
	if which_layer<-1 and which_layer not in [-6,-3,-8]:
		layer_name = densenet.layers[which_layer+1].name
		weight, bias = densenet.get_layer(layer_name).get_weights()
		return features_test_classes, weight, bias
	else:
		return features_test_classes

def plot_activation_signature(activation_count, temporal_index, layer_ord):
	signals_directory = "/home/rfml/wifi/scripts/images/"
	if not os.path.exists(signals_directory):
		os.makedirs(signals_directory)

	fig = plt.figure()

	ax = plt.gca()

	pos = ax.imshow(activation_count)

	plt.xticks(np.arange(0, 19, 5), np.arange(1, 20, 5))
	plt.yticks(np.arange(0, activation_count.shape[0], 5), np.arange(1, activation_count.shape[0]+1, 5))
	plt.xlabel("Classes")
	plt.ylabel("Neurons")
	plt.rc('xtick', labelsize = 6)
	plt.rc('ytick', labelsize = 8)

	fig_name = os.path.join(signals_directory, "activation-map-layer{:}-temp{:}-aug{:}".format(layer_ord, temporal_index, num_aug_train) + '.pdf')
	# plt.title(" Activation Map")
	# fig.colorbar(pos, ax=ax)

	fig.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')

def find_angles(weight):
	angles = np.array([[0.]*19]*19)
	for i in range(19):
		for j in range(19):
			angles[i,j] = weight[:,i].dot(weight[:,j])/(np.linalg.norm(weight[:,i])*np.linalg.norm(weight[:,j]))
	return angles

def find_angles_embedding_matched_filter(embedding, weight):
	angles = np.array([[[0.]*19]*100]*19)
	for i in range(19):
		for k in range(100):
			for j in range(19):
				angles[i,k,j] = embedding[i,k].dot(weight[:,j])/(np.linalg.norm(embedding[i,k])*np.linalg.norm(weight[:,j]))
	return angles

import ipdb; ipdb.set_trace()





plot_activation_map = True
if plot_activation_map == True:
	layer_ord = -4
	features_test_classes, weight, bias = get_layer_statistics(densenet, which_layer = layer_ord)
	temporal_index=0
	activation_count = np.array([[0]*19]*100)
	for i in range(19):
		for j in range(100):
			activation_count[np.where(np.abs(features_test_classes[i,j,temporal_index])>0.000001),i] +=1
	plot_activation_signature(activation_count, temporal_index, layer_ord)

	layer_ord = -5
	features_test_classes, weight, bias = get_layer_statistics(densenet, which_layer = layer_ord)
	temporal_index=0
	activation_count = np.array([[0]*19]*100)
	for i in range(19):
		for j in range(100):
			activation_count[np.where(np.abs(features_test_classes[i,j,temporal_index])>0.000001),i] +=1
	plot_activation_signature(activation_count, temporal_index, layer_ord)

	layer_ord = -3
	features_test_classes = get_layer_statistics(densenet, which_layer = layer_ord)
	temporal_index=0
	activation_count = np.array([[0]*19]*100)
	for i in range(19):
		for j in range(100):
			activation_count[np.where(np.abs(features_test_classes[i,j,temporal_index])>0.000001),i] +=1
	plot_activation_signature(activation_count, temporal_index, layer_ord)

	layer_ord = -2
	features_test_classes, weight, bias = get_layer_statistics(densenet, which_layer = layer_ord)
	temporal_index=0
	activation_count = np.array([[0]*19]*100)
	for i in range(19):
		for j in range(100):
			activation_count[np.where(np.abs(features_test_classes[i,j])>0.000001),i] +=1
	plot_activation_signature(activation_count, temporal_index, layer_ord)

	layer_ord = -1
	features_test_classes = get_layer_statistics(densenet, which_layer = layer_ord)
	temporal_index=0
	activation_count = np.array([[0]*19]*19)
	for i in range(19):
		for j in range(100):
			activation_count[np.argmax(np.abs(features_test_classes[i,j])),i] +=1
	plot_activation_signature(activation_count, temporal_index, layer_ord)

layer_ord = -2
features_test_classes, weight, bias = get_layer_statistics(densenet, which_layer = layer_ord)



angles = find_angles(weight)
angles_embed = find_angles_embedding_matched_filter(features_test_classes, weight)
angles_embed_avg = np.mean(angles_embed, axis=1)

def plot_angles(angles, name_fig):

	signals_directory = "/home/rfml/wifi/scripts/images/"
	if not os.path.exists(signals_directory):
		os.makedirs(signals_directory)

	fig = plt.figure()

	ax = plt.gca()

	pos = ax.imshow(angles)
	plt.xticks(np.arange(0, 19, 2), np.arange(1, 20, 2))
	plt.yticks(np.arange(0, 19, 2), np.arange(1, 20, 2))
	plt.xlabel("Weight Vector")
	plt.ylabel("Weight Vector")

	fig_name = os.path.join(signals_directory, name_fig + '.pdf')
	# plt.title("Cosine(i,j)")
	fig.colorbar(pos, ax=ax)

	fig.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')

plot_angles(angles, name_fig = "matched_filter"+str(num_aug_train))
plot_angles(angles_embed_avg, name_fig = "embedding_vs_filters"+str(num_aug_train))
# layer_ord = -7
# features_test_classes, weight, bias = get_layer_statistics(densenet, which_layer = layer_ord)
# temporal_index=0
# activation_count = np.array([[0]*19]*100)
# for i in range(19):
# 	for j in range(100):
# 		activation_count[np.where(np.abs(features_test_classes[i,j])>0.000001),i] +=1
# plot_activation_signature(activation_count, temporal_index, layer_ord)

# import ipdb; ipdb.set_trace()



# layer_name = densenet.layers[1].name
# weight = densenet.get_layer(layer_name).get_weights()

#########

# w = weight[:, 0, :200] + 1j*weight[:, 0, 200:]
# print(w.shape)

# X = w.copy() # shape [num_test, num_classes]
# ind = np.where(np.abs(X).max(axis=0) > 1e-10)[0]
# X = X[:, ind]
# X -= X.mean(axis=0)
# X /= X.std(axis=0)
# X = np.nan_to_num(X)
# rho_w = np.matmul(X.T.conj(), X) / X.shape[0]

# plt.figure(figsize=[5, 5])
# plt.imshow(rho_w.real, vmin=-1.0, vmax=1.0)   
# plt.colorbar() 
# plt.tight_layout(rect=[0, 0.03, 1, 0.9])
# plt.subplots_adjust(wspace=0.3, hspace=0.4)
# plt.title('Covariance of filters')
# # plt.savefig('filt_cov.pdf', format='pdf', dpi=1000, bbox_inches='tight')

# from IPython import embed; embed()
# ipdb.set_trace()
##########################

# for i in range(5):
# 	print(np.count_nonzero(logits_test[i, :]))

# for i in range(5):
# 	print(np.count_nonzero(features_test[i, :]))



############


# mu_train, cov_train, rho_train = mu_cov_calc(logits_train, y_train)
# mu_test, cov_test, rho_test = mu_cov_calc(logits_test, y_test)

# cov_eye = np.zeros([num_classes, num_classes, num_classes])
# for n in range(num_classes):
# 	cov_eye[n] = np.identity(num_classes)

# cov_train_diag = np.zeros([num_classes, num_classes, num_classes])
# for n in range(num_classes):
# 	cov_train_diag[n] = np.diag(np.diag(cov_train[n]))

# prec_rho_train = np.zeros([num_classes, num_classes, num_classes])
# for i in range(num_classes):
# 	prec_rho_train[i] = np.linalg.inv(rho_train[i])
# 	# if i<5:
# 	# 	print(100 - 100*np.count_nonzero(prec_rho_train[i])/num_classes/num_classes)
# 	# 	print(np.linalg.det(rho_train[i]))
# 	# 	print(np.linalg.cond(rho_train[i]))

# Z = Z_calc(logits_test, mu_train, cov_train_diag)

# label_z = Z.argmax(axis=1)
# ind_correct_z = np.where(label_z==label_act)[0] 
# test_acc_z = 100.*ind_correct_z.size / num_test
# print('\nTest accuracy: {:.2f}% \n Z acc: {:.2f}%'.format(test_acc, test_acc_z))

# # acc_z_i = np.zeros([num_classes])
# # for class_idx in range(num_classes):
# # 	idx_inclass = np.where(label_act==class_idx)[0]
# # 	ind_correct = np.where(label_z[idx_inclass]==label_act[idx_inclass])[0] 
# # 	acc_z_i[class_idx] = 100*ind_correct.size / idx_inclass.size

# from IPython import embed; embed()
# ipdb.set_trace()

# label_test = label_act.copy()

# for j in range(2):
# 	# w_proj = vh_test_class[j][:np.int(n_comp_test_class[j])].dot(weight)
# 	# w_proj = vh_test_class[j][:10].dot(weight)
# 	idx_inclass = np.where(label_test==j)[0]
# 	Z_j = Z[idx_inclass]

# 	plt.figure(figsize=(15,2))
# 	for i in range(5):
# 		plt.subplot(1, 5, i+1)
# 		plt.hist(Z_j[:, i], density=True, bins=15)
# 		# plt.hist(w_proj[:, i], density=True)
# 		# plt.xlim([-1, 1])
# 		# plt.ylim([0, 2.5])
# 		# plt.xlim([-3, 3])
# 		# plt.ylim([0, 1.2])
# 		# plt.ylim([0, 0.8])
# 		# plt.xticks(np.arange(1, k+1, 2))
# 		# plt.xlabel(r'$\left|\support\left(\bx\right) \cap \support\left(\bx+\be\right)\right|$, where K = {:.0f}'.format(k), fontsize=11)
# 		plt.title('Class {}'.format(i), fontsize=12)
# 		plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 
# 		# plt.ylim([0, 5000])
# 	plt.suptitle('Full decision rule for images of class {})'.format(j))	

# k = 100
# Z_top_k, evals_cov, snr_cov = Z_calc_smallest_k(logits_test, mu_train, cov_train_diag, k)


# for j in range(2):
# 	# w_proj = vh_test_class[j][:np.int(n_comp_test_class[j])].dot(weight)
# 	# w_proj = vh_test_class[j][:10].dot(weight)
# 	idx_inclass = np.where(label_test==j)[0]
# 	Z_j = Z_top_k[idx_inclass, :]

# 	plt.figure(figsize=(15,2))
# 	for i in range(5):
# 		plt.subplot(1, 5, i+1)
# 		plt.hist(Z_j[:, i], density=True, bins=15)
# 		# plt.hist(w_proj[:, i], density=True)
# 		# plt.xlim([-1, 1])
# 		# plt.ylim([0, 2.5])
# 		# plt.xlim([-3, 3])
# 		# plt.ylim([0, 1.2])
# 		# plt.ylim([0, 0.8])
# 		# plt.xticks(np.arange(1, k+1, 2))
# 		# plt.xlabel(r'$\left|\support\left(\bx\right) \cap \support\left(\bx+\be\right)\right|$, where K = {:.0f}'.format(k), fontsize=11)
# 		plt.title('Class {}'.format(i), fontsize=12)
# 		plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 
# 		# plt.ylim([0, 5000])

# 	plt.suptitle('Smallest-{} decision rule for images of class {})'.format(k, j))	

# # for j in range(2):

# # 	plt.figure(figsize=(15,2))
# # 	for i in range(5):
# # 		plt.subplot(1, 5, i+1)
# # 		plt.hist(evals_cov[i], density=True, bins=15)
# # 		# plt.hist(w_proj[:, i], density=True)
# # 		# plt.xlim([-1, 1])
# # 		# plt.ylim([0, 2.5])
# # 		# plt.xlim([-3, 3])
# # 		# plt.ylim([0, 1.2])
# # 		# plt.ylim([0, 0.8])
# # 		# plt.xticks(np.arange(1, k+1, 2))
# # 		# plt.xlabel(r'$\left|\support\left(\bx\right) \cap \support\left(\bx+\be\right)\right|$, where K = {:.0f}'.format(k), fontsize=11)
# # 		plt.title('Class {}'.format(i), fontsize=12)
# # 		plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 

# # 	plt.suptitle('Eigenvalues for images of class {})'.format(j))	


# plt.figure(figsize=(15,2))
# for i in np.arange(35, 45):
# 	plt.subplot(1, 10, i+1-35)
# 	plt.hist(snr_cov[i], density=True, bins=15)
# 	# plt.hist(w_proj[:, i], density=True)
# 	# plt.xlim([-1, 1])
# 	# plt.ylim([0, 2.5])
# 	# plt.xlim([-3, 3])
# 	# plt.ylim([0, 1.2])
# 	# plt.ylim([0, 0.8])
# 	# plt.xticks(np.arange(1, k+1, 2))
# 	# plt.xlabel(r'$\left|\support\left(\bx\right) \cap \support\left(\bx+\be\right)\right|$, where K = {:.0f}'.format(k), fontsize=11)
# 	plt.title('Class {}'.format(i), fontsize=12)
# 	plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 

# plt.suptitle('SNR')	

# plt.figure(figsize=(15,2))
# for i in np.arange(35, 45):
# 	plt.subplot(1, 10, i+1-35)
# 	plt.hist(1./evals_cov[i], density=True, bins=15)
# 	# plt.hist(w_proj[:, i], density=True)
# 	# plt.xlim([-1, 1])
# 	# plt.ylim([0, 2.5])
# 	# plt.xlim([-3, 3])
# 	# plt.ylim([0, 1.2])
# 	# plt.ylim([0, 0.8])
# 	# plt.xticks(np.arange(1, k+1, 2))
# 	# plt.xlabel(r'$\left|\support\left(\bx\right) \cap \support\left(\bx+\be\right)\right|$, where K = {:.0f}'.format(k), fontsize=11)
# 	plt.title('Class {}'.format(i), fontsize=12)
# 	plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 

# 	plt.suptitle('1/eigenvalues')	

# z_bias = np.array([100*(label_z==i).mean() for i in np.arange(100)])
# plt.figure()
# plt.plot(z_bias)
# plt.xlabel('Classes')
# plt.title('No of times each class is predicted by decision rule')

# plt.figure()
# plt.plot(Z.mean(axis=0))
# plt.xlabel('Classes')
# plt.title('Mean value of decision statistic for each class')


# num_classes_plot = 19
# plt.figure(figsize=[15, 6])
# num_rows = 4
# num_cols = 5
# for i in range(num_classes_plot):
# 	plt.subplot(num_rows, num_cols, i+1)
# 	plt.bar(np.arange(num_classes), height = mu_train[i]- mu_train[i].min())   
# 	plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 
# 	plt.title('Class {}'.format(i))
# plt.suptitle('Mean')

# plt.figure(figsize=[15, 10])
# for i in range(num_classes_plot):
# 	plt.subplot(num_rows, num_cols, i+1)
# 	plt.imshow(rho_train[i], vmin=-1.0, vmax=1.0)   
# 	plt.colorbar() 
# 	plt.tight_layout(rect=[0, 0.03, 1, 0.9])
# 	plt.subplots_adjust(wspace=0.3, hspace=0.4)
# 	plt.title('Class {} (acc = {:.0f})'.format(i, acc_class[i]))
# plt.suptitle('Covariance coefficient')


# plt.figure(figsize=[15, 10])
# for i in range(num_classes_plot):
# 	plt.subplot(num_rows, num_cols, i+1)
# 	plt.imshow(prec_rho_train[i])   
# 	plt.colorbar() 
# 	plt.tight_layout(rect=[0, 0.03, 1, 0.9])
# 	plt.subplots_adjust(wspace=0.3, hspace=0.4)
# 	plt.title('Class {} (acc = {:.0f})'.format(i, acc_class[i]))
# plt.suptitle('Precision')






