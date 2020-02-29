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

import keras
from keras.models import Model, load_model
from keras import backend as K
from keras.utils import plot_model
from keras import optimizers, regularizers, losses
from keras.layers import Dense, Input, Activation, Conv1D, Dropout, GlobalAveragePooling1D, Lambda, Average
from keras.regularizers import l2

from .models_globecom import network_20_modrelu_short, network_20_reim, network_20_reim_2x, network_20_reim_sqrt2x, network_20_mag, network_200_modrelu_short, network_200_reim, network_200_reim_2x, network_200_reim_sqrt2x, network_200_mag, network_200_modrelu_short_shared

from .complexnn import ComplexDense, ComplexConv1D, utils

from .models_adsb import Modrelu

from .train_globecom import train_20, train_200

# from freq_offset import estimate_freq_offset -- TO BE REMOVED!!! ---

from ..preproc.preproc_wifi import basic_equalize_preamble, offset_compensate_preamble
from ..preproc.fading_model import normalize, add_custom_fading_channel, add_freq_offset

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

	x_test = dict_wifi['x_test'].copy()
	y_test = dict_wifi['y_test'].copy()

	# print('\n-------------------------------')

	print('\nChannel augmentation')
	print('\tAugmentation type: {}'.format(aug_type))
	print('\tNo of augmentations: Train: {}, Test: {}\n\tKeep originals: Train: {}, Test: {}'.format(num_aug_train, num_aug_test, keep_orig_train, keep_orig_test))
	print('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))
	print('\tChannel type: Train: {}, Test: {}\n'.format(channel_type_aug_train, channel_type_aug_test))

	seed_aug_offset = np.max(seed_phy_train) + seed_phy_test + num_classes + 1
	print("Seed: Train: {:}".format(seed_aug_offset))



	x_test_aug = x_test.copy()
	y_test_aug = y_test.copy()

	if num_ch_test < -1:
		raise ValueError('num_ch_test')
	elif num_ch_test!=0:
		for k in tqdm(range(num_aug_test)):
			signal_ch = np.zeros(x_test.shape)
			for i in tqdm(range(num_test)):
				signal = dict_wifi['x_test'][i][:,0]+1j*dict_wifi['x_test'][i][:,1]
				if num_ch_test==-1:
					signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, 
																		seed=seed_aug_offset + num_train*num_aug_train + 1 + (i + k*num_test) % (num_test*num_aug_test), 
																		beta=0, 
																		delay_seed=delay_seed_aug_test,
																		channel_type=channel_type_aug_test,
																		channel_method=channel_method,
																		noise_method=noise_method)
				else:
					signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, 
																		# seed = 1, 
																		seed = seed_aug_offset + num_train*num_aug_train + 1 + (i % num_ch_test) + k * num_ch_test, 
																		beta=0, 
																		delay_seed=delay_seed_aug_test,
																		channel_type=channel_type_aug_test,
																		channel_method=channel_method,
																		noise_method=noise_method)
				
				signal_faded = normalize(signal_faded)
				signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1)
				# dict_wifi['x_test'][i] = signal_ch
			if keep_orig_test is False:
				if k==0:
					x_test_aug = signal_ch.copy()
					y_test_aug = y_test.copy()
				else:
					x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
					y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)
			else:
				x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
				y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)		

	

	dict_wifi['x_test'] = x_test_aug.copy()
	dict_wifi['y_test'] = y_test_aug.copy()

	del x_test_aug, y_test_aug

	# data_format = data_format + 'aug-{}-art-{}-ty-{}-nch-{}-snr-{:.0f}'.format(num_aug_train, channel_type_aug_train, aug_type, num_ch_train, snr_train)



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

print(densenet.summary())
# print('aaaa')
for layer in densenet.layers:
	print(layer.name)
# densenet = ...  # create the original model

######################################
# Mean and cov_train
######################################

layer_name = densenet.layers[-1].name
print(layer_name)
model_2 = Model(inputs=densenet.input,
                outputs=densenet.get_layer(layer_name).input)
weight, bias = densenet.get_layer(layer_name).get_weights()

logits_test = model_2.predict(x=x_test, batch_size=batch_size, verbose=0)
logits_test = logits_test.dot(weight) + bias

# logits_train = model_2.predict(x=x_train, batch_size=batch_size, verbose=0)
# logits_train = logits_train.dot(weight) + bias

layer_name = densenet.layers[-2].name
print(layer_name)
# layer_name = 'dense_1'
# layer_name = 'dense_2'
# layer_name = 'dense_3'
# layer_name = 'dense_5'
# layer_name = 'dense_35'
intermediate_layer_model = Model(inputs=densenet.input,
                                 outputs=densenet.get_layer(layer_name).output)
features_test = intermediate_layer_model.predict(x_test, batch_size=batch_size)
# features_train = intermediate_layer_model.predict(x_train, batch_size=batch_size)

layer_name = densenet.layers[1].name
weight = densenet.get_layer(layer_name).get_weights()

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






