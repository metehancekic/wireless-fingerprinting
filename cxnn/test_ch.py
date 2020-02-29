from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import numpy.random as random
from collections import OrderedDict as odict
from timeit import default_timer as timer
from tqdm import tqdm, trange
from sklearn import metrics

import keras
from keras import backend as K
from keras.utils import plot_model
from keras import optimizers, regularizers, losses
from keras.layers import Dense, Input, Activation, Conv1D, Dropout, GlobalAveragePooling1D, Lambda, Average
from keras.models import Model, load_model
from keras.regularizers import l2

# import complexnn
from .complexnn import ComplexDense, ComplexConv1D, utils

from .models_wifi import *
from .models_adsb import Modrelu

from ..preproc.preproc_wifi import rms, basic_equalize_preamble
from ..preproc.fading_model import normalize, add_custom_fading_channel

exp_dir = '/home/rfml/wifi/experiments/exp19'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2'

preprocess_type = 1
# preprocess_type = 2

# sample_rate = 20
sample_rate = 200

sample_duration = 16
# sample_duration = 32

#-------------------------------------------------
# Model name
#-------------------------------------------------

# Regular model
# model_name = '-modrelu-100-100-new-'
model_name = '-100C200x100-ModReLU-100C10x1-ModReLU-Abs-100shared_D-100shared_D-Avg'

# Early abs
# model_name = '-100C200x100-ModReLU-Abs-100shared_C10x1-100shared_D-100shared_D-Avg'

# Short stride
# model_name = '-100C200x10-ModReLU-100C10x1-ModReLU-Abs-100shared_D-100shared_D-Avg'

# Short conv
# model_name = '-100C40x10-ModReLU-100C10x1-ModReLU-Abs-100shared_D-100shared_D-Avg'

# Short conv, but no modrelu
# model_name = '-100C40x10-ModReLU-100C10x1-Abs-100shared_D-100shared_D-Avg'

# Short conv and early abs
# model_name = '-100C40x10-ModReLU-Abs-100shared_C10x1-100shared_D-100shared_D-Avg'

# Short conv and early abs, but no modrelu
# model_name = '-100C40x10-Abs-100shared_C10x1-100shared_D-100shared_D-Avg'


#-------------------------------------------------
# Physical channel params
#-------------------------------------------------
'''
channel type:
	1 - Extended Pedestrian A
	2 - Extended Vehicular A
	3 - Extended Typical Urban
'''
channel_type_phy_train = 1
channel_type_phy_test = 1

#-------------------------------------------------
# Physical seeds
#-------------------------------------------------

'''
phy_method:
	0 - same channel for all packets
	1 - different channel for each class, same channel for all packets in a class
	2 - train on 2 days, test on 1 day, with a diff channel for each class
'''
# phy_method = 0
phy_method = 1
# phy_method = 2
# phy_method = 3

#--------------------
# If phy_method = 0
#--------------------
# seed_phy_pairs = [(30, 31)]
# seed_phy_pairs = [(40, 41)]
# seed_phy_pairs = [(0, 1), (10, 11), (20, 21), (30, 31), (40, 41)]

#--------------------
# If phy_method = 1
#--------------------
# seed_phy_pairs = [(0, 20)]
seed_phy_pairs = [(40, 60)]
# seed_phy_pairs = [(80, 100)]
# seed_phy_pairs = [(0, 20), (40, 60), (80, 100), (120, 140), (160, 180)]

#--------------------
# If phy_method = 2
#--------------------
# seed_phy_pairs = [((0, 20), 40)]
# seed_phy_pairs = [((20, 40), 60)]
# seed_phy_pairs = [((40, 40), 60)]

#--------------------
# If phy_method = 3
#--------------------
# seed_phy_pairs = [((0, 20, 40), 60)]

#-------------------------------------------------
# Equalization params
#-------------------------------------------------
# equalize_train = False
equalize_train = True

# equalize_test = False
equalize_test = True

verbose_train = False
# verbose_train = True

verbose_test = False
# verbose_test = True

#-------------------------------------------------
# Augmentation channel params
#-------------------------------------------------
# channel_type_aug_trains = [1, 2, 3]
channel_type_aug_trains = [1]
# channel_type_aug_train = 1

channel_type_aug_tests = [1]

num_aug_train = 0
# num_aug_train = 5
# num_aug_train = 20
# num_aug_train = 100

num_aug_test = 0
# num_aug_test = 5
# num_aug_test = 20
# num_aug_test = 100

channel_method = 'FFT' 
# channel_method = 'RC' # Raised-cosine

noise_method = 'reg' # Regular
# noise_method = 'bl' # Bandlimited via root raised-cosine

# Random seed for delay perturbation
# Set to False for fixed delays
delay_seed_aug_train = False
delay_seed_aug_test = False
# delay_seed = None

'''
aug_type:
	0 - usual channel aug
	1 - same channel for ith example in each class
'''
# aug_type = 0
aug_type = 1

keep_orig_train = False
keep_orig_test = False
# keep_orig_train = True
# keep_orig_test = True

num_ch_train = -1
# num_ch_train = 1
num_ch_test = -1

snr_train = 500
snr_test = 500



data_format_plain = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

npz_filename = exp_dir + '/sym-' + data_format_plain + '.npz'

start = timer()
np_dict = np.load(npz_filename)
dict_wifi = {}
dict_wifi['x_train'] = np_dict['arr_0']
dict_wifi['y_train'] = np_dict['arr_1']
dict_wifi['x_test'] = np_dict['arr_2']
dict_wifi['y_test'] = np_dict['arr_3']
dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]
end = timer()
print('Load time: {:} sec'.format(end - start))

sampling_rate = sample_rate * 1e+6
fs = sample_rate * 1e+6

x_train = dict_wifi['x_train'].copy()
y_train = dict_wifi['y_train'].copy()
x_test = dict_wifi['x_test'].copy()
y_test = dict_wifi['y_test'].copy()
num_train = x_train.shape[0]
num_test = x_test.shape[0]
num_features = x_train.shape[1]
num_classes = dict_wifi['num_classes']

print('\nPreprocessing\n\tType: {}\n\tFs: {} MHz\n\tLength: {} us'.format(preprocess_type, sample_rate, sample_duration))

for seed_phy_train, seed_phy_test in seed_phy_pairs:
	#--------------------------------------------------------------------------------------------
	# Different day scenario simulation
	#--------------------------------------------------------------------------------------------

	print('\nPhysical channel simulation (different days)')
	print('\tMethod: {}'.format(phy_method))
	print('\tChannel type: Train: {}, Test: {}'.format(channel_type_phy_train, channel_type_phy_test))
	print('\tSeed: Train: {}, Test: {}\n'.format(seed_phy_train, seed_phy_test))

	# signal_ch = dict_wifi['x_test'].copy()
	# num_test = signal_ch.shape[0]
	# for i in tqdm(range(num_test)):
	# 	signal = dict_wifi['x_test'][i][:,0]+1j*dict_wifi['x_test'][i][:,1]
	# 	signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
	# 														seed=seed_phy_test, 
	# 														beta=0, 
	# 														delay_seed=False,
	# 														channel_type=channel_type_phy_test,
	# 														channel_method=channel_method,
	# 														noise_method=noise_method)
	# 	signal_faded = normalize(signal_faded)
	# 	signal_ch[i] = np.concatenate((signal_fadedt.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
	# # dict_wifi['x_test'] = signal_ch.copy()

	# x_test = signal_ch.copy()

	if phy_method == 0: # Same channel for all packets
		signal_ch = dict_wifi['x_test'].copy()
		num_test = signal_ch.shape[0]
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
		x_test = signal_ch.copy()

	else: # Different channel for each class, same channel for all packets in a class
		signal_ch = dict_wifi['x_test'].copy()
		num_test = signal_ch.shape[0]
		for n in trange(num_classes):
			ind_n = np.where(y_test.argmax(axis=1)==n)[0]
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
		x_test = signal_ch.copy()

	for channel_type_aug_train in channel_type_aug_trains:

		if equalize_train is False:
			if phy_method==0:
				data_format = 'aug-{}-phy-{}-s-{}-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(num_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, aug_type, num_ch_train, delay_seed_aug_train, snr_train) + data_format_plain
			else:
				data_format = 'a{}ug-{}-phy-{}-s-{}-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(phy_method, num_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, aug_type, num_ch_train, delay_seed_aug_train, snr_train) + data_format_plain
		else:
			if phy_method==0:
				data_format = 'aug-{}-phy-{}-s-{}-eq-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(num_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, aug_type, num_ch_train, delay_seed_aug_train, snr_train) + data_format_plain
			else:
				data_format = 'a{}ug-{}-phy-{}-s-{}-eq-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(phy_method, num_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, aug_type, num_ch_train, delay_seed_aug_train, snr_train) + data_format_plain

		# if phy_method==0:
		# 	data_format = 'aug-{}-phy-{}-s-{}-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(num_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, aug_type, num_ch_train, delay_seed_aug_train, snr_train) + data_format_plain
		# else:
		# 	data_format = 'a{}ug-{}-phy-{}-s-{}-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(phy_method, num_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, aug_type, num_ch_train, delay_seed_aug_train, snr_train) + data_format_plain

		# data_format = 'aug-{}-phy-{}-s-{}-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(num_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, aug_type, num_ch_train, delay_seed_aug_train, snr_train) + data_format_plain

		# checkpoint_in = exp_dir + '/ckpt-' + data_format +'.h5'
		# checkpoint_in += '-modrelu-100-100'+'-new-'
		# checkpoint_in += '-new.h5'

		#--------------------------------------------------------------------------------------------
		# Equalization
		#--------------------------------------------------------------------------------------------

		print('\nEqualization')
		print('\tTrain: {}, Test: {}'.format(equalize_train, equalize_test))

		if equalize_test is True:
			# print('\nEqualizing test preamble')
			complex_test = x_test[..., 0] + 1j* x_test[..., 1]

			for i in range(num_test):
				complex_test[i] = basic_equalize_preamble(complex_test[i], 
															  fs=fs, 
															  verbose=verbose_test)

			x_test = np.concatenate((complex_test.real[..., None], complex_test.imag[..., None]), axis=2).copy()

		# Checkpoint path
		checkpoint_in = exp_dir + '/ckpt-' + data_format +'.h5'
		checkpoint_in +=  model_name
		checkpoint_in += '-new.h5'

		print('\n-------------------------------')
		print("Loading model from checkpoint")
		print('Model name: {}'.format(model_name[1:]))
		batch_size = 100

		print('\nChannel augmentation')
		print('\tAugmentation type: {}'.format(aug_type))
		print('\tNo of augmentations: Train: {}, Test: {}, \n\tKeep originals: Train: {}, Test: {}'.format(num_aug_train, num_aug_test, keep_orig_train, keep_orig_test))
		print('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))

		#--------------------------------------------------------------------------------------------
		# Load network
		#--------------------------------------------------------------------------------------------

		if checkpoint_in is None:
			raise ValueError('Cannot test without a checkpoint')

		densenet = load_model(checkpoint_in, 
							  custom_objects={'ComplexConv1D':ComplexConv1D,
							  				  'GetAbs': utils.GetAbs,
							  				  'Modrelu': Modrelu})


		for channel_type_aug_test in channel_type_aug_tests:


			print('\tChannel type: Train: {}, Test: {}\n'.format(channel_type_aug_train, channel_type_aug_test))

			#--------------------------------------------------------------------------------------------
			# Channel augmentation for test data
			#--------------------------------------------------------------------------------------------

			seed_aug_offset = np.max(seed_phy_train) + seed_phy_test + num_classes + 1

			x_test_aug = x_test.copy()
			y_test_aug = y_test.copy()

			if num_ch_test < -1:
				raise ValueError('num_ch_test')
			elif num_ch_test!=0:
				for k in tqdm(range(num_aug_test)):
					signal_ch = np.zeros(x_test.shape)
					num_test = x_test.shape[0]
					for i in tqdm(range(num_test)):
						signal = x_test[i][:,0] + 1j*x_test[i][:,1]
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
																				seed = sseed_aug_offset +num_train*num_aug_train + 1 + (i % num_ch_test) + k * num_ch_test, 
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

			num_test = x_test_aug.shape[0]

			#--------------------------------------------------------------------------------------------
			# Add LLRs
			#--------------------------------------------------------------------------------------------

			# print('Training data size: {}'.format(x_train.shape))
			print('\nTest data size before aug: {}'.format(x_test.shape))
			print('Test data size after aug: {}\n'.format(x_test_aug.shape))

			output_dict = odict(acc=odict(), comp=odict(), loss=odict())

			if num_aug_test!=0:
				num_test_per_aug = num_test // num_aug_test

				logits = densenet.layers[-1].output

				model2 = Model(densenet.input, logits)

				logits_test = model2.predict(x=x_test_aug,
											 batch_size=batch_size,
										 	 verbose=0)		
				logits_test_new = np.zeros((num_test_per_aug, num_classes))
				for i in range(num_aug_test):
					# list_x_test.append(x_test_aug[i*num_test:(i+1)*num_test])

					logits_test_new += logits_test[i*num_test_per_aug:(i+1)*num_test_per_aug]

				# Adding LLRs for num_aug_test test augmentations
				label_pred_llr = logits_test_new.argmax(axis=1)
				label_act = y_test_aug[:num_test_per_aug].argmax(axis=1) 
				ind_correct = np.where(label_pred_llr==label_act)[0] 
				ind_wrong = np.where(label_pred_llr!=label_act)[0] 
				assert (num_test_per_aug == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
				test_acc_llr = 100.*ind_correct.size / num_test_per_aug

				# 1 test augmentation
				probs = densenet.predict(x=x_test_aug[:num_test_per_aug],
										 batch_size=batch_size,
										 verbose=0)
				label_pred = probs.argmax(axis=1)
				ind_correct = np.where(label_pred==label_act)[0] 
				ind_wrong = np.where(label_pred!=label_act)[0] 
				assert (num_test_per_aug == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
				test_acc = 100.*ind_correct.size / num_test_per_aug

				# No test augmentations
				probs = densenet.predict(x=x_test,
										 batch_size=batch_size,
										 verbose=0)
				label_pred = probs.argmax(axis=1)
				label_act = y_test.argmax(axis=1) 
				ind_correct = np.where(label_pred==label_act)[0] 
				ind_wrong = np.where(label_pred!=label_act)[0] 
				assert (x_test.shape[0] == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
				test_acc_no_aug = 100.*ind_correct.size / x_test.shape[0]

				# print("\n========================================") 
				print('Test accuracy (0 aug): {:.2f}%'.format(test_acc_no_aug))
				print('Test accuracy (1 aug): {:.2f}%'.format(test_acc))
				print('Test accuracy ({} aug): {:.2f}%'.format(num_aug_test, test_acc_llr))
				output_dict['acc']['test'] = test_acc_llr

			else:
				probs = densenet.predict(x=x_test,
										 batch_size=batch_size,
										 verbose=0)
				label_pred = probs.argmax(axis=1)
				label_act = y_test.argmax(axis=1) 
				ind_correct = np.where(label_pred==label_act)[0] 
				ind_wrong = np.where(label_pred!=label_act)[0] 
				assert (x_test.shape[0]== ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
				test_acc_no_aug = 100.*ind_correct.size / x_test.shape[0]

				# print("\n========================================") 
				print('Test accuracy (no aug): {:.2f}%'.format(test_acc_no_aug))
				output_dict['acc']['test'] = test_acc_no_aug

			print("\n========================================\n") 

			print('\nPreprocessing\n\tType: {}\n\tFs: {} MHz\n\tLength: {} us'.format(preprocess_type, sample_rate, sample_duration))

			print('\nPhysical channel simulation (different days)')
			print('\tMethod: {}'.format(phy_method))
			print('\tChannel type: Train: {}, Test: {}'.format(channel_type_phy_train, channel_type_phy_test))
			print('\tSeed: Train: {}, Test: {}'.format(seed_phy_train, seed_phy_test))

			print('\nEqualization')
			print('\tTrain: {}, Test: {}'.format(equalize_train, equalize_test))

			print('\nChannel augmentation')
			print('\tAugmentation type: {}'.format(aug_type))
			print('\tNo of augmentations: Train: {}, Test: {}\n\tKeep originals: Train: {}, Test: {}'.format(num_aug_train, num_aug_test, keep_orig_train, keep_orig_test))
			print('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))
			print('\tChannel type: Train: {}, Test: {}'.format(channel_type_aug_train, channel_type_aug_test))
			print("\n========================================\n") 

		import ipdb; ipdb.set_trace()

		del densenet
