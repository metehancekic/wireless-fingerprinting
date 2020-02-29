'''
Trains data for a WiFi experiment with carrier freq offset augmentation.

Data is read from npz files.
'''
import numpy as np
import numpy.random as random
from collections import OrderedDict as odict
from timeit import default_timer as timer
from tqdm import tqdm, trange
import ipdb

from ..preproc.preproc_wifi import rms
from ..preproc.fading_model import normalize, add_custom_fading_channel, add_freq_offset

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

exp_dirs = []
exp_dirs += ['/home/rfml/wifi/experiments/exp19']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2']

preprocess_type = 1
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
# Physical offset params
#-------------------------------------------------
df_phy_train = 40e-6
df_phy_test = 40e-6

# seed_phy_pairs = [(0, 20), (40, 60), (80, 100), (120, 140), (160, 180)]
# seed_phy_pairs = [(0, 20)]

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
	1 - different channel/offset for each class, same channel/offset for all packets in a class
	2 - train on 2 days, test on 1 day, with a diff channel/offset for each class
'''
# phy_method = 0
# phy_method = 1
# phy_method = 2
# phy_method = 3
phy_method = 4

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
# seed_phy_pairs = [(40, 60)]
# seed_phy_pairs = [(80, 100)]
# seed_phy_pairs = [(0, 20), (40, 60), (80, 100), (120, 140), (160, 180)]

#--------------------
# If phy_method = 2
#--------------------
# seed_phy_pairs = [((0, 20), 40)]
# seed_phy_pairs = [((20, 40), 60)]

#--------------------
# If phy_method = 3
#--------------------
# seed_phy_pairs = [((0, 20, 40), 60)]

#--------------------
# If phy_method = 4
#--------------------
seed_phy_pairs = [((0, 20, 40, 80), 60)]

#--------------------
# If phy_method = 10
#--------------------
# seed_phy_pairs = [((0, 20, 40, 80, 100, 120, 140, 160, 180, 200), 60)]

#-------------------------------------------------
# Augmentation channel params
#-------------------------------------------------
# channel_type_aug_trains = [1, 2, 3]
channel_type_aug_trains = [1]
# channel_type_aug_train = 1

channel_type_aug_tests = [1]

# num_channel_aug_train = 0
# num_channel_aug_train = 5
# num_channel_aug_train = 10
num_channel_aug_train = 20
# num_channel_aug_train = 50

# num_channel_aug_test = 0
num_channel_aug_test = 5
# num_channel_aug_test = 20
# num_channel_aug_test = 100

channel_method = 'FFT' 
# channel_method = 'RC' # Raised-cosine

noise_method = 'reg' # Regular
# noise_method = 'bl' # Bandlimited via root raised-cosine

# Random seed for delay perturbation
# Set to False for fixed delays
delay_seed_channel_aug_train = False
delay_seed_channel_aug_test = False
# delay_seed = None

'''
channel_aug_type:
	0 - usual channel aug
	1 - same channel for ith example in each class
'''
# channel_aug_type = 0
channel_aug_type = 1

keep_orig_train = False
keep_orig_test = False
# keep_orig_train = True
# keep_orig_test = True

num_ch_train = -1
num_ch_test = -1

snr_train = 500
snr_test = 500


#-------------------------------------------------
# Augmentation offset params
#-------------------------------------------------
df_aug_test = df_phy_train 
# df_aug_test = 200e-6  
rand_aug_test = 'unif'
# rand_aug_test = 'ber'
# rand_aug_test = 'False'

# num_df_aug_test = 0
num_df_aug_test = 1
# num_df_aug_test = 5
# num_df_aug_test = 20

keep_orig_test = False
# keep_orig_train = True


df_aug_train = df_phy_train 
# df_aug_train = 200e-6  
rand_aug_train = 'unif'
# rand_aug_train = 'ber'
# rand_aug_train = 'False'

# num_df_aug_train = 0
num_df_aug_train = 1
# num_df_aug_train = 5
# num_df_aug_train = 20

keep_orig_train = False
# keep_orig_train = True

'''
df_aug_type:
	0 - usual offset aug
	1 - same offset for ith example in each class
'''
df_aug_type = 0
# df_aug_type = 1

data_format_plain = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

exp_dir = exp_dirs[0]

npz_filename = exp_dir + '/sym-' + data_format_plain + '.npz'

start = timer()
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

num_train = dict_wifi['x_train'].shape[0]
num_test = dict_wifi['x_test'].shape[0]


fs = sample_rate * 1e+6
sampling_rate = sample_rate * 1e+6

x_train_orig = dict_wifi['x_train'].copy()
y_train_orig = dict_wifi['y_train'].copy()
num_classes = y_train_orig.shape[1]

x_test_orig = dict_wifi['x_test'].copy()
y_test_orig = dict_wifi['y_test'].copy()

fc_train_orig = dict_wifi['fc_train']
fc_test_orig = dict_wifi['fc_test']

for seed_phy_train, seed_phy_test in seed_phy_pairs:

	#--------------------------------------------------------------------------------------------
	# Physical offset simulation (different days)
	#--------------------------------------------------------------------------------------------

	print('\n---------------------------------------------')
	print('Physical offset simulation (different days)')
	print('---------------------------------------------')
	print('Physical offsets: Train: {}, Test: {} ppm'.format(df_phy_train*1e6, df_phy_test*1e6))
	print('Physical seeds: Train: {}, Test: {}\n'.format(seed_phy_train, seed_phy_test))

	signal_ch = x_test_orig.copy()
	for n in trange(num_classes):
		ind_n = np.where(y_test_orig.argmax(axis=1)==n)[0]
		seed_phy_test_n = seed_phy_test + n
		for i in ind_n:
			rv_n = np.random.RandomState(seed=seed_phy_test_n)
			signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																 df = rv_n.uniform(low=-df_phy_test, high=df_phy_test),
																 fc = fc_test_orig[i:i+1], 
																 fs = fs)
	dict_wifi['x_test'] = signal_ch.copy()


	#--------------------------------------------------------------------------------------------
	# Physical channel simulation (different days)
	#--------------------------------------------------------------------------------------------

	print('\n---------------------------------------------')
	print('Physical channel simulation (different days)')
	print('---------------------------------------------')
	print('Physical channel types: Train: {}, Test: {}'.format(channel_type_phy_train, channel_type_phy_test))
	print('Physical seeds: Train: {}, Test: {}\n'.format(seed_phy_train, seed_phy_test))

	if phy_method == 0: # Same channel for all packets

		signal_ch = dict_wifi['x_test'].copy()
		for i in tqdm(range(num_test)):
			signal = dict_wifi['x_test'][i][:,0] + 1j*dict_wifi['x_test'][i][:,1]
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

	else: # Different channel for each class, same channel for all packets in a class
		signal_ch = dict_wifi['x_test'].copy()
		for n in trange(num_classes):
			ind_n = np.where(y_test_orig.argmax(axis=1)==n)[0]
			seed_phy_test_n = seed_phy_test + n
			for i in ind_n:
				signal = dict_wifi['x_test'][i][:,0] + 1j*dict_wifi['x_test'][i][:,1]
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


	#--------------------------------------------------------------------------------------------
	# Channel augmentation
	#--------------------------------------------------------------------------------------------

	x_test = dict_wifi['x_test'].copy()
	y_test = dict_wifi['y_test'].copy()

	for channel_type_aug_train in channel_type_aug_trains:

		#--------------------------------------------------------------------------------------------
		# Data formats
		#--------------------------------------------------------------------------------------------

		data_format_offset = 'car-ch-offset-phy-{}-s-{}-aug-{}-df-{}-rand-{}-ty-{}-{}-'.format(df_phy_train*1e6, seed_phy_train, num_df_aug_train, df_aug_train*1e6, rand_aug_train, df_aug_type, keep_orig_train)


		if phy_method==0:
			data_format_ch = 'aug-{}-phy-{}-s-{}-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(num_channel_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, channel_aug_type, num_ch_train, delay_seed_channel_aug_train, snr_train)
		else:
			data_format_ch = 'a{}ug-{}-phy-{}-s-{}-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(phy_method, num_channel_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, channel_aug_type, num_ch_train, delay_seed_channel_aug_train, snr_train)

		data_format = data_format_ch + data_format_offset + 't-' + data_format_plain

		# Checkpoint path
		checkpoint_in = exp_dirs[0] + '/ckpt-' + data_format +'.h5'
		checkpoint_in +=  model_name
		checkpoint_in += '-new.h5'

		print('\n-------------------------------')
		print("Loading model from checkpoint")
		print('Model name: {}'.format(model_name[1:]))
		batch_size = 100

		if checkpoint_in is None:
			raise ValueError('Cannot test without a checkpoint')

		densenet = load_model(checkpoint_in, 
							  custom_objects={'ComplexConv1D':complexnn.ComplexConv1D,
							  				  'GetAbs': utils.GetAbs,
							  				  'Modrelu': Modrelu})

		for channel_type_aug_test in channel_type_aug_tests:

			print('-------------------------------')

			print('\nChannel augmentation:')
			print('\tNo. of augmentations: Train: {}, Test: {}'.format(num_channel_aug_train, num_channel_aug_test))
			print('\tAugmentation channel types: Train: {}, Test: {}'.format(channel_type_aug_train, channel_type_aug_test))
			print('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))

			print('\nOffset augmentation:')
			print('\tNo. of augmentations: Train: {}, Test: {}'.format(num_df_aug_train, num_df_aug_test))
			print('\tAugmentation offsets: Train: {}, {}, Test: {}, {} ppm\n'.format(rand_aug_train, df_aug_train*1e6, rand_aug_test, df_aug_test*1e6))
			print('Keep originals: Train: {}, Test: {}'.format(keep_orig_train, keep_orig_test))

			print('\nNoise augmentation:\n\tSNR: Train: {} dB, Test {} dB\n'.format(snr_train, snr_test))

			seed_aug_offset = np.max(seed_phy_train) + seed_phy_test + num_classes + 1

			x_test_aug = x_test.copy()
			y_test_aug = y_test.copy()
			fc_test_aug = fc_test_orig.copy()

			if num_ch_test < -1:
				raise ValueError('num_ch_test')
			elif num_ch_test!=0:
				for k in tqdm(range(num_channel_aug_test)):
					signal_ch = np.zeros(x_test.shape)
					for i in tqdm(range(num_test)):
						signal = dict_wifi['x_test'][i][:,0]+1j*dict_wifi['x_test'][i][:,1]
						if num_ch_test==-1:
							signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, 
																				seed=seed_aug_offset +num_train*num_channel_aug_train + 1 + (i + k*num_test) % (num_test*num_channel_aug_test), 
																				beta=0, 
																				delay_seed=delay_seed_channel_aug_test,
																				channel_type=channel_type_aug_test,
																				channel_method=channel_method,
																				noise_method=noise_method)
						else:
							signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, 
																				# seed = 1, 
																				seed = seed_aug_offset + num_train*num_channel_aug_train + 1 + (i % num_ch_test) + k * num_ch_test, 
																				beta=0, 
																				delay_seed=delay_seed_channel_aug_test,
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
							fc_test_aug = fc_test_orig.copy()
						else:
							x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
							y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)
							fc_test_aug = np.concatenate((fc_test_aug, fc_test_orig), axis=0)
					else:
						x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
						y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)	
						fc_test_aug = np.concatenate((fc_test_aug, fc_test_orig), axis=0)	


			dict_wifi['x_test'] = x_test_aug.copy()
			dict_wifi['y_test'] = y_test_aug.copy()
			dict_wifi['fc_test'] = fc_test_aug.copy()


			#--------------------------------------------------------------------------------------------
			# Offset augmentation
			#--------------------------------------------------------------------------------------------

			x_test_no_offset = dict_wifi['x_test'].copy()
			y_test_no_offset = dict_wifi['y_test'].copy()
			fc_test_aug = dict_wifi['fc_test'].copy()

			x_test_aug_offset = x_test_no_offset.copy()
			y_test_aug_offset = y_test_no_offset.copy()

			signal_ch = x_test_aug_offset.copy()
			signal_ch = add_freq_offset(signal_ch, rand = rand_aug_test,
												   df = df_aug_test,
												   fc = fc_test_aug, 
												   fs = fs)

			x_test_aug_offset = signal_ch.copy()
			num_test_aug_offset = x_test_aug_offset.shape[0]

			#--------------------------------------------------------------------------------------------
			# Add LLRs
			#--------------------------------------------------------------------------------------------

			# print('Training data size: {}'.format(x_train.shape))
			print('\nTest data size before aug: {}'.format(x_test.shape))
			print('Test data size after aug: {}\n'.format(x_test_aug.shape))

			output_dict = odict(acc=odict(), comp=odict(), loss=odict())

			if num_channel_aug_test!=0:
				num_test_per_aug = num_test_aug_offset // num_channel_aug_test

				logits = densenet.layers[-1].output

				model2 = Model(densenet.input, logits)

				logits_test = model2.predict(x=x_test_aug,
											 batch_size=batch_size,
										 	 verbose=0)		
				logits_test_new = np.zeros((num_test_per_aug, num_classes))
				for i in range(num_channel_aug_test):
					# list_x_test.append(x_test_aug[i*num_test:(i+1)*num_test])

					logits_test_new += logits_test[i*num_test_per_aug:(i+1)*num_test_per_aug]

				# Adding LLRs for num_channel_aug_test test augmentations
				label_pred_llr = logits_test_new.argmax(axis=1)
				label_act = y_test_aug_offset[:num_test_per_aug].argmax(axis=1) 
				ind_correct = np.where(label_pred_llr==label_act)[0] 
				ind_wrong = np.where(label_pred_llr!=label_act)[0] 
				assert (num_test_per_aug == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
				test_acc_llr = 100.*ind_correct.size / num_test_per_aug

				# 1 test augmentation
				probs = densenet.predict(x=x_test_aug_offset[:num_test_per_aug],
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
				print('Test accuracy ({} aug): {:.2f}%'.format(num_channel_aug_test, test_acc_llr))
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

			print('\nPhysical seeds: Train: {}, Test: {}\n'.format(seed_phy_train, seed_phy_test))

			print('\nChannel:')
			print('\tPhysical channel types: Train: {}, Test: {}'.format(channel_type_phy_train, channel_type_phy_test))
			print('\tNo. of augmentations: Train: {}, Test: {}'.format(num_channel_aug_train, num_channel_aug_test))
			print('\tAugmentation channel types: Train: {}, Test: {}'.format(channel_type_aug_train, channel_type_aug_test))
			print('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))

			print('\nOffset:')
			print('\tPhysical offsets: Train: {}, Test: {} ppm'.format(df_phy_train*1e6, df_phy_test*1e6))
			print('\tNo. of augmentations: Train: {}, Test: {}'.format(num_df_aug_train, num_df_aug_test))
			print('\tAugmentation offsets: Train: {}, {}, Test: {}, {} ppm\n'.format(rand_aug_train, df_aug_train*1e6, rand_aug_test, df_aug_test*1e6))
			print('Keep originals: Train: {}, Test: {}'.format(keep_orig_train, keep_orig_test))

			print('\nNoise augmentation:\n\tSNR: Train: {} dB, Test {} dB\n'.format(snr_train, snr_test))
		del densenet



