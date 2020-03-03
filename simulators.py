'''
All Simulation Codes needed for CFO and Channel Experiments

Need to add comment to the functions
'''

import numpy as np
from timeit import default_timer as timer
import argparse
from tqdm import trange, tqdm
import json
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


from preproc.preproc_wifi import basic_equalize_preamble, offset_compensate_preamble, get_residuals_preamble
from preproc.fading_model  import normalize, add_custom_fading_channel, add_freq_offset


# from freq_offset import estimate_freq_offset !!

def signal_power_effect(dict_wifi, data_format):
	# randd = np.random.uniform(-1.0,1.0, size = dict_wifi['x_train'].shape[0]) + 1j * np.random.uniform(-1.0,1.0,size=dict_wifi['x_train'].shape[0])
	# randd = randd[:, np.newaxis, np.newaxis]
	# dict_wifi['x_train'] = dict_wifi['x_train'] * randd
	# rand2 = np.random.uniform(-1.0,1.0, size = dict_wifi['x_test'].shape[0]) + 1j * np.random.uniform(-1.0,1.0,size=dict_wifi['x_test'].shape[0])
	# rand2 = rand2[:, np.newaxis, np.newaxis]
	# dict_wifi['x_test'] = dict_wifi['x_test'] * rand2

	dict_wifi['x_train'][:,25:,:] = 0
	dict_wifi['x_test'][:,25:,:] = 0
	data_format = data_format + '_spe_'

	return dict_wifi, data_format

def plot_signals(dict_wifi):

	signals_directory = "signal_images/"
	if not os.path.exists(signals_directory):
		os.makedirs(signals_directory)

	complex_signal = dict_wifi['x_test'][..., 0] + 1j* dict_wifi['x_test'][..., 1]
	for i in range(3):
		indd = np.where(np.argmax(dict_wifi['y_train'],axis=1)==i)
		signal_p = np.abs(complex_signal[indd[0][0],:])
		plt.plot(signal_p)
		plt.grid(True)
		fig_name = os.path.join(signals_directory,'signals_magnitude.pdf')
		plt.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')


	# complex_signal = dict_wifi['x_test'][..., 0]
	# for i in range(3):
	# 	indd = np.where(np.argmax(dict_wifi['y_train'],axis=1)==i)
	# 	signal_p = complex_signal[indd[0][0],:]
	# 	plt.plot(signal_p)
	# 	plt.grid(True)
	# 	fig_name = os.path.join(signals_directory,'signals_real.pdf')
	# 	plt.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')

	# complex_signal = dict_wifi['x_test'][..., 0] + 1j* dict_wifi['x_test'][..., 1]
	# for i in range(2):
	# 	indd = np.where(np.argmax(dict_wifi['y_train'],axis=1)==i)
	# 	for j in range(2):
	# 		signal_p = np.angle(complex_signal[indd[0][j],:])
	# 		plt.plot(signal_p[50:100])
	# 		plt.grid(True)
	# 		fig_name = os.path.join(signals_directory,'signals_phase.pdf')
	# 		plt.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')


def physical_layer_channel(dict_wifi, phy_method, channel_type_phy_train, channel_type_phy_test, channel_method, noise_method, seed_phy_train, seed_phy_test, sampling_rate, data_format):

	num_train = dict_wifi['x_train'].shape[0]
	num_test = dict_wifi['x_test'].shape[0]
	num_classes = dict_wifi['y_train'].shape[1]

	x_train_orig = dict_wifi['x_train'].copy()
	y_train_orig = dict_wifi['y_train'].copy()

	x_test_orig = dict_wifi['x_test'].copy()
	y_test_orig = dict_wifi['y_test'].copy()

	#--------------------------------------------------------------------------------------------
	# Physical channel simulation (different days)
	#--------------------------------------------------------------------------------------------
	
	print('\nPhysical channel simulation (different days)')
	print('\tMethod: {}'.format(phy_method))
	print('\tChannel type: Train: {}, Test: {}'.format(channel_type_phy_train, channel_type_phy_test))
	print('\tSeed: Train: {}, Test: {}\n'.format(seed_phy_train, seed_phy_test))

	if phy_method == 0: # Same channel for all packets
		signal_ch = dict_wifi['x_train'].copy()
		for i in tqdm(range(num_train)):
			signal = dict_wifi['x_train'][i][:,0]+1j*dict_wifi['x_train'][i][:,1]
			signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
																seed=seed_phy_train, 
																beta=0, 
																delay_seed=False, 
																channel_type=channel_type_phy_train,
																channel_method=channel_method,
																noise_method=noise_method)
			signal_faded = normalize(signal_faded)
			signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
		dict_wifi['x_train'] = signal_ch.copy()

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
		signal_ch = dict_wifi['x_train'].copy()
		for n in trange(num_classes):
			ind_n = np.where(y_train_orig.argmax(axis=1)==n)[0]
			seed_phy_train_n = seed_phy_train[0] + n
			# print('{}: {}'.format(n, ind_n))
			for i in ind_n:
				signal = dict_wifi['x_train'][i][:,0]+1j*dict_wifi['x_train'][i][:,1]
				signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
																	seed=seed_phy_train_n, 
																	beta=0, 
																	delay_seed=False, 
																	channel_type=channel_type_phy_train,
																	channel_method=channel_method,
																	noise_method=noise_method)
				signal_faded = normalize(signal_faded)
				# ipdb.set_trace()
				signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
		dict_wifi['x_train'] = signal_ch.copy()

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
		signal_ch = dict_wifi['x_train'].copy()
		for n in trange(num_classes):
			ind_n = np.where(y_train_orig.argmax(axis=1)==n)[0]
			num_signals_per_day = len(ind_n)//phy_method # per class

			# Day j
			for j in range(phy_method-1):

				seed_phy_train_n_j = seed_phy_train[j] + n

				for i in ind_n[j*num_signals_per_day:(j+1)*num_signals_per_day]:
					signal = dict_wifi['x_train'][i][:,0]+1j*dict_wifi['x_train'][i][:,1]
					signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
																		seed=seed_phy_train_n_j, 
																		beta=0, 
																		delay_seed=False, 
																		channel_type=channel_type_phy_train,
																		channel_method=channel_method,
																		noise_method=noise_method)
					signal_faded = normalize(signal_faded)
					signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))

			# Last day
			seed_phy_train_n_j = seed_phy_train[phy_method-1] + n
			for i in ind_n[(phy_method-1)*num_signals_per_day:]:
				signal = dict_wifi['x_train'][i][:,0]+1j*dict_wifi['x_train'][i][:,1]
				signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
																	seed=seed_phy_train_n_j, 
																	beta=0, 
																	delay_seed=False, 
																	channel_type=channel_type_phy_train,
																	channel_method=channel_method,
																	noise_method=noise_method)
				signal_faded = normalize(signal_faded)
				signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
		dict_wifi['x_train'] = signal_ch.copy()

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

	data_format = data_format + '[-phy-{}-m-{}-s-{}]-'.format(channel_type_phy_train, phy_method, np.max(seed_phy_train))

	return dict_wifi, data_format



def physical_layer_cfo(dict_wifi, df_phy_train, df_phy_test, seed_phy_train_cfo, seed_phy_test_cfo, sampling_rate, phy_method_cfo,  data_format):
	
	print('\n---------------------------------------------')
	print('Physical offset simulation (different days)')
	print('---------------------------------------------')
	print('Physical offsets: Train: {}, Test: {} ppm'.format(df_phy_train*1e6, df_phy_test*1e6))
	print('Physical seeds: Train: {}, Test: {}\n'.format(seed_phy_train_cfo, seed_phy_test_cfo))

	x_train_orig = dict_wifi['x_train'].copy()
	y_train_orig = dict_wifi['y_train'].copy()
	num_classes = y_train_orig.shape[1]

	x_test_orig = dict_wifi['x_test'].copy()
	y_test_orig = dict_wifi['y_test'].copy()

	fc_train_orig = dict_wifi['fc_train']
	fc_test_orig = dict_wifi['fc_test']

	if phy_method_cfo==1: # Different offset for each class, same offset for all packets in a class
		signal_ch = x_train_orig.copy()
		for n in trange(num_classes):
			ind_n = np.where(y_train_orig.argmax(axis=1)==n)[0]
			seed_phy_train_n = seed_phy_train_cfo[0] + n
			for i in ind_n:
				rv_n = np.random.RandomState(seed=seed_phy_train_n)
				signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																	 df = rv_n.uniform(low=-df_phy_train, high=df_phy_train),
																	 fc = fc_train_orig[i:i+1], 
																	 fs = sampling_rate)
		dict_wifi['x_train'] = signal_ch.copy()

		signal_ch = x_test_orig.copy()
		for n in trange(num_classes):
			ind_n = np.where(y_test_orig.argmax(axis=1)==n)[0]
			seed_phy_test_n = seed_phy_test_cfo + n
			for i in ind_n:
				rv_n = np.random.RandomState(seed=seed_phy_test_n)
				signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																	 df = rv_n.uniform(low=-df_phy_test, high=df_phy_test),
																	 fc = fc_test_orig[i:i+1], 
																	 fs = sampling_rate)
		dict_wifi['x_test'] = signal_ch.copy()

	else:  # Train on multiple days, test on 1 day, with a diff offset for each class
		signal_ch = x_train_orig.copy()
		for n in trange(num_classes):
			ind_n = np.where(y_train_orig.argmax(axis=1)==n)[0]
			
			num_signals_per_day = len(ind_n)//phy_method_cfo # per class

			# Day j
			for j in range(phy_method_cfo-1):
				seed_phy_train_n_j = seed_phy_train_cfo[j] + n

				for i in ind_n[j*num_signals_per_day:(j+1)*num_signals_per_day]:
					rv_n = np.random.RandomState(seed=seed_phy_train_n_j)
					signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																		 df = rv_n.uniform(low=-df_phy_train, high=df_phy_train),
																		 fc = fc_train_orig[i:i+1], 
																		 fs = sampling_rate)

			# Last day
			seed_phy_train_n_j = seed_phy_train_cfo[phy_method_cfo-1] + n
			for i in ind_n[(phy_method_cfo-1)*num_signals_per_day:]:
					rv_n = np.random.RandomState(seed=seed_phy_train_n_j)
					signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																		 df = rv_n.uniform(low=-df_phy_train, high=df_phy_train),
																		 fc = fc_train_orig[i:i+1], 
																		 fs = sampling_rate)	
		dict_wifi['x_train'] = signal_ch.copy()

		signal_ch = x_test_orig.copy()
		for n in trange(num_classes):
			ind_n = np.where(y_test_orig.argmax(axis=1)==n)[0]
			seed_phy_test_n = seed_phy_test_cfo + n
			for i in ind_n:
				rv_n = np.random.RandomState(seed=seed_phy_test_n)
				signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																	 df = rv_n.uniform(low=-df_phy_test, high=df_phy_test),
																	 fc = fc_test_orig[i:i+1], 
																	 fs = sampling_rate)
		dict_wifi['x_test'] = signal_ch.copy()

	del signal_ch, x_test_orig, x_train_orig, y_train_orig, y_test_orig, fc_train_orig, fc_test_orig
	

	data_format = data_format + '[_cfo_{}-s-{}]-'.format(np.int(df_phy_train*1000000), np.max(seed_phy_train_cfo))

	return dict_wifi, data_format

def cfo_compansator(dict_wifi, sampling_rate, data_format):

	x_train = dict_wifi['x_train'].copy()
	x_test = dict_wifi['x_test'].copy()

	num_train = dict_wifi['x_train'].shape[0]
	num_test = dict_wifi['x_test'].shape[0]
	num_classes = dict_wifi['y_train'].shape[1]

	complex_train = x_train[..., 0] + 1j* x_train[..., 1]
	complex_test = x_test[..., 0] + 1j* x_test[..., 1]

	del x_test, x_train

	complex_train_removed_cfo = complex_train.copy()
	complex_test_removed_cfo = complex_test.copy()

	freq_train = np.zeros([num_train, 2])
	freq_test = np.zeros([num_test, 2])
	for i in trange(num_train):
		complex_train_removed_cfo[i], freq_train[i] = offset_compensate_preamble(complex_train[i], fs = sampling_rate, verbose=False, option = 2)
	for i in trange(num_test):
		complex_test_removed_cfo[i], freq_test[i] = offset_compensate_preamble(complex_test[i], fs = sampling_rate, verbose=False, option = 2)

	dict_wifi['x_train'] = np.concatenate((complex_train_removed_cfo.real[..., None], complex_train_removed_cfo.imag[..., None]), axis= -1)
	dict_wifi['x_test'] = np.concatenate((complex_test_removed_cfo.real[..., None], complex_test_removed_cfo.imag[..., None]), axis= -1)
	
	data_format = data_format + '[_comp]-'

	return dict_wifi, data_format

def equalize_channel(dict_wifi, sampling_rate, data_format, verbosity, which_set = 'x_train'):
	# print('\nEqualizing training preamble')	

	num_set = dict_wifi[which_set].shape[0]

	complex_train = dict_wifi[which_set][..., 0] + 1j* dict_wifi[which_set][..., 1]

	for i in trange(num_set):
		complex_train[i] = basic_equalize_preamble(complex_train[i], 
												   fs = sampling_rate, 
												   verbose = verbosity)

	dict_wifi[which_set] = np.concatenate((complex_train.real[..., None], complex_train.imag[..., None]), axis=2)

	del complex_train

	return dict_wifi, data_format

def get_residual(dict_wifi, sampling_rate, data_format, verbosity, which_set = 'x_train'):
	# print('\nEqualizing training preamble')	

	num_set = dict_wifi[which_set].shape[0]

	complex_train = dict_wifi[which_set][..., 0] + 1j* dict_wifi[which_set][..., 1]

	# assert complex_train.shape[1] == 3200

	for i in trange(num_set):
		complex_train[i], _ = get_residuals_preamble(complex_train[i], 
												   fs = sampling_rate, 
												   verbose = verbosity)

	dict_wifi[which_set] = np.concatenate((complex_train.real[..., None], complex_train.imag[..., None]), axis=2)

	del complex_train

	data_format = data_format + '[_res]-'

	return dict_wifi, data_format

def augment_with_channel(dict_wifi, aug_type, channel_method, num_aug_train, num_aug_test, keep_orig_train, keep_orig_test, num_ch_train, num_ch_test, channel_type_aug_train, channel_type_aug_test, delay_seed_aug_train, snr_train, noise_method, seed_aug, sampling_rate, data_format):

	x_train = dict_wifi['x_train'].copy()
	y_train = dict_wifi['y_train'].copy()

	x_test = dict_wifi['x_test'].copy()
	y_test = dict_wifi['y_test'].copy()

	num_train = dict_wifi['x_train'].shape[0]
	num_test = dict_wifi['x_test'].shape[0]
	num_classes = dict_wifi['y_train'].shape[1]

	# print('\n-------------------------------')

	print('\nChannel augmentation')
	print('\tAugmentation type: {}'.format(aug_type))
	print('\tNo of augmentations: Train: {}, Test: {}\n\tKeep originals: Train: {}, Test: {}'.format(num_aug_train, num_aug_test, keep_orig_train, keep_orig_test))
	print('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))
	print('\tChannel type: Train: {}, Test: {}\n'.format(channel_type_aug_train, channel_type_aug_test))


	print("Seed: Train: {:}".format(seed_aug))


	x_train_aug = x_train.copy()
	y_train_aug = y_train.copy()

	channel_dict = {}
	for i in range(401):
		channel_dict[i] = seed_aug

	if num_ch_train < -1:
		raise ValueError('num_ch_train')
	elif num_ch_train != 0:
		for k in tqdm(range(num_aug_train)):
			signal_ch = np.zeros(x_train.shape)
			for i in tqdm(range(num_train)):
				signal = x_train[i][:,0]+1j*x_train[i][:,1]
				if num_ch_train==-1:
					signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, 
																		seed=seed_aug + (i + k*num_train) % (num_train*num_aug_train), 
																		beta=0, 
																		delay_seed=delay_seed_aug_train, 
																		channel_type=channel_type_aug_train,
																		channel_method=channel_method,
																		noise_method=noise_method)
				elif aug_type==1:
					signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, 
																		seed=channel_dict[np.argmax(y_train[i])],
																		beta=0, 
																		delay_seed=delay_seed_aug_train,
																		channel_type=channel_type_aug_train,
																		channel_method=channel_method,
																		noise_method=noise_method)
					channel_dict[np.argmax(y_train[i])] += 1
				elif aug_type==0:
					signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, 
																		# seed = 0, 
																		seed = seed_aug + k * num_ch_train + (i % num_ch_train), 
																		beta=0, 
																		delay_seed=delay_seed_aug_train,
																		channel_type=channel_type_aug_train,
																		channel_method=channel_method,
																		noise_method=noise_method)

				signal_faded = normalize(signal_faded)
				signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))

			if keep_orig_train is False:
				if k==0:
					x_train_aug = signal_ch.copy()
					y_train_aug = y_train.copy()
				else:
					x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
					y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)
			else:
				x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
				y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)					


	dict_wifi['x_train'] = x_train_aug.copy()
	dict_wifi['y_train'] = y_train_aug.copy()
	dict_wifi['fc_train'] = np.tile(dict_wifi['fc_train'],num_aug_train)

	del x_train, y_train, x_train_aug, y_train_aug


	data_format = data_format + '[aug-{}-art-{}-ty-{}-nch-{}-snr-{:.0f}]-'.format(num_aug_train, channel_type_aug_train, aug_type, num_ch_train, snr_train)

	return dict_wifi, data_format

def augment_with_channel_test(dict_wifi, aug_type, channel_method, num_aug_train, num_aug_test, keep_orig_train, keep_orig_test, num_ch_train, num_ch_test, channel_type_aug_train, channel_type_aug_test, delay_seed_aug_test, snr_test, noise_method, seed_aug, sampling_rate, data_format):

	x_test = dict_wifi['x_test'].copy()
	y_test = dict_wifi['y_test'].copy()

	num_train = dict_wifi['x_train'].shape[0]
	num_test = dict_wifi['x_test'].shape[0]
	num_classes = dict_wifi['y_train'].shape[1]

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
																		seed=seed_aug + num_train*num_aug_train + 1 + (i + k*num_test) % (num_test*num_aug_test), 
																		beta=0, 
																		delay_seed=delay_seed_aug_test,
																		channel_type=channel_type_aug_test,
																		channel_method=channel_method,
																		noise_method=noise_method)
				else:
					signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, 
																		# seed = 1, 
																		seed = seed_aug + num_train*num_aug_train + 1 + (i % num_ch_test) + k * num_ch_test, 
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
	dict_wifi['fc_test'] = np.tile(dict_wifi['fc_test'], num_aug_test)

	del x_test_aug, y_test_aug

	return dict_wifi, data_format

def augment_with_cfo(dict_wifi, aug_type_cfo, df_aug_train, num_aug_train_cfo, keep_orig_train_cfo, rand_aug_train, sampling_rate, seed_aug_cfo, data_format):


	print('\nCFO augmentation')
	print('\tAugmentation type: {}'.format(aug_type_cfo))
	print('\tNo of augmentations: Train: {}, \n\tKeep originals: Train: {}'.format(num_aug_train_cfo, keep_orig_train_cfo))
	

	x_train_aug = dict_wifi['x_train'].copy()
	y_train_aug = dict_wifi['y_train'].copy()

	fc_train_orig = dict_wifi['fc_train']
	fc_test_orig = dict_wifi['fc_test']

	num_train = dict_wifi['x_train'].shape[0]
	num_test = dict_wifi['x_test'].shape[0]
	num_classes = dict_wifi['y_train'].shape[1]


	if aug_type_cfo == 0:
		for k in tqdm(range(num_aug_train_cfo)):
			signal_ch = dict_wifi['x_train'].copy()
			# import ipdb; ipdb.set_trace()
			signal_ch = add_freq_offset(signal_ch, rand = rand_aug_train,
												   df = df_aug_train,
												   fc = fc_train_orig, 
												   fs = sampling_rate)
			if keep_orig_train_cfo is False:
				if k==0:
					x_train_aug = signal_ch.copy()
					y_train_aug = dict_wifi['y_train'].copy()
				else:
					x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
					y_train_aug = np.concatenate((y_train_aug, dict_wifi['y_train']), axis=0)
			else:
				x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
				y_train_aug = np.concatenate((y_train_aug, dict_wifi['y_train']), axis=0)		
	
	elif aug_type_cfo == 1:
		offset_dict = {}
		for i in range(401):
			offset_dict[i] = seed_aug_cfo	
		for k in tqdm(range(num_aug_train_cfo)):
			signal_ch = dict_wifi['x_train'].copy()
			for i in tqdm(range(num_train)):
				rv_n = np.random.RandomState(seed=offset_dict[np.argmax(dict_wifi['y_train'][i])])
				if rand_aug_train=='unif':
					df_n = rv_n.uniform(low=-df_aug_train, high=df_aug_train)
				elif rand_aug_train=='ber':
					df_n = rv_n.choice(a=np.array([-df_aug_train, df_aug_train]))
				signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																	 df = df_n,
																	 fc = fc_train_orig[i:i+1], 
																	 fs = sampling_rate)
				offset_dict[np.argmax(dict_wifi['y_train'][i])] += 1
			if keep_orig_train_cfo is False:
				if k==0:
					x_train_aug = signal_ch.copy()
					y_train_aug = dict_wifi['y_train'].copy()
				else:
					x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
					y_train_aug = np.concatenate((y_train_aug, dict_wifi['y_train']), axis=0)
			else:
				x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
				y_train_aug = np.concatenate((y_train_aug, dict_wifi['y_train']), axis=0)			


	dict_wifi['x_train'] = x_train_aug.copy()
	dict_wifi['y_train'] = y_train_aug.copy()

	del x_train_aug, y_train_aug, fc_train_orig, fc_test_orig


	data_format = data_format + '[augcfo-{}-df-{}-rand-{}-ty-{}-{}-t-]-'.format(num_aug_train_cfo, df_aug_train*1e6, rand_aug_train, aug_type_cfo, keep_orig_train_cfo)

	return dict_wifi, data_format

def augment_with_cfo_test(dict_wifi, aug_type_cfo, df_aug_test, num_aug_test_cfo, keep_orig_test_cfo, rand_aug_test, sampling_rate):
	
	print('\nCFO augmentation')
	print('\tAugmentation type: {}'.format(aug_type_cfo))
	print('\tNo of augmentations: Test: {}, \n\tKeep originals: Test: {}'.format(num_aug_test_cfo, keep_orig_test_cfo))
	
	print('\tCFO aug type: {}\n'.format(aug_type_cfo))

	x_test_aug = dict_wifi['x_test'].copy()
	y_test_aug = dict_wifi['y_test'].copy()

	fc_test_orig = dict_wifi['fc_test']

	# if aug_type_cfo == 0:
	for k in tqdm(range(num_aug_test_cfo)):
		signal_ch = dict_wifi['x_test'].copy()
		# import ipdb; ipdb.set_trace()
		signal_ch = add_freq_offset(signal_ch, rand = False,
											   df = np.random.uniform(-df_aug_test,df_aug_test),
											   fc = fc_test_orig, 
											   fs = sampling_rate)
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
	
	# elif aug_type_cfo == 1:
	# 	offset_dict = {}
	# 	for i in range(401):
	# 		offset_dict[i] = seed_phy_test_cfo+seed_phy_test_cfo+num_classes+1			
	# 	for k in tqdm(range(num_aug_test_cfo)):
	# 		signal_ch = dict_wifi['x_test'].copy()
	# 		for i in tqdm(range(num_test)):
	# 			rv_n = np.random.RandomState(seed=offset_dict[np.argmax(dict_wifi['y_test'][i])])
	# 			if rand_aug_test=='unif':
	# 				df_n = rv_n.uniform(low=-df_aug_test, high=df_aug_test)
	# 			elif rand_aug_test=='ber':
	# 				df_n = rv_n.choice(a=np.array([-df_aug_test, df_aug_test]))
	# 			signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
	# 																 df = df_n,
	# 																 fc = fc_test_orig[i:i+1], 
	# 																 fs = fs)
	# 			offset_dict[np.argmax(dict_wifi['y_test'][i])] += 1
	# 		if keep_orig_test_cfo is False:
	# 			if k==0:
	# 				x_test_aug = signal_ch.copy()
	# 				y_test_aug = dict_wifi['y_test'].copy()
	# 			else:
	# 				x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
	# 				y_test_aug = np.concatenate((y_test_aug, dict_wifi['y_test']), axis=0)
	# 		else:
	# 			x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
	# 			y_test_aug = np.concatenate((y_test_aug, dict_wifi['y_test']), axis=0)			


	dict_wifi['x_test'] = x_test_aug.copy()
	dict_wifi['y_test'] = y_test_aug.copy()

	del x_test_aug, y_test_aug, fc_test_orig

	return dict_wifi


