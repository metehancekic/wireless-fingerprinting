"""
Script to detect novel devices using Mahalanobis distance method (Lee et al, 2018)
"""

import numpy as np
from timeit import default_timer as timer
import argparse
from tqdm import trange, tqdm
import json
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
import ipdb
from num2words import num2words

from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_curve, roc_auc_score

from keras.models import Model, load_model

from train_novel_device import clip_novel_devices, parse_novel_args
from simulators import signal_power_effect, plot_signals, physical_layer_channel, physical_layer_cfo, cfo_compansator, equalize_channel, augment_with_channel, augment_with_cfo
from preproc.fading_model  import normalize, add_custom_fading_channel, add_freq_offset
from preproc.preproc_wifi import basic_equalize_preamble, offset_compensate_preamble
from cxnn.train_globecom import train_200, train_20
from cxnn.complexnn import ComplexConv1D, GetAbs
from cxnn.models_adsb import Modrelu

# mpl.rc('text', usetex=True)

def physical_layer_channel_test(dict_wifi, phy_method, channel_type_phy_train, channel_type_phy_test, channel_method, noise_method, seed_phy_train, seed_phy_test, sampling_rate, data_format):

	num_test = dict_wifi['x_test'].shape[0]
	num_classes = dict_wifi['y_test'].shape[1]

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

	data_format = data_format + '[-phy-{}-m-{}-s-{}]-'.format(channel_type_phy_train, phy_method, np.max(seed_phy_train))

	return dict_wifi, data_format

def physical_layer_cfo_test(dict_wifi, df_phy_train, df_phy_test, seed_phy_train_cfo, seed_phy_test_cfo, sampling_rate, phy_method_cfo,  data_format):
	
	print('\n---------------------------------------------')
	print('Physical offset simulation (different days)')
	print('---------------------------------------------')
	print('Physical offsets: Train: {}, Test: {} ppm'.format(df_phy_train*1e6, df_phy_test*1e6))
	print('Physical seeds: Train: {}, Test: {}\n'.format(seed_phy_train_cfo, seed_phy_test_cfo))

	x_test_orig = dict_wifi['x_test'].copy()
	y_test_orig = dict_wifi['y_test'].copy()
	num_classes = y_test_orig.shape[1]
	fc_test_orig = dict_wifi['fc_test']

	if phy_method_cfo==1: # Different offset for each class, same offset for all packets in a class
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

	del signal_ch, x_test_orig, y_test_orig, fc_test_orig

	data_format = data_format + '[_cfo_{}-s-{}]-'.format(np.int(df_phy_train*1000000), np.max(seed_phy_train_cfo))

	return dict_wifi, data_format

def cfo_compansator_test(dict_wifi, sampling_rate, data_format):

	print('\n---------------------------------------------')
	print('CFO compensation')
	print('---------------------------------------------')

	x_test = dict_wifi['x_test'].copy()

	num_test = dict_wifi['x_test'].shape[0]
	num_classes = dict_wifi['y_test'].shape[1]

	complex_test = x_test[..., 0] + 1j* x_test[..., 1]

	del x_test

	complex_test_removed_cfo = complex_test.copy()

	freq_test = np.zeros([num_test, 2])

	for i in trange(num_test):
		complex_test_removed_cfo[i], freq_test[i] = offset_compensate_preamble(complex_test[i], fs = sampling_rate, verbose=False, option = 2)

	dict_wifi['x_test'] = np.concatenate((complex_test_removed_cfo.real[..., None], complex_test_removed_cfo.imag[..., None]), axis= -1)
	
	data_format = data_format + '[_comp]-'

	return dict_wifi, data_format

def add_comp_aug_cfo_channel_novel(dict_wifi, dict_novel, exp_dir, data_format, experiment_setup, sample_rate, days_multi, exp_i):

	if (experiment_setup['add_cfo'] is True) and (experiment_setup['add_channel'] is False):
		sub_dir = 'cfo'
		exp_dir = os.path.join(exp_dir, sub_dir)
		if not os.path.isdir(exp_dir):
			os.mkdir(exp_dir)

	elif (experiment_setup['add_cfo'] is False) and (experiment_setup['add_channel'] is True):
		sub_dir = 'channel'
		exp_dir = os.path.join(exp_dir, sub_dir)
		if not os.path.isdir(exp_dir):
			os.mkdir(exp_dir)

	elif (experiment_setup['add_cfo'] is True) and (experiment_setup['add_channel'] is True):
		sub_dir = 'cfo_channel'
		exp_dir = os.path.join(exp_dir, sub_dir)
		if not os.path.isdir(exp_dir):
			os.mkdir(exp_dir)
	
	data_format = 'ndays_{}_exp_{}_'.format(days_multi[0], exp_i) + data_format

	max_seed = (max(days_multi)+1) * 20
	seed_test = exp_i * max_seed + 60
	exp_list = [1, 2, 3, 4, 5]
	seeds_train_multi = [[exp_i * max_seed + s*20 if exp_i * max_seed + s*20<seed_test else exp_i * max_seed + (s+1)*20 for s in range(days)] for days in days_multi]
	for i in range(len(seeds_train_multi)):
		assert seed_test not in seeds_train_multi[i]
	num_days = days_multi[0]
	seed_days = seeds_train_multi[0]
	seed_test_day = seed_test

	with open('/home/rfml/wifi/scripts/novel_config_cfo_channel.json') as config_file:
	    config = json.load(config_file, encoding='utf-8')

	sampling_rate = sample_rate * 1e+6
	fs = sample_rate * 1e+6

	#-------------------------------------------------
	# Analysis
	#-------------------------------------------------

	plot_signal = False
	check_signal_power_effect = False

	#-------------------------------------------------
	# Equalization before any preprocessing
	#-------------------------------------------------
	equalize_train_before = experiment_setup['equalize_train_before']
	equalize_test_before = experiment_setup['equalize_test_before']


	#-------------------------------------------------
	# Physical Channel Parameters
	#-------------------------------------------------
	add_channel = experiment_setup['add_channel']

	phy_method = num_days
	seed_phy_train = seed_days
	# seed_phy_test = config['seed_phy_test']
	seed_phy_test = seed_test_day
	channel_type_phy_train = config['channel_type_phy_train']
	channel_type_phy_test = config['channel_type_phy_test']
	phy_noise = config['phy_noise']
	snr_train_phy = config['snr_train_phy']
	snr_test_phy = config['snr_test_phy']

	#-------------------------------------------------
	# Physical CFO parameters
	#-------------------------------------------------

	add_cfo = experiment_setup['add_cfo']
	remove_cfo = experiment_setup['remove_cfo']

	phy_method_cfo = phy_method  # config["phy_method_cfo"]
	df_phy_train = config['df_phy_train']
	df_phy_test = config['df_phy_test']
	seed_phy_train_cfo = seed_phy_train # config['seed_phy_train_cfo']
	seed_phy_test_cfo = seed_phy_test # config['seed_phy_test_cfo']

	#-------------------------------------------------
	# Equalization params
	#-------------------------------------------------
	equalize_train = experiment_setup['equalize_train']
	equalize_test = experiment_setup['equalize_test']
	verbose_train = False
	verbose_test = False

	#-------------------------------------------------
	# Augmentation channel parameters
	#-------------------------------------------------
	augment_channel = experiment_setup['augment_channel']

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
	augment_cfo = experiment_setup['augment_cfo']

	df_aug_train = df_phy_train 
	rand_aug_train = config['rand_aug_train']
	num_aug_train_cfo = config['num_aug_train_cfo']
	keep_orig_train_cfo = config['keep_orig_train_cfo']
	aug_type_cfo = config['aug_type_cfo']

	#---------------------------------------------

	if equalize_train_before or equalize_test_before:
		print('\nEqualization Before')
		print('\tTrain: {}, Test: {}'.format(equalize_train_before, equalize_test_before))

		data_format = data_format + '-eq'

	if equalize_train_before is True:
		pass

	if equalize_test_before is True:
		dict_wifi, data_format = equalize_channel(dict_wifi = dict_wifi, 
												  sampling_rate = sampling_rate, 
												  data_format = data_format, 
												  verbosity = verbose_test, 
												  which_set = 'x_test')

		dict_novel, _ = equalize_channel(dict_wifi = dict_novel, 
										sampling_rate = sampling_rate, 
										data_format = data_format, 
										verbosity = verbose_train, 
										which_set = 'x_test')

	#--------------------------------------------------------------------------------------------
	# Physical channel simulation (different days)
	#--------------------------------------------------------------------------------------------
	if add_channel:
		dict_wifi, data_format = physical_layer_channel_test(dict_wifi = dict_wifi, 
															 phy_method = phy_method, 
															 channel_type_phy_train = channel_type_phy_train, 
															 channel_type_phy_test = channel_type_phy_test, 
															 channel_method = channel_method, 
															 noise_method = noise_method, 
															 seed_phy_train = seed_phy_train, 
															 seed_phy_test = seed_phy_test, 
															 sampling_rate = sampling_rate, 
															 data_format = data_format)

		dict_novel, _ = physical_layer_channel_test(dict_wifi = dict_novel, 
													phy_method = phy_method, 
													channel_type_phy_train = channel_type_phy_train, 
													channel_type_phy_test = channel_type_phy_test, 
													channel_method = channel_method, 
													noise_method = noise_method, 
													seed_phy_train = seed_phy_train, 
													seed_phy_test = seed_phy_test, 
													sampling_rate = sampling_rate, 
													data_format = data_format)


	#--------------------------------------------------------------------------------------------
	# Physical offset simulation (different days)
	#--------------------------------------------------------------------------------------------
	if add_cfo:

		dict_wifi, data_format = physical_layer_cfo_test(dict_wifi = dict_wifi,
														 df_phy_train = df_phy_train,
														 df_phy_test = df_phy_test, 
														 seed_phy_train_cfo = seed_phy_train_cfo, 
														 seed_phy_test_cfo = seed_phy_test_cfo, 
														 sampling_rate = sampling_rate, 
														 phy_method_cfo = phy_method_cfo, 
														 data_format = data_format)

		dict_novel, _ = physical_layer_cfo_test(dict_wifi = dict_novel,
												df_phy_train = df_phy_train,
												df_phy_test = df_phy_test, 
												seed_phy_train_cfo = seed_phy_train_cfo, 
												seed_phy_test_cfo = seed_phy_test_cfo, 
												sampling_rate = sampling_rate, 
												phy_method_cfo = phy_method_cfo, 
												data_format = data_format)

	#--------------------------------------------------------------------------------------------
	# Physical offset compensation 
	#--------------------------------------------------------------------------------------------
	if remove_cfo:
		dict_wifi, data_format = cfo_compansator_test(dict_wifi = dict_wifi, 
													  sampling_rate = sampling_rate, 
													  data_format = data_format)
		
		dict_novel, _ = cfo_compansator_test(dict_wifi = dict_novel, 
											 sampling_rate = sampling_rate, 
											 data_format = data_format)
	#--------------------------------------------------------------------------------------------
	# Equalization
	#--------------------------------------------------------------------------------------------
	if equalize_train or equalize_test:
		print('\nEqualization')
		print('\tTrain: {}, Test: {}'.format(equalize_train, equalize_test))

		data_format = data_format + '-eq'

	if equalize_train is True:
		pass

	if equalize_test is True:
		dict_wifi, data_format = equalize_channel(dict_wifi = dict_wifi, 
												  sampling_rate = sampling_rate, 
												  data_format = data_format, 
												  verbosity = verbose_test, 
												  which_set = 'x_test')

		dict_novel, _ = equalize_channel(dict_wifi = dict_novel, 
										 sampling_rate = sampling_rate, 
										 data_format = data_format, 
										 verbosity = verbose_train, 
										 which_set = 'x_test')

	#--------------------------------------------------------------------------------------------
	# Channel augmentation
	#--------------------------------------------------------------------------------------------
	if augment_channel is True:
		
		# seed_aug = np.max(seed_phy_train) + seed_phy_test + num_classes + 1

		# dict_wifi, data_format = augment_with_channel(dict_wifi = dict_wifi, 
		# 											  aug_type = aug_type, 
		# 											  channel_method = channel_method, 
		# 											  num_aug_train = num_aug_train, 
		# 											  num_aug_test = num_aug_test, 
		# 											  keep_orig_train = keep_orig_train, 
		# 											  keep_orig_test = keep_orig_test, 
		# 											  num_ch_train = num_ch_train, 
		# 											  num_ch_test = num_ch_test, 
		# 											  channel_type_aug_train = channel_type_aug_train, 
		# 											  channel_type_aug_test = channel_type_aug_test, 
		# 											  delay_seed_aug_train = delay_seed_aug_train, 
		# 											  snr_train = snr_train, 
		# 											  noise_method = noise_method, 
		# 											  seed_aug = seed_aug, 
		# 											  sampling_rate = sampling_rate,
		# 											  data_format = data_format)

		data_format = data_format + '[aug-{}-art-{}-ty-{}-nch-{}-snr-{:.0f}]-'.format(num_aug_train, channel_type_aug_train, aug_type, num_ch_train, snr_train)

		
	#--------------------------------------------------------------------------------------------
	# Carrier Frequency Offset augmentation
	#--------------------------------------------------------------------------------------------
	if augment_cfo is True:

		# seed_aug_cfo = np.max(seed_phy_train_cfo) + seed_phy_test_cfo + num_classes + 1

		# dict_wifi, data_format =  augment_with_cfo(dict_wifi = dict_wifi, 
		# 										   aug_type_cfo = aug_type_cfo, 
		# 										   df_aug_train = df_aug_train, 
		# 										   num_aug_train_cfo = num_aug_train_cfo, 
		# 										   keep_orig_train_cfo = keep_orig_train_cfo, 
		# 										   rand_aug_train = rand_aug_train, 
		# 										   sampling_rate = sampling_rate, 
		# 										   seed_aug_cfo = seed_aug_cfo, 
		# 										   data_format = data_format)

		data_format = data_format + '[augcfo-{}-df-{}-rand-{}-ty-{}-{}-t-]-'.format(num_aug_train_cfo, df_aug_train*1e6, rand_aug_train, aug_type_cfo, keep_orig_train_cfo)

	print('\nData format: {}'.format(data_format))
	#----------------------------------

	with open(exp_dir + '/config-' + data_format + '.json', 'w') as f:
	    json.dump(config, f)

	with open(exp_dir + '/setup-' + data_format + '.json', 'w') as f:
	    json.dump(experiment_setup, f)

	return dict_wifi, dict_novel, exp_dir, data_format

if __name__=='__main__':

	args, novel_device_list = parse_novel_args()

	architecture = args.arch
	sample_rate = args.fs
	epochs = args.epochs

	exp_dir = '/home/rfml/wifi/experiments/exp19'
	num_aug_test = 0
	preprocess_type = 1
	sample_duration = 16

	#-------------------------------------------------
	# Loading Data
	#-------------------------------------------------

	data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)
	outfile = exp_dir + '/sym-' + data_format + '.npz'

	np_dict = np.load(outfile)
	dict_wifi, x_novel, y_novel, fc_novel = clip_novel_devices(np_dict, novel_device_list)

	exp_dir = exp_dir + '_novel_{}'.format(len(novel_device_list))
	if not os.path.isdir(exp_dir):
		os.mkdir(exp_dir)
	exp_dir = exp_dir + '/{}'.format('_'.join(str(n) for n in novel_device_list))
	# print(exp_dir)
	# ipdb.set_trace()
	if not os.path.isdir(exp_dir):
		os.mkdir(exp_dir)

	num_classes = dict_wifi['y_train'].shape[1]
	data_format += '_{}'.format(architecture)

	x_train = dict_wifi['x_train']
	y_train = dict_wifi['y_train']
	label_train = y_train.argmax(axis=1)
	n_devices = y_train.shape[1]

	cfo_channel_sim = True

	# ipdb.set_trace()

	if cfo_channel_sim is True:

		experiment_setup = {'equalize_train_before': False,
							'equalize_test_before':  False,

							'add_channel':           True,

							'add_cfo':               True,
							'remove_cfo':            True,

							'equalize_train':        True,
							'equalize_test':         True,

							'augment_channel':       False,

							'augment_cfo':           False}

		exp_i = 0

		# days_multi = [1]
		# days_multi = [3]
		# days_multi = [10]
		days_multi = [20]

		dict_novel = dict(x_test = x_novel.copy(),
						  y_test = y_novel.copy(),
						  fc_test = fc_novel.copy())

		dict_wifi, dict_novel, exp_dir, data_format = add_comp_aug_cfo_channel_novel(dict_wifi, dict_novel, exp_dir, data_format, experiment_setup, sample_rate, days_multi, exp_i)

	x_novel = dict_novel['x_test'].copy()

	x_test = np.concatenate((dict_wifi['x_test'], x_novel))
	y_test = np.concatenate((np.zeros(dict_wifi['x_test'].shape[0]), np.ones(x_novel.shape[0])))

	# Checkpoint path
	# checkpoint = str(exp_dir + '/ckpt-' + data_format + '.h5-new.h5')
	checkpoint = str(exp_dir + '/ckpt-' + data_format + '.h5.h5')

	model = load_model(checkpoint, custom_objects={'ComplexConv1D': ComplexConv1D,
								  				  'GetAbs': GetAbs,
								  				  'Modrelu': Modrelu})
	model.summary()

	inlier_preds = model.predict(dict_wifi['x_test'])
	inlier_acc = 100*np.array(inlier_preds.argmax(axis=1)==dict_wifi['y_test'].argmax(axis=1)).astype(int).mean()
	print('\nInlier accuracy: {:.2f}'.format(inlier_acc))

	# ipdb.set_trace()

	print('\nNovel devices: {}\n'.format(novel_device_list))

	# layers = ['Input', 'Abs', 'Shared_Dense1', 'Avg', 'Dense3']
	layers = ['Abs', 'Shared_Dense1', 'Avg', 'Dense3']
	means = []
	precs = []
	print('-----------\nLayer, AUC\n-----------')
	for i, layer in enumerate(layers):
		if layer=='Input':
			feats_train = np.abs(x_train.dot(np.array([1, 1j])))
			feats_test = np.abs(x_test.dot(np.array([1, 1j])))
		else:
			intermed_model = Model(model.input, model.get_layer(layer).output)
			feats_train = intermed_model.predict(x_train)
			feats_test = intermed_model.predict(x_test)
			if feats_train.ndim > 2:
				feats_train = feats_train.mean(axis=1)
				feats_test = feats_test.mean(axis=1)

		cov_ests = []
		for n in range(n_devices):
			cov_est = EmpiricalCovariance(assume_centered=False)
			cov_est.fit(feats_train[label_train==n])
			cov_ests.append(cov_est)

		scores_train = []
		scores_test = []
		for cov_est in cov_ests:
			scores_train.append(cov_est.mahalanobis(feats_train))
			scores_test.append(cov_est.mahalanobis(feats_test))

		score_train = np.array(scores_train).min(axis=0)
		score_test = np.array(scores_test).min(axis=0)

		auc = roc_auc_score(y_test, score_test)
		fpr, tpr, _ = roc_curve(y_test, score_test)
		print('{}, {:.2f}'.format(layer, 100*auc))

		plt.figure(1, figsize=(14, 5))
		plt.subplot(1, len(layers), i+1)
		plt.hist(score_test[y_test==0], bins=50, color='navy', alpha=0.5, label='Inliers')
		plt.hist(score_test[y_test==1], bins=50, color='red', alpha=0.5, label='Outliers')
		plt.legend(loc='upper right')
		plt.title(r'Layer: {}'.format(layer), fontsize=10)

		plt.figure(2, figsize=(14, 5))
		plt.subplot(1, len(layers), i+1)
		plt.plot(fpr, tpr, lw=2, color='navy', label='Area: {:.1f}%'.format(100*auc))
		plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
		plt.legend(loc='lower right')
		plt.xlabel(r'$P_{FA}$', fontsize=12)
		plt.ylabel(r'$P_{D}$', fontsize=12)
		plt.title(r'Layer: {}'.format(layer), fontsize=10)

	plt.figure(1)
	plt.suptitle('Mahalanobis scores')
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.savefig(exp_dir + '/mah-'+data_format+'.pdf', format='pdf', dpi=1000, bbox_inches='tight')

	plt.figure(2)
	plt.suptitle('ROC curves')
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.savefig(exp_dir + '/roc-'+data_format+'.pdf', format='pdf', dpi=1000, bbox_inches='tight')
	# plt.show()

	# ipdb.set_trace()
