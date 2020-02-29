'''
Real life channel and CFO experiments are done in this code.


 - Physical Layer Channel Simulation
 - Physical Layer CFO Simulation
 - Channel Equalization
 - CFO Compensation
 - Channel Augmentation
 - CFO Augmentation
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


from .preproc.preproc_wifi import basic_equalize_preamble, offset_compensate_preamble
from .preproc.fading_model  import normalize, add_custom_fading_channel, add_freq_offset
from .cxnn.train_globecom  import train_20, train_200

# from freq_offset import estimate_freq_offset !!

from .simulators import signal_power_effect, plot_signals, physical_layer_channel, physical_layer_cfo, cfo_compansator, equalize_channel, augment_with_channel, augment_with_cfo, get_residual

def multiple_day_fingerprint(architecture, config, num_days, seed_days, seed_test_day, experiment_setup, n_val=True):


	# print(architecture)

	#-------------------------------------------------
	# Analysis
	#-------------------------------------------------

	plot_signal = False
	check_signal_power_effect = False

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

	#-------------------------------------------------
	# Residuals
	#-------------------------------------------------

	obtain_residuals = experiment_setup['obtain_residuals']

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


	if check_signal_power_effect == True:
		dict_wifi, data_format = signal_power_effect(dict_wifi = dict_wifi, 
													 data_format = data_format)

	if plot_signal == True:
		plot_signals(dict_wifi = dict_wifi)

	if equalize_train_before or equalize_test_before:
		print('\nEqualization Before')
		print('\tTrain: {}, Test: {}'.format(equalize_train_before, equalize_test_before))

		data_format = data_format + '-eq'

	if equalize_train_before is True:
		dict_wifi, data_format = equalize_channel(dict_wifi = dict_wifi, 
												  sampling_rate = sampling_rate, 
												  data_format = data_format, 
												  verbosity = verbose_train, 
												  which_set = 'x_train')

	if equalize_test_before is True:
		dict_wifi, data_format = equalize_channel(dict_wifi = dict_wifi, 
												  sampling_rate = sampling_rate, 
												  data_format = data_format, 
												  verbosity = verbose_test, 
												  which_set = 'x_test')



	#--------------------------------------------------------------------------------------------
	# Physical channel simulation (different days)
	#--------------------------------------------------------------------------------------------
	if add_channel:
		dict_wifi, data_format = physical_layer_channel(dict_wifi = dict_wifi, 
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

		dict_wifi, data_format = physical_layer_cfo(dict_wifi = dict_wifi,
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
		dict_wifi, data_format = cfo_compansator(dict_wifi = dict_wifi, 
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
		dict_wifi, data_format = equalize_channel(dict_wifi = dict_wifi, 
												  sampling_rate = sampling_rate, 
												  data_format = data_format, 
												  verbosity = verbose_train, 
												  which_set = 'x_train')

	if equalize_test is True:
		dict_wifi, data_format = equalize_channel(dict_wifi = dict_wifi, 
												  sampling_rate = sampling_rate, 
												  data_format = data_format, 
												  verbosity = verbose_test, 
												  which_set = 'x_test')

	#--------------------------------------------------------------------------------------------
	# Channel augmentation
	#--------------------------------------------------------------------------------------------
	if augment_channel is True:
		
		seed_aug = np.max(seed_phy_train) + seed_phy_test + num_classes + 1

		dict_wifi, data_format = augment_with_channel(dict_wifi = dict_wifi, 
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
													  delay_seed_aug_train = delay_seed_aug_train, 
													  snr_train = snr_train, 
													  noise_method = noise_method, 
													  seed_aug = seed_aug, 
													  sampling_rate = sampling_rate,
													  data_format = data_format)

		
	#--------------------------------------------------------------------------------------------
	# Carrier Frequency Offset augmentation
	#--------------------------------------------------------------------------------------------
	if augment_cfo is True:

		seed_aug_cfo = np.max(seed_phy_train_cfo) + seed_phy_test_cfo + num_classes + 1

		dict_wifi, data_format =  augment_with_cfo(dict_wifi = dict_wifi, 
												   aug_type_cfo = aug_type_cfo, 
												   df_aug_train = df_aug_train, 
												   num_aug_train_cfo = num_aug_train_cfo, 
												   keep_orig_train_cfo = keep_orig_train_cfo, 
												   rand_aug_train = rand_aug_train, 
												   sampling_rate = sampling_rate, 
												   seed_aug_cfo = seed_aug_cfo, 
												   data_format = data_format)


	if obtain_residuals is True:
		print('Residuals are being obtained.')
		
		dict_wifi, data_format = get_residual(dict_wifi = dict_wifi, 
											  sampling_rate = sampling_rate, 
											  data_format = data_format, 
											  verbosity = verbose_train, 
											  which_set = 'x_train')

		dict_wifi, _ = get_residual(dict_wifi = dict_wifi, 
											  sampling_rate = sampling_rate, 
											  data_format = data_format, 
											  verbosity = verbose_test, 
											  which_set = 'x_test')


								
	

	print(data_format)
	#--------------------------------------------------------------------------------------------
	# Train
	#--------------------------------------------------------------------------------------------

	# Checkpoint path
	exp_dir += "/CFO_channel_experiments"
	checkpoint = str(exp_dir + '/ckpt-' + data_format)

	if augment_channel is False:
		num_aug_test = 0

	print(checkpoint)
	print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
	if sample_rate==20:
		train_output, model_name, summary = train_20(dict_wifi, checkpoint_in=None,
														     num_aug_test = num_aug_test, 
															 checkpoint_out=checkpoint,
															 architecture=architecture,
															 epochs = epochs)
	elif sample_rate==200:
		train_output, model_name, summary = train_200(dict_wifi, checkpoint_in=None,
														     num_aug_test = num_aug_test, 
															 checkpoint_out=checkpoint,
															 architecture=architecture,
															 epochs = epochs,
															 n_val=n_val)

	else:
		raise NotImplementedError
	print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

	#--------------------------------------------------------------------------------------------
	# Write in log file
	#--------------------------------------------------------------------------------------------

	# Write logs
	with open(exp_dir + '/logs-' + data_format  + '.txt', 'a+') as f:
		f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')

		# f.write('Different day scenario\n')
		# if equalize_train is True:
		# 	f.write('With preamble equalization\n\n')
		# f.write('Channel augmentations: {}, keep_orig: {} \n'.format(num_aug_train, keep_orig_train))
		# f.write('Channel type: Phy_train: {}, Phy_test: {}, Aug_Train: {}, Aug_Test: {} \n'.format(channel_type_phy_train, channel_type_phy_test, channel_type_aug_train, channel_type_aug_test))
		# f.write('Seed: Phy_train: {}, Phy_test: {}'.format(seed_phy_train, seed_phy_test))
		# f.write('No of channels: Train: {}, Test: {} \n'.format(num_ch_train, num_ch_test))
		# f.write('SNR: Train: {} dB, Test {} dB\n'.format(snr_train, snr_test))

		f.write('\nPreprocessing\n\tType: {}\n\tFs: {} MHz\n\tLength: {} us'.format(preprocess_type, sample_rate, sample_duration))

		if equalize_train_before == True:
			f.write('\nEqualized signals before any preprocessing')

		if add_channel is True:
			f.write('\nPhysical Channel is added!')
			f.write('\nPhysical channel simulation (different days)')
			f.write('\tMethod: {}'.format(phy_method))
			f.write('\tChannel type: Train: {}, Test: {}'.format(channel_type_phy_train, channel_type_phy_test))
			f.write('\tSeed: Train: {}, Test: {}'.format(seed_phy_train, seed_phy_test))
		else:
			f.write('\nPhysical Channel is not added!')
		
		if add_cfo is True:
			f.write('\nPhysical CFO is added!')
			f.write('\nPhysical CFO simulation (different days)')
			f.write('\tMethod: {}'.format(phy_method_cfo))
			f.write('\tdf_train: {}, df_test: {}'.format(df_phy_train, df_phy_test))
			f.write('\tSeed: Train: {}, Test: {}'.format(seed_phy_train_cfo, seed_phy_test_cfo))
		else:
			f.write('\nPhysical CFO is not added!')
		
		if remove_cfo is True:
			f.write('\nCFO is compensated!')
		else:
			f.write('\nCFO is not compensated!')

		f.write('\nEqualization')
		f.write('\tTrain: {}, Test: {}'.format(equalize_train, equalize_test))

		if augment_channel is True:
			f.write('\nChannel augmentation')
			f.write('\tAugmentation type: {}'.format(aug_type))
			f.write('\tChannel Method: {}'.format(channel_method))
			f.write('\tNoise Method: {}'.format(noise_method))
			f.write('\tNo of augmentations: Train: {}, Test: {}\n\tKeep originals: Train: {}, Test: {}'.format(num_aug_train, num_aug_test, keep_orig_train, keep_orig_test))
			f.write('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))
			f.write('\tChannel type: Train: {}, Test: {}'.format(channel_type_aug_train, channel_type_aug_test))
			f.write('\tSNR: Train: {}, Test: {}'.format(snr_train, snr_test))
			f.write('\tBeta {}'.format(beta))
		else:
			f.write('\nChannel is not augmented')

		if augment_cfo is True:
			f.write('\nCFO augmentation')
			f.write('\tAugmentation type: {}'.format(aug_type_cfo))
			f.write('\tNo of augmentations: Train: {}, \n\tKeep originals: Train: {}'.format(num_aug_train_cfo, keep_orig_train))
		else:
			f.write('\nCFO is not augmented')



		for keys, dicts in train_output.items():
			f.write(str(keys)+':\n')
			for key, value in dicts.items():
				f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
		f.write('\n'+str(summary))

	print('\nPreprocessing\n\tType: {}\n\tFs: {} MHz\n\tLength: {} us'.format(preprocess_type, sample_rate, sample_duration))

	if equalize_train_before == True:
		print('\nEqualized signals before any preprocessing')

	if add_channel is True:
		print('\nPhysical Channel is added!')
		print('\nPhysical channel simulation (different days)')
		print('\tMethod: {}'.format(phy_method))
		print('\tChannel type: Train: {}, Test: {}'.format(channel_type_phy_train, channel_type_phy_test))
		print('\tSeed: Train: {}, Test: {}'.format(seed_phy_train, seed_phy_test))
	else:
		print('\nPhysical Channel is not added!')

	if add_cfo is True:
		print('\nPhysical CFO is added!')
		print('\nPhysical CFO simulation (different days)')
		print('\tMethod: {}'.format(phy_method_cfo))
		print('\tdf_train: {}, df_test: {}'.format(df_phy_train, df_phy_test))
		print('\tSeed: Train: {}, Test: {}'.format(seed_phy_train_cfo, seed_phy_test_cfo))
	else:
		print('\nPhysical CFO is not added!')

	if remove_cfo is True:
		print('\nCFO is compensated!')
	else:
		print('\nCFO is not compensated!')

	print('\nEqualization')
	print('\tTrain: {}, Test: {}'.format(equalize_train, equalize_test))

	if augment_channel is True:
		print('\nChannel augmentation')
		print('\tAugmentation type: {}'.format(aug_type))
		print('\tChannel Method: {}'.format(channel_method))
		print('\tNoise Method: {}'.format(noise_method))
		print('\tNo of augmentations: Train: {}, Test: {}\n\tKeep originals: Train: {}, Test: {}'.format(num_aug_train, num_aug_test, keep_orig_train, keep_orig_test))
		print('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))
		print('\tChannel type: Train: {}, Test: {}'.format(channel_type_aug_train, channel_type_aug_test))
		print('\tSNR: Train: {}, Test: {}'.format(snr_train, snr_test))
		print('\tBeta {}'.format(beta))
	else:
		print('\nChannel is not augmented')

	if augment_cfo is True:
		print('\nCFO augmentation')
		print('\tAugmentation type: {}'.format(aug_type_cfo))
		print('\tNo of augmentations: Train: {}, \n\tKeep originals: Train: {}'.format(num_aug_train_cfo, keep_orig_train))
	else:
		print('\nCFO is not augmented')

	return train_output


if __name__=='__main__':

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

	experiment_setup = {'equalize_train_before': False,
						'equalize_test_before':  False,

						'add_channel':           True,

						'add_cfo':               False,
						'remove_cfo':            False,

						'equalize_train':        False,
						'equalize_test':         False,

						'augment_channel':       False,

						'augment_cfo':           False,
						'obtain_residuals':      True}

	with open('/home/rfml/wifi/scripts/config_cfo_channel.json') as config_file:
	    config = json.load(config_file, encoding='utf-8')

	num_experiments = 5
	for exp_i in range(num_experiments):
		days_multi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
		max_seed = (max(days_multi)+1) * 20
		seed_test = exp_i * max_seed + 60
		exp_list = [1, 2, 3, 4, 5]
		seeds_train_multi = [[exp_i * max_seed + s*20 if exp_i * max_seed + s*20<seed_test else exp_i * max_seed + (s+1)*20 for s in range(days)] for days in days_multi]
		for i in range(len(seeds_train_multi)):
			assert seed_test not in seeds_train_multi[i]

		

		for indexx, day_count in enumerate(days_multi):
			_ = multiple_day_fingerprint(architecture, config, num_days = day_count, seed_days = seeds_train_multi[indexx], seed_test_day = seed_test, experiment_setup = experiment_setup)

			with open(config['exp_dir'] + "/CFO_channel_experiments/" + log_name + '.txt', 'a+') as f:

				f.write('Number of training days: {:}\n'.format(day_count))
				f.write('\tExperiment: {:}\n'.format(exp_i + 1))
				f.write('\tSeed train: {:}\n'.format(seeds_train_multi[indexx]))
				f.write('\tSeed test: {:}\n'.format(seed_test))

				for keys, dicts in train_output.items():
					f.write(str(keys)+':\n')
					for key, value in dicts.items():
						f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')

				if day_count == days_multi[-1]:
					f.write("#------------------------------------------------------------------------------------------#")


