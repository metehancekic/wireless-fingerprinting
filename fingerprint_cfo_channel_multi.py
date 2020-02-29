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
mpl.use('Agg')
import matplotlib.pyplot as plt


from .preproc.preproc_wifi import basic_equalize_preamble, offset_compensate_preamble
from .preproc.fading_model  import normalize, add_custom_fading_channel, add_freq_offset
from .cxnn.train_globecom  import train_20, train_200

# from freq_offset import estimate_freq_offset !!


with open('/home/rfml/wifi/scripts/config_cfo_channel_multi.json') as config_file:
    config = json.load(config_file, encoding='utf-8')

num_days = [1,2,3,4,5,6,7,8,9,10,15,20]
# num_days.reverse()


for days in range(8,len(num_days)):
	for experiment_i in range(5):

		seed_days = [0 + experiment_i*400,
	 			  	 [0 + experiment_i*400, 20 + experiment_i*400],
	 			  	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400],
				  	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400],
				  	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400],
	  			  	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400],
				  	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400, 140 + experiment_i*400],
				  	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400, 140 + experiment_i*400, 160 + experiment_i*400],
	             	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400, 140 + experiment_i*400, 160 + experiment_i*400, 180 + experiment_i*400],
				 	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400, 140 + experiment_i*400, 160 + experiment_i*400, 180 + experiment_i*400, 200 + experiment_i*400],
				 	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400, 140 + experiment_i*400, 160 + experiment_i*400, 180 + experiment_i*400, 200 + experiment_i*400, 220 + experiment_i*400, 240 + experiment_i*400, 260 + experiment_i*400, 280 + experiment_i*400, 300 + experiment_i*400],
				 	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400, 140 + experiment_i*400, 160 + experiment_i*400, 180 + experiment_i*400, 200 + experiment_i*400, 220 + experiment_i*400, 240 + experiment_i*400, 260 + experiment_i*400, 280 + experiment_i*400, 300 + experiment_i*400, 320 + experiment_i*400, 340 + experiment_i*400, 360 + experiment_i*400, 380 + experiment_i*400, 400 + experiment_i*400]]

		# seed_days.reverse()



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
		# Physical Channel Parameters
		#-------------------------------------------------
		add_channel = True

		phy_method = num_days[days]
		seed_phy_train = seed_days[days]
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
		remove_cfo = True

		phy_method_cfo = phy_method  # config["phy_method_cfo"]
		df_phy_train = config['df_phy_train']
		df_phy_test = config['df_phy_test']
		seed_phy_train_cfo = seed_phy_train # config['seed_phy_train_cfo']
		seed_phy_test_cfo = seed_phy_test # config['seed_phy_test_cfo']

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
		augment_cfo = False

		df_aug_train = df_phy_train 
		rand_aug_train = config['rand_aug_train']
		num_aug_train_cfo = config['num_aug_train_cfo']
		keep_orig_train_cfo = config['keep_orig_train_cfo']
		aug_type_cfo = config['aug_type_cfo']

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

		if check_signal_power_effect is True:
			
			# randd = np.random.uniform(-1.0,1.0, size = dict_wifi['x_train'].shape[0]) + 1j * np.random.uniform(-1.0,1.0,size=dict_wifi['x_train'].shape[0])
			# randd = randd[:, np.newaxis, np.newaxis]
			# dict_wifi['x_train'] = dict_wifi['x_train'] * randd
			# rand2 = np.random.uniform(-1.0,1.0, size = dict_wifi['x_test'].shape[0]) + 1j * np.random.uniform(-1.0,1.0,size=dict_wifi['x_test'].shape[0])
			# rand2 = rand2[:, np.newaxis, np.newaxis]
			# dict_wifi['x_test'] = dict_wifi['x_test'] * rand2

			dict_wifi['x_train'][:,25:,:] = 0
			dict_wifi['x_test'][:,25:,:] = 0

		if plot_signal == True:

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

		#--------------------------------------------------------------------------------------------
		# Physical channel simulation (different days)
		#--------------------------------------------------------------------------------------------
		if add_channel:
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
					seed_phy_train_n = seed_phy_train + n
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

			data_format = data_format + '-phy-{}-m-{}-s-{}'.format(channel_type_phy_train, phy_method, np.max(seed_phy_train))	

		#--------------------------------------------------------------------------------------------
		# Physical offset simulation (different days)
		#--------------------------------------------------------------------------------------------
		if add_cfo:

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

			if phy_method==1: # Different offset for each class, same offset for all packets in a class
				signal_ch = x_train_orig.copy()
				for n in trange(num_classes):
					ind_n = np.where(y_train_orig.argmax(axis=1)==n)[0]
					seed_phy_train_n = seed_phy_train + n
					for i in ind_n:
						rv_n = np.random.RandomState(seed=seed_phy_train_n)
						signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																			 df = rv_n.uniform(low=-df_phy_train, high=df_phy_train),
																			 fc = fc_train_orig[i:i+1], 
																			 fs = fs)
				dict_wifi['x_train'] = signal_ch.copy()

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

			else:  # Train on multiple days, test on 1 day, with a diff offset for each class
				signal_ch = x_train_orig.copy()
				for n in trange(num_classes):
					ind_n = np.where(y_train_orig.argmax(axis=1)==n)[0]
					
					num_signals_per_day = len(ind_n)//phy_method # per class

					# Day j
					for j in range(phy_method-1):
						seed_phy_train_n_j = seed_phy_train[j] + n

						for i in ind_n[j*num_signals_per_day:(j+1)*num_signals_per_day]:
							rv_n = np.random.RandomState(seed=seed_phy_train_n_j)
							signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																				 df = rv_n.uniform(low=-df_phy_train, high=df_phy_train),
																				 fc = fc_train_orig[i:i+1], 
																				 fs = fs)

					# Last day
					seed_phy_train_n_j = seed_phy_train[phy_method-1] + n
					for i in ind_n[(phy_method-1)*num_signals_per_day:]:
							rv_n = np.random.RandomState(seed=seed_phy_train_n_j)
							signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																				 df = rv_n.uniform(low=-df_phy_train, high=df_phy_train),
																				 fc = fc_train_orig[i:i+1], 
																				 fs = fs)	
				dict_wifi['x_train'] = signal_ch.copy()

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

			del signal_ch, x_test_orig, x_train_orig, y_train_orig, y_test_orig, fc_train_orig, fc_test_orig
			

			data_format = data_format + '_cfo_{}'.format(np.int(df_phy_train*1000000))

		#--------------------------------------------------------------------------------------------
		# Physical offset compensation 
		#--------------------------------------------------------------------------------------------
		if remove_cfo:

			x_train = dict_wifi['x_train'].copy()
			x_test = dict_wifi['x_test'].copy()



			complex_train = x_train[..., 0] + 1j* x_train[..., 1]
			complex_test = x_test[..., 0] + 1j* x_test[..., 1]

			del x_test, x_train

			complex_train_removed_cfo = complex_train.copy()
			complex_test_removed_cfo = complex_test.copy()

			freq_train = np.zeros([num_train, 2])
			freq_test = np.zeros([num_test, 2])
			for i in trange(num_train):
				complex_train_removed_cfo[i], freq_train[i] = offset_compensate_preamble(complex_train[i], fs = fs,verbose=False, option = 2)
			for i in trange(num_test):
				complex_test_removed_cfo[i], freq_test[i] = offset_compensate_preamble(complex_test[i], fs = fs,verbose=False, option = 2)

			dict_wifi['x_train'] = np.concatenate((complex_train_removed_cfo.real[..., None], complex_train_removed_cfo.imag[..., None]), axis= -1)
			dict_wifi['x_test'] = np.concatenate((complex_test_removed_cfo.real[..., None], complex_test_removed_cfo.imag[..., None]), axis= -1)
			
			data_format = data_format + '_comp'

		#--------------------------------------------------------------------------------------------
		# Equalization
		#--------------------------------------------------------------------------------------------
		if equalize_train or equalize_test:
			print('\nEqualization')
			print('\tTrain: {}, Test: {}'.format(equalize_train, equalize_test))

			data_format = data_format + '-eq'

		if equalize_train is True:
			# print('\nEqualizing training preamble')	

			complex_train = dict_wifi['x_train'][..., 0] + 1j* dict_wifi['x_train'][..., 1]

			for i in trange(num_train):
				complex_train[i] = basic_equalize_preamble(complex_train[i], 
															   fs=fs, 
															   verbose=verbose_train)

			dict_wifi['x_train'] = np.concatenate((complex_train.real[..., None], complex_train.imag[..., None]), axis=2)

			del complex_train

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
			
			x_train = dict_wifi['x_train'].copy()
			y_train = dict_wifi['y_train'].copy()

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


			x_train_aug = x_train.copy()
			y_train_aug = y_train.copy()

			channel_dict = {}
			for i in range(401):
				channel_dict[i] = seed_aug_offset

			if num_ch_train < -1:
				raise ValueError('num_ch_train')
			elif num_ch_train != 0:
				for k in tqdm(range(num_aug_train)):
					signal_ch = np.zeros(x_train.shape)
					for i in tqdm(range(num_train)):
						signal = x_train[i][:,0]+1j*x_train[i][:,1]
						if num_ch_train==-1:
							signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, 
																				seed=seed_aug_offset + (i + k*num_train) % (num_train*num_aug_train), 
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
																				seed = seed_aug_offset + k * num_ch_train + (i % num_ch_train), 
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

			# x_test_aug = x_test.copy()
			# y_test_aug = y_test.copy()

			# if num_ch_test < -1:
			# 	raise ValueError('num_ch_test')
			# elif num_ch_test!=0:
			# 	for k in tqdm(range(num_aug_test)):
			# 		signal_ch = np.zeros(x_test.shape)
			# 		for i in tqdm(range(num_test)):
			# 			signal = dict_wifi['x_test'][i][:,0]+1j*dict_wifi['x_test'][i][:,1]
			# 			if num_ch_test==-1:
			# 				signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, 
			# 																	seed=seed_aug_offset + num_train*num_aug_train + 1 + (i + k*num_test) % (num_test*num_aug_test), 
			# 																	beta=0, 
			# 																	delay_seed=delay_seed_aug_test,
			# 																	channel_type=channel_type_aug_test,
			# 																	channel_method=channel_method,
			# 																	noise_method=noise_method)
			# 			else:
			# 				signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, 
			# 																	# seed = 1, 
			# 																	seed = seed_aug_offset + num_train*num_aug_train + 1 + (i % num_ch_test) + k * num_ch_test, 
			# 																	beta=0, 
			# 																	delay_seed=delay_seed_aug_test,
			# 																	channel_type=channel_type_aug_test,
			# 																	channel_method=channel_method,
			# 																	noise_method=noise_method)
						
			# 			signal_faded = normalize(signal_faded)
			# 			signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1)
			# 			# dict_wifi['x_test'][i] = signal_ch
			# 		if keep_orig_test is False:
			# 			if k==0:
			# 				x_test_aug = signal_ch.copy()
			# 				y_test_aug = y_test.copy()
			# 			else:
			# 				x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
			# 				y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)
			# 		else:
			# 			x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
			# 			y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)		

			

			# dict_wifi['x_test'] = x_test_aug.copy()
			# dict_wifi['y_test'] = y_test_aug.copy()

			# del x_test_aug, y_test_aug

			data_format = data_format + 'aug-{}-art-{}-ty-{}-nch-{}-snr-{:.0f}'.format(num_aug_train, channel_type_aug_train, aug_type, num_ch_train, snr_train)



		#--------------------------------------------------------------------------------------------
		# Carrier Frequency Offset augmentation
		#--------------------------------------------------------------------------------------------
		if augment_cfo is True:

			print('\nCFO augmentation')
			print('\tAugmentation type: {}'.format(aug_type_cfo))
			print('\tNo of augmentations: Train: {}, \n\tKeep originals: Train: {}'.format(num_aug_train_cfo, keep_orig_train_cfo))
			
			print('\tCFO aug type: {}\n'.format(aug_type_cfo))

			x_train_aug = dict_wifi['x_train'].copy()
			y_train_aug = dict_wifi['y_train'].copy()

			fc_train_orig = dict_wifi['fc_train']
			fc_test_orig = dict_wifi['fc_test']

			seed_aug_cfo = np.max(seed_phy_train_cfo) + seed_phy_test_cfo + num_classes + 1

			if aug_type_cfo == 0:
				for k in tqdm(range(num_aug_train_cfo)):
					signal_ch = dict_wifi['x_train'].copy()
					# import ipdb; ipdb.set_trace()
					signal_ch = add_freq_offset(signal_ch, rand = rand_aug_train,
														   df = df_aug_train,
														   fc = fc_train_orig, 
														   fs = fs)
					if keep_orig_train is False:
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
																			 fs = fs)
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


			data_format = data_format + 'augcfo-{}-df-{}-rand-{}-ty-{}-{}-t-'.format(num_aug_train, df_aug_train*1e6, rand_aug_train, aug_type, keep_orig_train)

		print(data_format)
		#--------------------------------------------------------------------------------------------
		# Train
		#--------------------------------------------------------------------------------------------

		# Checkpoint path
		checkpoint = str(exp_dir + '/ckpt-' + data_format + '.h5')

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
																 epochs = epochs)

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


		with open(exp_dir + '/logs-' + 'multi_days_cfo_comp_cont'  + '.txt', 'a+') as f:
			f.write('\n\n----------{:}-days------------\n'.format(days)+'\n\n')
			f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')
			f.write('\n\nSeed : {:}\n'.format(seed_days[days])+'\n\n')
			f.write('\n\nExperiment : {:}\n'.format(experiment_i+1)+'\n\n')
			for keys, dicts in train_output.items():
				f.write(str(keys)+':\n')
				for key, value in dicts.items():
					f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
			
