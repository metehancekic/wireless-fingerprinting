'''
Trains data for a WiFi experiment.
Data is read from npz files.
'''

import numpy as np
import numpy.random as random
from timeit import default_timer as timer

from .preproc.preproc_wifi import rms, basic_equalize_preamble
from .preproc.fading_model  import normalize, add_custom_fading_channel

# from .cxnn.train_network _small import train
# from .cxnn.train_llr  import train
from .cxnn.train_llr  import train_20, train_200

from tqdm import tqdm, trange
import ipdb

exp_dirs = []
exp_dirs += ['/home/rfml/wifi/experiments/exp19']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2']

preprocess_type = 1
# preprocess_type = 2

# sample_rate = 20
sample_rate = 200

sample_duration = 16
# sample_duration = 32

batch_size = 100
epochs = 200

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
# phy_method = 1
phy_method = 2
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

# seed_phy_pairs = [(40, 60)]
# seed_phy_pairs = [(40, 40)]

# seed_phy_pairs = [(80, 100)]
# seed_phy_pairs = [(0, 20), (40, 60), (80, 100), (120, 140), (160, 180)]

#--------------------
# If phy_method = 2
#--------------------
# seed_phy_pairs = [((0, 20), 40)]
seed_phy_pairs = [((20, 40), 60)]
# seed_phy_pairs = [((40, 40), 60)]

#--------------------
# If phy_method = 3
#--------------------
# seed_phy_pairs = [((0, 20, 40), 60)]


#-------------------------------------------------
# Equalization params
#-------------------------------------------------
equalize_train = False
# equalize_train = True

equalize_test = False
# equalize_test = True

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
channel_type_aug_test = 1

# num_aug_trains = [0]
num_aug_trains = [5]
# num_aug_trains = [20]
# num_aug_trains = [100]
# num_aug_trains = [5, 10, 20]
# num_aug_train = 0

# num_aug_test = 0
num_aug_test = 1

'''
aug_type:
	0 - usual channel aug
	1 - same channel for ith example in each class
'''
# aug_type = 0
aug_type = 1


num_ch_train = -1
num_ch_test = -1

# num_ch_train = 1
# num_ch_test = 1

# snr_train = 500
# snr_test = 500

channel_method = 'FFT' 
# channel_method = 'RC' # Raised-cosine

noise_method = 'reg' # Regular
# noise_method = 'bl' # Bandlimited via root raised-cosine

# Random seed for delay perturbation
# Set to False for fixed delays
delay_seed_aug_train = False
delay_seed_aug_test = False
# delay_seed = None


keep_orig_train = False
keep_orig_test = False
# keep_orig_train = True
# keep_orig_test = True

snr_trains = [500]
snr_tests = [500]

# from IPython import embed; embed()
# ipdb.set_trace()

data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

exp_dir = exp_dirs[0]

npz_filename = exp_dir + '/sym-' + data_format + '.npz'

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

num_train = dict_wifi['x_train'].shape[0]
num_test = dict_wifi['x_test'].shape[0]

sampling_rate = sample_rate * 1e+6
fs = sample_rate * 1e+6

x_train_orig = dict_wifi['x_train'].copy()
y_train_orig = dict_wifi['y_train'].copy()

x_test_orig = dict_wifi['x_test'].copy()
y_test_orig = dict_wifi['y_test'].copy()

num_classes = y_train_orig.shape[1]

print('\nPreprocessing\n\tType: {}\n\tFs: {} MHz\n\tLength: {} us'.format(preprocess_type, sample_rate, sample_duration))

for seed_phy_train, seed_phy_test in seed_phy_pairs:

	dict_wifi['x_train'] = x_train_orig.copy()
	dict_wifi['y_train'] = y_train_orig.copy()
	dict_wifi['x_test'] = x_test_orig.copy()
	dict_wifi['y_test'] = y_test_orig.copy()

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
			# seed_phy_train_n_1 = seed_phy_train[0] + n
			# seed_phy_train_n_2 = seed_phy_train[1] + n

			# # Day 1
			# for i in ind_n[:len(ind_n)//2]:
			# 	signal = dict_wifi['x_train'][i][:,0]+1j*dict_wifi['x_train'][i][:,1]
			# 	signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
			# 														seed=seed_phy_train_n_1, 
			# 														beta=0, 
			# 														delay_seed=False, 
			# 														channel_type=channel_type_phy_train,
			# 														channel_method=channel_method,
			# 														noise_method=noise_method)
			# 	signal_faded = normalize(signal_faded)
			# 	signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))

			# # Day 2
			# for i in ind_n[len(ind_n)//2:]:
			# 	signal = dict_wifi['x_train'][i][:,0]+1j*dict_wifi['x_train'][i][:,1]
			# 	signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
			# 														seed=seed_phy_train_n_2, 
			# 														beta=0, 
			# 														delay_seed=False, 
			# 														channel_type=channel_type_phy_train,
			# 														channel_method=channel_method,
			# 														noise_method=noise_method)
			# 	signal_faded = normalize(signal_faded)
			# 	signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))

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

	#--------------------------------------------------------------------------------------------
	# Equalization
	#--------------------------------------------------------------------------------------------
	print('\nEqualization')
	print('\tTrain: {}, Test: {}'.format(equalize_train, equalize_test))

	if equalize_train is True:
		# print('\nEqualizing training preamble')	

		complex_train = dict_wifi['x_train'][..., 0] + 1j* dict_wifi['x_train'][..., 1]

		for i in range(num_train):
			complex_train[i] = basic_equalize_preamble(complex_train[i], 
														   fs=fs, 
														   verbose=verbose_train)

		dict_wifi['x_train'] = np.concatenate((complex_train.real[..., None], complex_train.imag[..., None]), axis=2)

	if equalize_test is True:
		# print('\nEqualizing test preamble')
		complex_test = dict_wifi['x_test'][..., 0] + 1j* dict_wifi['x_test'][..., 1]

		for i in range(num_test):
			complex_test[i] = basic_equalize_preamble(complex_test[i], 
														  fs=fs, 
														  verbose=verbose_test)

		dict_wifi['x_test'] = np.concatenate((complex_test.real[..., None], complex_test.imag[..., None]), axis=2)

	#--------------------------------------------------------------------------------------------
	# Channel augmentation
	#--------------------------------------------------------------------------------------------

	x_train = dict_wifi['x_train'].copy()
	y_train = dict_wifi['y_train'].copy()

	x_test = dict_wifi['x_test'].copy()
	y_test = dict_wifi['y_test'].copy()

	for num_aug_train in num_aug_trains:
		for channel_type_aug_train in channel_type_aug_trains:

			# print('\n-------------------------------')

			print('\nChannel augmentation')
			print('\tAugmentation type: {}'.format(aug_type))
			print('\tNo of augmentations: Train: {}, Test: {}\n\tKeep originals: Train: {}, Test: {}'.format(num_aug_train, num_aug_test, keep_orig_train, keep_orig_test))
			print('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))
			print('\tChannel type: Train: {}, Test: {}\n'.format(channel_type_aug_train, channel_type_aug_test))

			seed_aug_offset = np.max(seed_phy_train) + seed_phy_test + num_classes + 1

			for snr_train in snr_trains:
				for snr_test in snr_tests:
					data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)
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

					if equalize_train is False:
						if phy_method==0:
							data_format = 'aug-{}-phy-{}-s-{}-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(num_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, aug_type, num_ch_train, delay_seed_aug_train, snr_train) + data_format
						else:
							data_format = 'a{}ug-{}-phy-{}-s-{}-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(phy_method, num_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, aug_type, num_ch_train, delay_seed_aug_train, snr_train) + data_format
					else:
						if phy_method==0:
							data_format = 'aug-{}-phy-{}-s-{}-eq-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(num_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, aug_type, num_ch_train, delay_seed_aug_train, snr_train) + data_format
						else:
							data_format = 'a{}ug-{}-phy-{}-s-{}-eq-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(phy_method, num_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, aug_type, num_ch_train, delay_seed_aug_train, snr_train) + data_format

					# Checkpoint path
					checkpoint = exp_dirs[0] + '/ckpt-' + data_format +'.h5'


					print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
					if sample_rate==20:
						train_output, model_name, summary = train_20(dict_wifi, num_aug_test=num_aug_test, 
																				checkpoint_in=None, 
																				checkpoint_out=checkpoint,  
																				batch_size=batch_size, 
																				epochs=epochs)
					elif sample_rate==200:
						train_output, model_name, summary = train_200(dict_wifi, num_aug_test=num_aug_test, 
																				 checkpoint_in=None, 
																				 checkpoint_out=checkpoint,  
																				 batch_size=batch_size, 
																				 epochs=epochs)
					else:
						raise NotImplementedError
					print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

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

						f.write('\nPhysical channel simulation (different days)')
						f.write('\tMethod: {}'.format(phy_method))
						f.write('\tChannel type: Train: {}, Test: {}'.format(channel_type_phy_train, channel_type_phy_test))
						f.write('\tSeed: Train: {}, Test: {}'.format(seed_phy_train, seed_phy_test))

						f.write('\nEqualization')
						f.write('\tTrain: {}, Test: {}'.format(equalize_train, equalize_test))

						f.write('\nChannel augmentation')
						f.write('\tAugmentation type: {}'.format(aug_type))
						f.write('\tNo of augmentations: Train: {}, Test: {}\n\tKeep originals: Train: {}, Test: {}'.format(num_aug_train, num_aug_test, keep_orig_train, keep_orig_test))
						f.write('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))
						f.write('\tChannel type: Train: {}, Test: {}'.format(channel_type_aug_train, channel_type_aug_test))

						for keys, dicts in train_output.items():
							f.write(str(keys)+':\n')
							for key, value in dicts.items():
								f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
						f.write('\n'+str(summary))

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
