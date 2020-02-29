'''
Trains data for a WiFi experiment with carrier freq offset augmentation.

Data is read from npz files.
'''
import numpy as np
import numpy.random as random
from timeit import default_timer as timer

from .preproc.preproc_wifi import rms, get_sliding_window, basic_equalize_preamble
from .preproc.fading_model  import normalize, add_custom_fading_channel, add_freq_offset

# from .cxnn.train_network _small import train
# from .cxnn.train_network _aug import train
from .cxnn.train_llr _str import train as train_llr_stride
from .cxnn.train_llr  import train_200 as train_llr_plain
# from .cxnn.train_llr  import train as train_llr_plain

from tqdm import tqdm, trange
import ipdb


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

epochs = 200
# epochs = 100

#-------------------------------------------------
# Use sliding window
#-------------------------------------------------

use_sliding_window = False
# use_sliding_window = True
window_size = 10 	# in symbols
stride = 1 			# in symbols

#-------------------------------------------------
# Physical offset params
#-------------------------------------------------
df_phy_train = 40e-6
df_phy_test = 40e-6

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
	2 - train on 2 days, test on 1 day
	3 - train on 3 days, test on 1 day
'''
# phy_method = 0
phy_method = 1
# phy_method = 2
# phy_method = 3
# phy_method = 4

#--------------------
# If phy_method = 0
#--------------------
# seed_phy_pairs = [(30, 31)]
# seed_phy_pairs = [(40, 41)]
# seed_phy_pairs = [(0, 1), (10, 11), (20, 21), (30, 31), (40, 41)]

#--------------------
# If phy_method = 1
#--------------------
seed_phy_pairs = [(40, 40)] # same day
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
# seed_phy_pairs = [((0, 20, 40, 80), 60)]

#--------------------
# If phy_method = 10
#--------------------
# seed_phy_pairs = [((0, 20, 40, 80, 100, 120, 140, 160, 180, 200), 60)]

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

num_channel_aug_trains = [0]
# num_channel_aug_trains = [5]
# num_channel_aug_trains = [10]
# num_channel_aug_trains = [20]
# num_channel_aug_trains = [20, 100]
# num_channel_aug_trains = [50]
# num_channel_aug_trains = [100]
# num_channel_aug_trains = [5, 10, 20]
# num_channel_aug_train = 0

num_channel_aug_test = 0
# num_channel_aug_test = 20

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
channel_aug_type = 0
# channel_aug_type = 1

keep_orig_train = False
keep_orig_test = False
# keep_orig_train = True
# keep_orig_test = True

num_ch_train = -1
num_ch_test = -1

# snr_train = 500
# snr_test = 500

snr_trains = [500]
snr_tests = [500]

#-------------------------------------------------
# Augmentation offset params
#-------------------------------------------------
df_aug_train = df_phy_train 
# df_aug_train = 200e-6 

rand_aug_train = 'unif'
# rand_aug_train = 'ber'
# rand_aug_train = 'False'

num_df_aug_train = 0
# num_df_aug_train = 1
# num_df_aug_train = 5
# num_df_aug_train = 20

keep_orig_train = False
# keep_orig_train = True

'''
df_aug_type:
	0 - usual offset aug
	1 - same offset for ith example in each class
'''
# df_aug_type = 0
df_aug_type = 1 



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

print('\nPreprocessing\n\tType: {}\n\tFs: {} MHz\n\tLength: {} us'.format(preprocess_type, sample_rate, sample_duration))

for seed_phy_train, seed_phy_test in seed_phy_pairs:

	#--------------------------------------------------------------------------------------------
	# Physical offset simulation (different days)
	#--------------------------------------------------------------------------------------------

	print('\nPhysical carrier offset simulation (different days)')
	print('\tMethod: {}'.format(phy_method))
	print('\tOffsets: Train: {}, Test: {} ppm'.format(df_phy_train*1e6, df_phy_test*1e6))
	print('\tSeed: Train: {}, Test: {}\n'.format(seed_phy_train, seed_phy_test))

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
			
			# seed_phy_train_n_1 = seed_phy_train[0] + n
			# seed_phy_train_n_2 = seed_phy_train[1] + n
			# for i in ind_n[:len(ind_n)//2]:
			# 	rv_n = np.random.RandomState(seed=seed_phy_train_n_1)
			# 	signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
			# 														 df = rv_n.uniform(low=-df_phy_train, high=df_phy_train),
			# 														 fc = fc_train_orig[i:i+1], 
			# 														 fs = fs)
			# for i in ind_n[len(ind_n)//2:]:
			# 	rv_n = np.random.RandomState(seed=seed_phy_train_n_2)
			# 	signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
			# 														 df = rv_n.uniform(low=-df_phy_train, high=df_phy_train),
			# 														 fc = fc_train_orig[i:i+1], 
			# 														 fs = fs)	

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
			signal = dict_wifi['x_train'][i][:,0] + 1j*dict_wifi['x_train'][i][:,1]
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

	elif phy_method==1: # Different channel for each class, same channel for all packets in a class
		signal_ch = dict_wifi['x_train'].copy()
		for n in trange(num_classes):
			ind_n = np.where(y_train_orig.argmax(axis=1)==n)[0]
			seed_phy_train_n = seed_phy_train + n
			# print('{}: {}'.format(n, ind_n))
			for i in ind_n:
				signal = dict_wifi['x_train'][i][:,0] + 1j*dict_wifi['x_train'][i][:,1]
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

	else: # Train on multiple days, test on 1 day, with a diff channel for each class
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
		complex_train = dict_wifi['x_train'][..., 0] + 1j* dict_wifi['x_train'][..., 1]

		for i in range(num_train):
			complex_train[i] = basic_equalize_preamble(complex_train[i], 
														   fs=fs, 
														   verbose=verbose_train)

		dict_wifi['x_train'] = np.concatenate((complex_train.real[..., None], complex_train.imag[..., None]), axis=2)

	if equalize_test is True:
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

	for num_channel_aug_train in num_channel_aug_trains:
		for channel_type_aug_train in channel_type_aug_trains:

			print('-------------------------------')

			if use_sliding_window is True:
				print('\nSliding window augmentation\n\tWindow size: {}\n\tStride: {}'.format(window_size, stride))

			print('\nChannel augmentation')
			print('\tAugmentation type: {}'.format(channel_aug_type))
			print('\tNo. of augmentations: Train: {}, Test: {}'.format(num_channel_aug_train, num_channel_aug_test))
			print('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))
			print('\tChannel types: Train: {}, Test: {}'.format(channel_type_aug_train, channel_type_aug_test))

			print('\nCarrier offset augmentation')
			print('\tAugmentation type: {}'.format(df_aug_type))
			print('\tNo. of augmentations: Train: {}'.format(num_df_aug_train))
			print('\tAugmentation offsets: Train: {}, {}ppm\n'.format(rand_aug_train, df_aug_train*1e6))
			print('Keep originals: Train: {}, Test: {}'.format(keep_orig_train, keep_orig_test))

			seed_aug_offset = np.max(seed_phy_train) + seed_phy_test + num_classes + 1

			for snr_train in snr_trains:
				for snr_test in snr_tests:

					print('\nNoise augmentation:\n\tSNR: Train: {} dB, Test {} dB\n'.format(snr_train, snr_test))

					x_train_aug = x_train.copy()
					y_train_aug = y_train.copy()
					fc_train_aug = fc_train_orig.copy()

					channel_dict = {}
					for i in range(401):
						channel_dict[i] = seed_aug_offset

					if num_ch_train < -1:
						raise ValueError('num_ch_train')
					elif num_ch_train != 0:
						for k in tqdm(range(num_channel_aug_train)):
							signal_ch = np.zeros(x_train.shape)
							for i in tqdm(range(num_train)):
								signal = x_train[i][:,0]+1j*x_train[i][:,1]
								if num_ch_train==-1:
									signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, 
																						seed=seed_aug_offset + (i + k*num_train) % (num_train*num_channel_aug_train), 
																						beta=0, 
																						delay_seed=delay_seed_channel_aug_train, 
																						channel_type=channel_type_aug_train,
																						channel_method=channel_method,
																						noise_method=noise_method)
								elif channel_aug_type==1:
									signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, 
																						seed=channel_dict[np.argmax(y_train[i])],
																						beta=0, 
																						delay_seed=delay_seed_channel_aug_train,
																						channel_type=channel_type_aug_train,
																						channel_method=channel_method,
																						noise_method=noise_method)
									channel_dict[np.argmax(y_train[i])] += 1
								elif channel_aug_type==0:
									signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, 
																						# seed = 0, 
																						seed = seed_aug_offset + k * num_ch_train + (i % num_ch_train), 
																						beta=0, 
																						delay_seed=delay_seed_channel_aug_train,
																						channel_type=channel_type_aug_train,
																						channel_method=channel_method,
																						noise_method=noise_method)

								signal_faded = normalize(signal_faded)
								signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))

							if keep_orig_train is False:
								if k==0:
									x_train_aug = signal_ch.copy()
									y_train_aug = y_train.copy()
									fc_train_aug = fc_train_orig.copy()
								else:
									x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
									y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)
									fc_train_aug = np.concatenate((fc_train_aug, fc_train_orig), axis=0)
							else:
								x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
								y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)
								fc_train_aug = np.concatenate((fc_train_aug, fc_train_orig), axis=0)					


					dict_wifi['x_train'] = x_train_aug.copy()
					dict_wifi['y_train'] = y_train_aug.copy()
					dict_wifi['fc_train'] = fc_train_aug.copy()

					x_test_aug = x_test.copy()
					y_test_aug = y_test.copy()

					if num_ch_test < -1:
						raise ValueError('num_ch_test')
					elif num_ch_test!=0:
						for k in tqdm(range(num_channel_aug_test)):
							signal_ch = np.zeros(x_test.shape)
							for i in tqdm(range(num_test)):
								signal = dict_wifi['x_test'][i][:,0]+1j*dict_wifi['x_test'][i][:,1]
								if num_ch_test==-1:
									signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, 
																						seed=seed_aug_offset + num_train*num_channel_aug_train + 1 + (i + k*num_test) % (num_test*num_channel_aug_test), 
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
								else:
									x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
									y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)
							else:
								x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
								y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)		


					dict_wifi['x_test'] = x_test_aug.copy()
					dict_wifi['y_test'] = y_test_aug.copy()


					#--------------------------------------------------------------------------------------------
					# Offset augmentation
					#--------------------------------------------------------------------------------------------

					x_train_no_offset = dict_wifi['x_train'].copy()
					y_train_no_offset = dict_wifi['y_train'].copy()
					fc_train_aug = dict_wifi['fc_train'].copy()

					x_train_aug_offset = x_train_no_offset.copy()
					y_train_aug_offset = y_train_no_offset.copy()

					if df_aug_type == 0:
						for k in tqdm(range(num_df_aug_train)):
							signal_ch = x_train_no_offset.copy()
							signal_ch = add_freq_offset(signal_ch, rand = rand_aug_train,
																   df = df_aug_train,
																   fc = fc_train_aug, 
																   fs = fs)
							if keep_orig_train is False:
								if k==0:
									x_train_aug_offset = signal_ch.copy()
									y_train_aug_offset = y_train_no_offset.copy()
								else:
									x_train_aug_offset = np.concatenate((x_train_aug_offset, signal_ch), axis=0)
									y_train_aug_offset = np.concatenate((y_train_aug_offset, y_train_no_offset), axis=0)
							else:
								x_train_aug_offset = np.concatenate((x_train_aug_offset, signal_ch), axis=0)
								y_train_aug_offset = np.concatenate((y_train_aug_offset, y_train_no_offset), axis=0)		
					elif df_aug_type == 1:
						offset_dict = {}
						for i in range(401):
							offset_dict[i] = seed_aug_offset		
						for k in tqdm(range(num_df_aug_train)):
							signal_ch = x_train_no_offset.copy()
							for i in tqdm(range(num_train)):
								rv_n = np.random.RandomState(seed=offset_dict[np.argmax(y_train_no_offset[i])])
								signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																					 df = rv_n.uniform(low=-df_aug_train, high=df_aug_train),
																					 fc = fc_train_aug[i:i+1], 
																					 fs = fs)
								offset_dict[np.argmax(y_train_no_offset[i])] += 1
							if keep_orig_train is False:
								if k==0:
									x_train_aug_offset = signal_ch.copy()
									y_train_aug_offset = y_train_no_offset.copy()
								else:
									x_train_aug_offset = np.concatenate((x_train_aug_offset, signal_ch), axis=0)
									y_train_aug_offset = np.concatenate((y_train_aug_offset, y_train_no_offset), axis=0)
							else:
								x_train_aug_offset = np.concatenate((x_train_aug_offset, signal_ch), axis=0)
								y_train_aug_offset = np.concatenate((y_train_aug_offset, y_train_no_offset), axis=0)			


					dict_wifi['x_train'] = x_train_aug_offset.copy()
					dict_wifi['y_train'] = y_train_aug_offset.copy()

					print('x_train: {}'.format(dict_wifi['x_train'].shape))
					print('y_train: {}'.format(dict_wifi['y_train'].shape))


					#--------------------------------------------------------------------------------------------
					# Sliding window augmentation
					#--------------------------------------------------------------------------------------------

					if use_sliding_window is True:
						
						original_datasize = dict_wifi['x_train'].shape[0]
						dict_wifi['x_train'] = get_sliding_window(dict_wifi['x_train'], 
																	window_size=window_size,
																	stride=stride,
																	fs=fs)
						new_datasize = dict_wifi['x_train'].shape[0]

						dict_wifi['y_train'] = np.repeat(dict_wifi['y_train'], new_datasize//original_datasize, axis=0)
						# dict_wifi['fc_train'] = np.repeat(dict_wifi['fc_train'], new_datasize//original_datasize, axis=0)
						# fc_train_orig = dict_wifi['fc_train'].copy()


					#--------------------------------------------------------------------------------------------
					# Data formats
					#--------------------------------------------------------------------------------------------

					data_format_offset = 'car-ch-offset-phy-{}-s-{}-aug-{}-df-{}-rand-{}-ty-{}-{}-'.format(df_phy_train*1e6, seed_phy_train, num_df_aug_train, df_aug_train*1e6, rand_aug_train, df_aug_type, keep_orig_train)

					if equalize_train is False:
						if phy_method==0:
							data_format_ch = 'aug-{}-phy-{}-s-{}-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(num_channel_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, channel_aug_type, num_ch_train, delay_seed_channel_aug_train, snr_train)
						else:
							data_format_ch = 'a{}ug-{}-phy-{}-s-{}-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(phy_method, num_channel_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, channel_aug_type, num_ch_train, delay_seed_channel_aug_train, snr_train)
					else:
						if phy_method==0:
							data_format_ch = 'aug-{}-phy-{}-s-{}-eq-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(num_channel_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, channel_aug_type, num_ch_train, delay_seed_channel_aug_train, snr_train)
						else:
							data_format_ch = 'a{}ug-{}-phy-{}-s-{}-eq-art-{}-ty-{}-nch-{}-de-{}-snr-{:.0f}-'.format(phy_method, num_channel_aug_train, channel_type_phy_train, seed_phy_train, channel_type_aug_train, channel_aug_type, num_ch_train, delay_seed_channel_aug_train, snr_train)

					data_format = data_format_ch + data_format_offset + 't-' + data_format_plain

					# Checkpoint path
					
					if use_sliding_window is True:
						checkpoint = exp_dirs[0] + '/ckpt-slide-{}-{}-aug-'.format(window_size, stride) + data_format +'.h5'
						print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
						train_output, model_name, summary = train_llr_stride(dict_wifi, num_aug_test=num_channel_aug_test, checkpoint_in=None, checkpoint_out=checkpoint, batch_size=100, epochs=epochs)
						print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

					else:
						checkpoint = exp_dirs[0] + '/ckpt-' + data_format +'.h5'
						print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
						train_output, model_name, summary = train_llr_plain(dict_wifi, num_aug_test=num_channel_aug_test, checkpoint_in=None, checkpoint_out=checkpoint, batch_size=100, epochs=epochs)
						print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')


					# Write logs
					with open(exp_dir + '/logs-' + data_format  + '.txt', 'a+') as f:
						f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')
						f.write('\nPreprocessing\n\tType: {}\n\tFs: {} MHz\n\tLength: {} us'.format(preprocess_type, sample_rate, sample_duration))

						f.write('\nEqualization')
						f.write('\tTrain: {}, Test: {}'.format(equalize_train, equalize_test))

						f.write('\nPhysical carrier offset simulation (different days)')
						f.write('\tMethod: {}'.format(phy_method))
						f.write('\tOffsets: Train: {}, Test: {} ppm'.format(df_phy_train*1e6, df_phy_test*1e6))
						f.write('\tSeed: Train: {}, Test: {}'.format(seed_phy_train, seed_phy_test))

						f.write('\nCarrier offset augmentation')
						f.write('\tAugmentation type: {}'.format(df_aug_type))
						f.write('\tNo. of augmentations: Train: {}'.format(num_df_aug_train))
						f.write('\tAugmentation offsets: Train: {}, {}ppm'.format(rand_aug_train, df_aug_train*1e6))

						f.write('\nPhysical channel simulation (different days)')
						f.write('\tMethod: {}'.format(phy_method))
						f.write('\tChannel type: Train: {}, Test: {}'.format(channel_type_phy_train, channel_type_phy_test))
						f.write('\tSeed: Train: {}, Test: {}'.format(seed_phy_train, seed_phy_test))

						f.write('\nChannel augmentation')
						f.write('\tAugmentation type: {}'.format(channel_aug_type))
						f.write('\tNo. of augmentations: Train: {}, Test: {}'.format(num_channel_aug_train, num_channel_aug_test))
						f.write('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))
						f.write('\tChannel types: Train: {}, Test: {}'.format(channel_type_aug_train, channel_type_aug_test))

						if use_sliding_window is True:
							print('\nSliding window augmentation\n\tWindow size: {}\n\tStride: {}'.format(window_size, stride))

						print('\nNoise augmentation\n\tSNR: Train: {} dB, Test {} dB\n'.format(snr_train, snr_test))
						print('\nKeep originals: Train: {}, Test: {}\n'.format(keep_orig_train, keep_orig_test))

						# if use_sliding_window is True:
						# 	f.write('Sliding window augmentation\n\tWindow size: {}\n\tStride: {}\n'.format(window_size, stride))

						# f.write('Carrier offset augmentation\n')
						# f.write('\tAugmentation type: {}'.format(df_aug_type))
						# f.write('\tPhysical offsets: Train: {}, Test:{} ppm\n'.format(df_phy_train*1e6, df_aug_train*1e6))
						# f.write('\tPhysical seeds: Train: {}, Test:{}\n'.format(seed_phy_train, seed_phy_test))
						# f.write('\tAugmentations: {}, Keep orig train: {} \n'.format(num_df_aug_train, keep_orig_train))
						# f.write('\tAugmentation offset: Train: {}, {} ppm\n'.format(rand_aug_train, df_aug_train*1e6))


						# f.write('\nChannel augmentation\n')
						# f.write('\tAugmentation type: {}'.format(channel_aug_type))
						# f.write('\tChannel augmentations: {}, keep_orig: {} \n'.format(num_channel_aug_train, keep_orig_train))
						# f.write('\tChannel type: Phy_train: {}, Phy_test: {}, Aug_Train: {}, Aug_Test: {} \n'.format(channel_type_phy_train, channel_type_phy_test, channel_type_aug_train, channel_type_aug_test))
						# f.write('\tSeed: Phy_train: {}, Phy_test: {}'.format(seed_phy_train, seed_phy_test))
						# f.write('\tNo of channels: Train: {}, Test: {} \n'.format(num_ch_train, num_ch_test))
						# f.write('\tSNR: Train: {} dB, Test {} dB\n'.format(snr_train, snr_test))
						for keys, dicts in train_output.items():
							f.write(str(keys)+':\n')
							for key, value in dicts.items():
								f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
						f.write('\n'+str(summary))



					print('\nPreprocessing\n\tType: {}\n\tFs: {} MHz\n\tLength: {} us'.format(preprocess_type, sample_rate, sample_duration))

					print('\nEqualization')
					print('\tTrain: {}, Test: {}'.format(equalize_train, equalize_test))

					print('\nPhysical carrier offset simulation (different days)')
					print('\tMethod: {}'.format(phy_method))
					print('\tOffsets: Train: {}, Test: {} ppm'.format(df_phy_train*1e6, df_phy_test*1e6))
					print('\tSeed: Train: {}, Test: {}'.format(seed_phy_train, seed_phy_test))

					print('\nCarrier offset augmentation')
					print('\tAugmentation type: {}'.format(df_aug_type))
					print('\tNo. of augmentations: Train: {}'.format(num_df_aug_train))
					print('\tAugmentation offsets: Train: {}, {}ppm'.format(rand_aug_train, df_aug_train*1e6))

					print('\nPhysical channel simulation (different days)')
					print('\tMethod: {}'.format(phy_method))
					print('\tChannel type: Train: {}, Test: {}'.format(channel_type_phy_train, channel_type_phy_test))
					print('\tSeed: Train: {}, Test: {}'.format(seed_phy_train, seed_phy_test))

					print('\nChannel augmentation')
					print('\tAugmentation type: {}'.format(channel_aug_type))
					print('\tNo. of augmentations: Train: {}, Test: {}'.format(num_channel_aug_train, num_channel_aug_test))
					print('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))
					print('\tChannel types: Train: {}, Test: {}'.format(channel_type_aug_train, channel_type_aug_test))

					if use_sliding_window is True:
						print('\nSliding window augmentation\n\tWindow size: {}\n\tStride: {}'.format(window_size, stride))

					# print('\nChannel:')
					# print('\tAugmentation type: {}'.format(channel_aug_type))
					# print('\tNo. of augmentations: Train: {}, Test: {}'.format(num_channel_aug_train, num_channel_aug_test))
					# print('\tPhysical channel types: Train: {}, Test: {}'.format(channel_type_phy_train, channel_type_phy_test))
					# print('\tAugmentation channel types: Train: {}, Test: {}'.format(channel_type_aug_train, channel_type_aug_test))
					# print('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))

					# print('\nCarrier offset:')
					# print('\tAugmentation type: {}'.format(df_aug_type))
					# print('\tNo. of augmentations: Train: {}'.format(num_df_aug_train))
					# print('\tPhysical offsets: Train: {}, Test: {} ppm'.format(df_phy_train*1e6, df_phy_test*1e6))
					# print('\tAugmentation offsets: Train: {}, {}ppm\n'.format(rand_aug_train, df_aug_train*1e6))
					# print('Keep originals: Train: {}, Test: {}\n'.format(keep_orig_train, keep_orig_test))

					print('\nNoise augmentation\n\tSNR: Train: {} dB, Test {} dB\n'.format(snr_train, snr_test))
					print('\nKeep originals: Train: {}, Test: {}\n'.format(keep_orig_train, keep_orig_test))

