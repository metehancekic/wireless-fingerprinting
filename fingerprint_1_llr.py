'''
Trains data for a WiFi experiment.
Data is read from npz files.
'''

import numpy as np
import numpy.random as random
from timeit import default_timer as timer

from .preproc.preproc_wifi import rms
from .preproc.fading_model  import normalize
from .preproc.fading_model  import add_custom_fading_channel

# from .cxnn.train_network _small import train
from .cxnn.train_llr  import train

from tqdm import tqdm
import ipdb


exp_dirs = []
# exp_dirs += ['/home/rfml/wifi/experiments/exp19']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2']
exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2']

preprocess_type = 1
sample_rate = 200
sample_duration = 16
# Set this to 16 to avoid plane ID !!!

# snr_trains_test = [20, 50, 100]
# snr_trains_train = [500]


'''
channel type:
	1 - Extended Pedestrian A
	2 - Extended Vehicular A
	3 - Extended Typical Urban
'''
channel_type_train = 1
channel_type_test = 1

seed_train = 0
seed_test = 1
# seed_test = 0

channel_type_aug_train = 2
channel_type_aug_test = 2

channel_method = 'FFT' 
# channel_method = 'RC' # Raised-cosine

noise_method = 'reg' # Regular
# noise_method = 'bl' # Bandlimited via root raised-cosine

# Random seed for delay perturbation
# Set to False for fixed delays
# delay_seed = False
delay_seed = None

'''
aug_type:
	0 - usual channel aug
	1 - same channel for ith example in each class
'''
aug_type = 0

aug_train = 10
aug_test = 10

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


#--------------------------------------------------------------------------------------------
# Different day scenario simulation
#--------------------------------------------------------------------------------------------

# signal_ch = dict_wifi['x_train'].copy()
# for i in tqdm(range(num_train)):
# 	signal = dict_wifi['x_train'][i][:,0]+1j*dict_wifi['x_train'][i][:,1]
# 	signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
# 														seed=seed_train, 
# 														beta=0, 
# 														delay_seed=False, 
# 														channel_type=channel_type_train,
# 														channel_method=channel_method,
# 														noise_method=noise_method)
# 	signal_faded = normalize(signal_faded)
# 	signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
# dict_wifi['x_train'] = signal_ch.copy()

# signal_ch = dict_wifi['x_test'].copy()
# for i in tqdm(range(num_test)):
# 	signal = dict_wifi['x_test'][i][:,0]+1j*dict_wifi['x_test'][i][:,1]
# 	signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
# 														seed=seed_test, 
# 														beta=0, 
# 														delay_seed=False,
# 														channel_type=channel_type_test,
# 														channel_method=channel_method,
# 														noise_method=noise_method)
# 	signal_faded = normalize(signal_faded)
# 	signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
# dict_wifi['x_test'] = signal_ch.copy()


#--------------------------------------------------------------------------------------------
# Augmentation
#--------------------------------------------------------------------------------------------


x_train = dict_wifi['x_train'].copy()
y_train = dict_wifi['y_train'].copy()

x_test = dict_wifi['x_test'].copy()
y_test = dict_wifi['y_test'].copy()

for snr_train in snr_trains:
	for snr_test in snr_tests:
		data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)
		x_train_aug = x_train.copy()
		y_train_aug = y_train.copy()

		channel_dict = {}
		for i in range(401):
			channel_dict[i] = 0

		if num_ch_train < -1:
			raise ValueError('num_ch_train')
		elif num_ch_train != 0:
			for k in tqdm(range(aug_train)):
				signal_ch = np.zeros(x_train.shape)
				for i in tqdm(range(num_train)):
					signal = x_train[i][:,0]+1j*x_train[i][:,1]
					if num_ch_train==-1:
						signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, 
																			seed=(i + k*num_train) % (num_train*aug_train), 
																			beta=0, 
																			delay_seed=delay_seed, 
																			channel_type=channel_type_aug_train,
																			channel_method=channel_method,
																			noise_method=noise_method)
					elif aug_type==1:
						signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, 
																			seed=channel_dict[np.argmax(y_train[i])],
																			beta=0, 
																			delay_seed=delay_seed,
																			channel_type=channel_type_aug_train,
																			channel_method=channel_method,
																			noise_method=noise_method)
						channel_dict[np.argmax(y_train[i])] += 1
					elif aug_type==0:
						signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, 
																			# seed = 0, 
																			seed = k * num_ch_train + (i % num_ch_train), 
																			beta=0, 
																			delay_seed=delay_seed,
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


		dict_wifi['x_train'] = x_train_aug
		dict_wifi['y_train'] = y_train_aug

		x_test_aug = x_test.copy()
		y_test_aug = y_test.copy()

		if num_ch_test < -1:
			raise ValueError('num_ch_test')
		elif num_ch_test!=0:
			for k in tqdm(range(aug_test)):
				signal_ch = np.zeros(x_test.shape)
				for i in tqdm(range(num_test)):
					signal = dict_wifi['x_test'][i][:,0]+1j*dict_wifi['x_test'][i][:,1]
					if num_ch_test==-1:
						signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, 
																			seed=num_train*aug_train + 1 + (i + k*num_test) % (num_test*aug_test), 
																			beta=0, 
																			delay_seed=delay_seed,
																			channel_type=channel_type_aug_test,
																			channel_method=channel_method,
																			noise_method=noise_method)
					else:
						signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, 
																			# seed = 1, 
																			seed = num_train*aug_train + 1 + (i % num_ch_test) + k * num_ch_test, 
																			beta=0, 
																			delay_seed=delay_seed,
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


		dict_wifi['x_test'] = x_test_aug
		dict_wifi['y_test'] = y_test_aug

		data_format = 'aug-{}-ty-{}-nch-{}-{}-snr-{:.0f}-{:.0f}-'.format(aug_train, aug_type, num_ch_train, num_ch_test, snr_train, snr_test) + data_format

		# Checkpoint path
		checkpoint = exp_dirs[0] + '/ckpt-' + data_format +'.h5'

		end = timer()
		print('Load time: {:} sec'.format(end - start))

		print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
		train_output, model_name, summary = train(dict_wifi, aug_test=aug_test, checkpoint_in=None, checkpoint_out=checkpoint)
		print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

		# Write logs
		with open(exp_dir + '/logs-' + data_format  + '.txt', 'a+') as f:
			f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')
			f.write('\n\n Channel type: Train: {}, Test: {} \n'.format(channel_type_train, channel_type_test))
			f.write('SNR train {} and SNR test {}'.format(snr_train,snr_test))
			for keys, dicts in train_output.items():
				f.write(str(keys)+':\n')
				for key, value in dicts.items():
					f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
			f.write('\n'+str(summary))
