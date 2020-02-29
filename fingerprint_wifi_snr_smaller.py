'''
Trains data for a WiFi experiment.
Data is read from npz files.
'''

import numpy as np
import numpy.random as random
from timeit import default_timer as timer
from tqdm import trange

from .preproc.preproc_wifi import rms
from .preproc.fading_model  import normalize

from .cxnn.train_network _smaller import train


exp_dirs = []
# exp_dirs += ['/home/rfml/wifi/experiments/exp19']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2']
exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2']

preprocess_type = 1
sample_rate = 20
sample_duration = 16
# Set this to 16 to avoid plane ID !!!

# snrs_test = [20, 50, 100, 500]
# snrs_test = [100, 500]
# snrs_test = [50]
snrs_test = [500]

# snrs_train = [10, 15, 20, 25, 500]
# snrs_train = [20]
snrs_train = [500]

data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

for exp_dir in exp_dirs:

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

	# Checkpoint path
	checkpoint = exp_dir + '/ckpt-' + data_format

	# print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
	# train_output, model_name, summary = train(dict_wifi, checkpoint_in=None, checkpoint_out=checkpoint)
	# print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

	# # Write logs
	# with open(exp_dir + '/logs-' + data_format  + '.txt', 'a+') as f:
	# 	f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')
	# 	for keys, dicts in train_output.items():
	# 		f.write(str(keys)+':\n')
	# 		for key, value in dicts.items():
	# 			f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
	# 	f.write('\n'+str(summary))

	x_train_clean = dict_wifi['x_train'].copy()
	x_test_clean = dict_wifi['x_test'].copy()

	for snr_test in snrs_test:
		# snrs_train = list(np.arange(0, snr_test+1, 5))
		for snr_train in snrs_train:

			x_train = x_train_clean.copy()
			x_test = x_test_clean.copy()

			if snr_train < 500:
				print("Train SNR {}".format(snr_train))
				for i in trange(x_train.shape[0]):
					
					signal = x_train[i,:,0] + 1j*x_train[i,:,1]

					rv_noise = random.RandomState(seed=None)
					E_b = (np.abs(signal)**2).mean()
					N_0 = E_b/(10**(snr_train/10))
					N = len(signal)
					n = rv_noise.normal(loc=0, scale=np.sqrt(N_0/2), size=(N, 2)).dot(np.array([1, 1j]))
					signal += n

					signal = normalize(signal)
					x_train[i,:,0] = signal.real
					x_train[i,:,1] = signal.imag
				dict_wifi['x_train'] = x_train.copy()

			if snr_test < 500:
				print("Test SNR {}".format(snr_test))
				for i in trange(x_test.shape[0]):
					
					signal = x_test[i,:,0] + 1j*x_test[i,:,1]

					rv_noise = random.RandomState(seed=None)
					E_b = (np.abs(signal)**2).mean()
					N_0 = E_b/(10**(snr_test/10))
					N = len(signal)
					n = rv_noise.normal(loc=0, scale=np.sqrt(N_0/2), size=(N, 2)).dot(np.array([1, 1j]))
					signal += n

					signal = normalize(signal)
					x_test[i,:,0] = signal.real
					x_test[i,:,1] = signal.imag
				dict_wifi['x_test'] = x_test.copy()

			data_format_snr = 'snr-{:.0f}-{:.0f}-'.format(snr_train, snr_test) + data_format

			# Checkpoint path
			checkpoint = exp_dir + '/ckpt-' + data_format_snr

			print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
			train_output, model_name, summary = train(dict_wifi, checkpoint_in=None, checkpoint_out=checkpoint)
			print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

			# Write logs
			with open(exp_dir + '/logs-' + data_format_snr  + '.txt', 'a+') as f:
				f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')
				for keys, dicts in train_output.items():
					f.write(str(keys)+':\n')
					for key, value in dicts.items():
						f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
				f.write('\n'+str(summary))

