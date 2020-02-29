'''
Trains data for a WiFi experiment with carrier freq offset augmentation.

Data is read from npz files.
'''
import numpy as np
import numpy.random as random
from timeit import default_timer as timer
from tqdm import tqdm, trange
import ipdb

from ..preproc.fading_model import normalize, add_custom_fading_channel, add_freq_offset
from ..preproc.preproc_wifi import rms, get_sliding_window, offset_compensate_preamble

# from .cxnn.train_network _small import train
# from .cxnn.train_network _aug import train
# from train_llr import train
from ..cxnn.train_llr import train_200, train_20


exp_dirs = []
exp_dirs += ['/home/rfml/wifi/experiments/exp19']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2']

preprocess_type = 1
sample_rate = 200
sample_duration = 16

#-------------------------------
# Physical offset params
#-------------------------------
df_phy_train = 20e-6
df_phy_test = 20e-6

# seed_phy_pairs = [(0, 20), (40, 60), (80, 100), (120, 140), (160, 180)]
seed_phy_pairs = [(0, 20)]

#-------------------------------
# Compensation params
#-------------------------------
compensate_train = True
compensate_test = True

verbose_train = False
verbose_test = False

#-------------------------------
# Noise augmentation params
#-------------------------------

# snrs_test = [20, 50, 100, 500]
# snrs_test = [100, 500]
# snrs_test = [50]
# snrs_test = [500, 100]
snrs_test = [500]

# snrs_train = [10, 15, 20, 25, 500]
# snrs_train = [20]
snrs_train = [500]

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
dict_wifi['fc_train'] = np_dict['arr_4']
dict_wifi['fc_test'] = np_dict['arr_5']
dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]
end = timer()
print('Load time: {:} sec'.format(end - start))

num_train = dict_wifi['x_train'].shape[0]
num_test = dict_wifi['x_test'].shape[0]
fc_train = dict_wifi['fc_train']
fc_test = dict_wifi['fc_test']

fs = sample_rate * 1e+6

print('\nPreprocessing\n\tType: {}\n\tFs: {} MHz\n\tLength: {} us'.format(preprocess_type, sample_rate, sample_duration))

#--------------------------------------------------------------------------------------------
# Offset compensation
#--------------------------------------------------------------------------------------------

print('\nOffset compensation')
print('\tTrain: {}, Test: {}'.format(compensate_train, compensate_test))

if compensate_train is True:
	complex_train = dict_wifi['x_train'][..., 0] + 1j* dict_wifi['x_train'][..., 1]

	for i in range(num_train):
		complex_train[i] = offset_compensate_preamble(complex_train[i], 
													   fs=fs, 
													   verbose=verbose_train)

	dict_wifi['x_train'] = np.concatenate((complex_train.real[..., None], complex_train.imag[..., None]), axis=2)

if compensate_test is True:
	complex_test = dict_wifi['x_test'][..., 0] + 1j* dict_wifi['x_test'][..., 1]

	for i in range(num_test):
		complex_test[i] = offset_compensate_preamble(complex_test[i], 
													  fs=fs, 
													  verbose=verbose_test)

	dict_wifi['x_test'] = np.concatenate((complex_test.real[..., None], complex_test.imag[..., None]), axis=2)


x_train_orig = dict_wifi['x_train'].copy()
y_train_orig = dict_wifi['y_train'].copy()
num_classes = y_train_orig.shape[1]

x_test_orig = dict_wifi['x_test'].copy()
y_test_orig = dict_wifi['y_test'].copy()

for seed_phy_train, seed_phy_test in seed_phy_pairs:

	print('\nPhysical carrier offset simulation (different days)')
	# print('\tMethod: {}'.format(phy_method))
	print('\tOffsets: Train: {}, Test: {} ppm'.format(df_phy_train*1e6, df_phy_test*1e6))
	print('\tSeed: Train: {}, Test: {}\n'.format(seed_phy_train, seed_phy_test))

	#--------------------------------------------------------------------------------------------
	# Physical offset simulation (different days)
	#--------------------------------------------------------------------------------------------
	signal_ch = x_train_orig.copy()
	for n in trange(num_classes):
		ind_n = np.where(y_train_orig.argmax(axis=1)==n)[0]
		seed_phy_train_n = seed_phy_train + n
		for i in ind_n:
			rv_n = np.random.RandomState(seed=seed_phy_train_n)
			signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																 df = rv_n.uniform(low=-df_phy_train, high=df_phy_train),
																 fc = fc_train[i:i+1], 
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
																 fc = fc_test[i:i+1], 
																 fs = fs)
	dict_wifi['x_test'] = signal_ch.copy()


	#--------------------------------------------------------------------------------------------
	# Offset compensation
	#--------------------------------------------------------------------------------------------

	print('\nOffset compensation')
	print('\tTrain: {}, Test: {}'.format(compensate_train, compensate_test))

	if compensate_train is True:
		complex_train = dict_wifi['x_train'][..., 0] + 1j* dict_wifi['x_train'][..., 1]

		for i in range(num_train):
			complex_train[i] = offset_compensate_preamble(complex_train[i], 
														   fs=fs, 
														   verbose=verbose_train)

		dict_wifi['x_train'] = np.concatenate((complex_train.real[..., None], complex_train.imag[..., None]), axis=2)

	if compensate_test is True:
		complex_test = dict_wifi['x_test'][..., 0] + 1j* dict_wifi['x_test'][..., 1]

		for i in range(num_test):
			complex_test[i] = offset_compensate_preamble(complex_test[i], 
														  fs=fs, 
														  verbose=verbose_test)

		dict_wifi['x_test'] = np.concatenate((complex_test.real[..., None], complex_test.imag[..., None]), axis=2)

	#--------------------------------------------------------------------------------------------
	# Noise augmentation
	#--------------------------------------------------------------------------------------------

	x_train_clean = dict_wifi['x_train'].copy()
	x_test_clean = dict_wifi['x_test'].copy()

	for snr_test in snrs_test:
		for snr_train in snrs_train:

			print('\nNoise augmentation: \n\tTrain: {} dB, Test: {} dB'.format(snr_train, snr_test))

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

			data_format_snr = 'snr-{:.0f}-{:.0f}-'.format(snr_train, snr_test)

			data_format_offset = 'offset-phy-{}-s-{}-comp-{}-t-'.format(df_phy_train*1e6, seed_phy_train, compensate_train)

			data_format = data_format_offset + data_format_snr + '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

			# Checkpoint path
			checkpoint = exp_dirs[0] + '/ckpt-' + data_format +'.h5'

			end = timer()
			print('Load time: {:} sec'.format(end - start))

			print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
			if sample_rate==20:
				train_output, model_name, summary = train_20(dict_wifi, checkpoint_in=None, checkpoint_out=checkpoint)
			elif sample_rate==200:
				train_output, model_name, summary = train_200(dict_wifi, checkpoint_in=None, checkpoint_out=checkpoint)
			print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

			# Write logs
			with open(exp_dir + '/logs-' + data_format  + '.txt', 'a+') as f:
				f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')

				f.write('\nPreprocessing\n\tType: {}\n\tFs: {} MHz\n\tLength: {} us'.format(preprocess_type, sample_rate, sample_duration))

				f.write('\nPhysical carrier offset simulation (different days)')
				f.write('\tOffsets: Train: {}, Test: {} ppm'.format(df_phy_train*1e6, df_phy_test*1e6))
				f.write('\tSeed: Train: {}, Test: {}'.format(seed_phy_train, seed_phy_test))

				f.write('\nCarrier offset compensation: \n\tTrain: {}, Test: {}'.format(compensate_train, compensate_test))

				f.write('\nNoise augmentation: \n\tTrain: {} dB, Test: {} dB'.format(snr_train, snr_test))

				for keys, dicts in train_output.items():
					f.write(str(keys)+':\n')
					for key, value in dicts.items():
						f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
				f.write('\n'+str(summary))

			print('\nPreprocessing\n\tType: {}\n\tFs: {} MHz\n\tLength: {} us'.format(preprocess_type, sample_rate, sample_duration))

			print('\nPhysical carrier offset simulation (different days)')
			print('\tOffsets: Train: {}, Test: {} ppm'.format(df_phy_train*1e6, df_phy_test*1e6))
			print('\tSeed: Train: {}, Test: {}'.format(seed_phy_train, seed_phy_test))

			print('\nCarrier offset compensation: \n\tTrain: {}, Test: {}'.format(compensate_train, compensate_test))

			print('\nNoise augmentation: \n\tTrain: {} dB, Test: {} dB'.format(snr_train, snr_test))