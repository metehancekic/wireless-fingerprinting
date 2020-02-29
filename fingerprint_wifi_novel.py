'''
Trains data for a WiFi experiment.

Data is read from npz files.
'''

import numpy as np
from timeit import default_timer as timer

novel_classes = np.array([0])
# novel_classes = np.array([0, 1])
# novel_classes = np.array([0, 1, 2])

# exp_dir = '/home/rfml/wifi/experiments/exp19'
# exp_dir = '/home/rfml/wifi/experiments/exp19_S1'
# exp_dir = '/home/rfml/wifi/experiments/exp19_S2'
# exp_dir = '/home/rfml/wifi/experiments/exp100_S1'
# exp_dir = '/home/rfml/wifi/experiments/exp100_S2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2'
exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2'

# for novel_classes in [np.array([0]),
# 					  np.array([1]),
# 					  np.array([2]),
# 					  np.array([3]),
# 					  np.array([4]),
# 					  np.array([1, 7])]:
	# for exp_dir in ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2',
	# 				'/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2']:
	# for exp_dir in ['/home/rfml/wifi/experiments/exp19',
	# 				'/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2',
	# 				'/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2',
	# 				'/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2']:

for novel_classes in [np.array([3]),
					  np.array([0, 1]),
					  np.array([0, 1, 2]),
					  np.array([0, 1, 2, 3]),
					  np.array([0, 1, 2, 3, 4])]:
	for exp_dir in ['/home/rfml/wifi/experiments/exp19']:

		sample_rate = 20
		preprocess_type = 1
		sample_duration = 16

		noise = False
		snr_train = 30
		snr_test = 500

		channel = False
		diff_day = False
		num_ch_train = 1
		num_ch_test = 0
		beta = 2
		seed = 0

		if channel is True:
			data_format = 'dd-{:}-snr-{:.0f}-{:.0f}-b-{:.0f}-n-{:}-{:}-{:.0f}-pp-{:.0f}-fs-{:.0f}-s-{:}'.format(int(diff_day), snr_train, snr_test, 100*beta, num_ch_train, num_ch_test, sample_duration, preprocess_type, sample_rate, seed)
		elif noise is True:
			data_format = 'snr-{:.0f}-{:.0f}-l-{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(snr_train, snr_test, sample_duration, preprocess_type, sample_rate)
		else:
			data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

		outfile = exp_dir + '/sym-' + data_format + '.npz'
		data_format = 'l-' + data_format
		np_dict = np.load(outfile)
		x_train = np_dict['arr_0']
		y_train = np_dict['arr_1']
		x_test = np_dict['arr_2']
		y_test = np_dict['arr_3']
		num_classes = y_test.shape[1]

		print('\n-----------------\nOriginal data\n-----------------')
		print('x_train.shape = {}'.format(x_train.shape))
		print('y_train.shape = {}'.format(y_train.shape))
		ind_novel = np.empty([0], dtype=np.int)
		for n in novel_classes:
			ind_n = np.where(y_train.argmax(axis=1)==n)[0]
			print('ind_{}.shape = {}'.format(n, ind_n.shape))
			ind_novel = np.concatenate((ind_novel, ind_n))
			data_format = '-{}'.format(n) + data_format
		data_format = 'novel' + data_format

		print('ind_novel.shape = {}'.format(ind_novel.shape))

		mask = np.ones(x_train.shape[0], dtype=bool)
		mask[list(ind_novel)] = False
		x_train = x_train[mask]
		y_train = y_train[mask]

		mask = np.ones(num_classes, dtype=bool)
		mask[list(novel_classes)] = False
		y_train = y_train[:, mask]
		y_test = y_test[:, mask]

		# x_novel = np.concatenate

		print('\n-----------------\nData without classes {}\n-----------------'.format(novel_classes))
		print('x_train.shape = {}'.format(x_train.shape))
		print('y_train.shape = {}'.format(y_train.shape))
		print('x_test.shape = {}'.format(x_test.shape))
		print('y_test.shape = {}'.format(y_test.shape))
		for n in novel_classes:
			ind_n = np.where(y_train.argmax()==n)[0]
			print('ind_{}.shape = {}'.format(n, ind_n.shape))

		dict_wifi = {}
		dict_wifi['x_train'] = x_train
		dict_wifi['y_train'] = y_train
		dict_wifi['x_test'] = x_test
		dict_wifi['y_test'] = y_test
		dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]

		# Checkpoint path
		checkpoint = exp_dir + '/ckpt-' + data_format + '.h5'

		from .cxnn.train_network _small import train

		print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
		train_output, model_name, summary = train(dict_wifi, checkpoint_in=None, checkpoint_out=checkpoint)
		print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

		# Write logs
		with open(exp_dir + '/logs-' + data_format  + '.txt', 'a+') as f:
			f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')
			for keys, dicts in train_output.items():
				f.write(str(keys)+':\n')
				for key, value in dicts.items():
					f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
			f.write('\n'+str(summary))

