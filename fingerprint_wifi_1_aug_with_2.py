'''
Trains data for a WiFi experiment.

Data is read from npz files.
'''

import numpy as np
from timeit import default_timer as timer

from .cxnn.train_network  import train

# exp_dir = '/home/rfml/wifi/experiments/exp19'
# exp_dir = '/home/rfml/wifi/experiments/exp19_S1'
# exp_dir = '/home/rfml/wifi/experiments/exp19_S2'
# exp_dir = '/home/rfml/wifi/experiments/exp100_S1'
# exp_dir = '/home/rfml/wifi/experiments/exp100_S2'
exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2'

sample_duration = 16

preprocess_type = 1
# preprocess_type = 2
# preprocess_type = 3

# sample_rate = 200
sample_rate = 20

file_name_1 = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

outfile_1 = exp_dir + '/sym-' + file_name_1 + '.npz'


beta = 0.5
seed = 0

channel = True
# channel = False

diff_day = False
# diff_day = True

num_ch_train = 1
# num_ch_train = 10

num_ch_test = 0
# num_ch_test = 1

# snr_train = 100
# snr_train = 50
snr_train = 10

snr_test = 30

# file_name_2 = 'dd-{:}-snr-{:.0f}-{:.0f}-b-{:.0f}-n-{:}-{:}-{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(int(diff_day), snr_train, snr_test, 100*beta, num_ch_train, num_ch_test, sample_duration, preprocess_type, sample_rate)


exp_dir = '/home/rfml/wifi/experiments/exp19'

# snr = 0
# snr = 5
# snr = 10
snr = 15
# snr = 20

sample_rate = 200

file_name_2 = '{:.0f}-pp-{:.0f}-fs-{:.0f}-dd-{:}-snr-{:.0f}-b-{:.0f}-s-{:}'.format(sample_duration, preprocess_type, sample_rate, int(diff_day), snr, 100*beta, seed)

outfile_2 = exp_dir + '/sym-' + file_name_2 + '.npz'

start = timer()

np_dict_1 = np.load(outfile_1)
np_dict_2 = np.load(outfile_2)

end = timer()
print('Load time: {:} sec'.format(end - start))

dict_wifi = {}
dict_wifi['x_test'] = np_dict_1['arr_2']

dict_wifi['x_train'] = np.concatenate((np_dict_1['arr_0'], np_dict_2['arr_0'][:, ::10, :]), axis=0)

n1 = np_dict_1['arr_1'].shape[0]
n2 = np_dict_2['arr_1'].shape[0]
num_classes_1 = np_dict_1['arr_3'].shape[1]
y_train_1 = np.concatenate((np_dict_1['arr_1'], np.zeros([n1, 19])), axis=1)
y_train_2 = np.concatenate((np.zeros([n2, num_classes_1]), np_dict_2['arr_1']), axis=1)

n3 = np_dict_1['arr_3'].shape[0]
y_test = np.concatenate((np_dict_1['arr_3'], np.zeros([n3, 19])), axis=1)

dict_wifi['y_train'] = np.concatenate((y_train_1, y_train_2), axis=0)
dict_wifi['y_test'] = y_test
dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]

data_format = 'aug-wifi-2-' + file_name_2

# dict_wifi['x_train'] = np_dict_1['arr_0']
# dict_wifi['y_train'] = np_dict_1['arr_1']
# data_format = file_name_1

# Checkpoint path
checkpoint = exp_dir + '/ckpt-' + data_format + '.h5'

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

