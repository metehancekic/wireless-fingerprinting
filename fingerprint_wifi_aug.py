'''
Trains data for a WiFi experiment.

Data is read from npz files.
'''

import numpy as np
from timeit import default_timer as timer

from .cxnn.train_network  import train

start = timer()

# exp_dir = '/home/rfml/wifi/experiments/exp19'
# exp_dir = '/home/rfml/wifi/experiments/exp19_S1'
# exp_dir = '/home/rfml/wifi/experiments/exp19_S2'
# exp_dir = '/home/rfml/wifi/experiments/exp100_S1'
# exp_dir = '/home/rfml/wifi/experiments/exp100_S2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2'
exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2'

########################################################
# Clean data
########################################################
sample_duration = 16
preprocess_type = 1
sample_rate = 20

file_name_1 = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

outfile_1 = exp_dir + '/sym-' + file_name_1 + '.npz'
np_dict_1 = np.load(outfile_1)
dict_wifi = {}
dict_wifi['x_train'] = np_dict_1['arr_0']
dict_wifi['y_train'] = np_dict_1['arr_1']
dict_wifi['x_test'] = np_dict_1['arr_2']
dict_wifi['y_test'] = np_dict_1['arr_3']
dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]

########################################################
#  Data with channel
########################################################
channel = True
diff_day = False
num_ch_train = 1
num_ch_test = 0
snr_train = 500
snr_test = 500
# beta = 0.5
beta = 2

seeds = [0]
# seeds = [0, 1]
# seeds = [0, 1, 2]
# seeds = [0, 1, 2, 3, 4, 5, 6, 7]

data_format = 'aug-s-'
for seed in seeds:
	file_name_2 = 'dd-{:}-snr-{:.0f}-{:.0f}-b-{:.0f}-n-{:}-{:}-{:.0f}-pp-{:.0f}-fs-{:.0f}-s-{:}'.format(int(diff_day), snr_train, snr_test, 100*beta, num_ch_train, num_ch_test, sample_duration, preprocess_type, sample_rate, seed)

	outfile_2 = exp_dir + '/sym-' + file_name_2 + '.npz'
	np_dict_2 = np.load(outfile_2)
	dict_wifi['x_train'] = np.concatenate((dict_wifi['x_train'], np_dict_2['arr_0']), axis=0)
	dict_wifi['y_train'] = np.concatenate((dict_wifi['y_train'], np_dict_2['arr_1']), axis=0)

	data_format += '{:}-'.format(seed)
data_format += file_name_1

# Checkpoint path
checkpoint = exp_dir + '/ckpt-' + data_format + '.h5'

end = timer()
print('Load time: {:} sec'.format(end - start))

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

