'''
Trains data for a WiFi experiment.

Data is read from npz files.
'''

import numpy as np
from timeit import default_timer as timer
import argparse
import os

from .cxnn.train_network_reim_mag import train

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

exp_dir = os.environ['path_to_data']

# sample_rate = 20
sample_rate = 200

preprocess_type = 1
sample_duration = 16

noise = False
snr_train = 500
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
np_dict = np.load(outfile)
dict_wifi = {}
dict_wifi['x_train'] = np_dict['arr_0']
dict_wifi['y_train'] = np_dict['arr_1']
dict_wifi['x_test'] = np_dict['arr_2']
dict_wifi['y_test'] = np_dict['arr_3']
dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]

data_format += '_{}'.format(architecture)

# Checkpoint path
checkpoint = exp_dir + '/ckpt-' + data_format + '.h5'

print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
# train_output, model_name, summary, conf_matrix_test = train(dict_wifi, checkpoint_in=None, checkpoint_out=checkpoint)
train_output, model_name, summary = train(dict_wifi, checkpoint_in=None, 
													 checkpoint_out=checkpoint,
													 architecture=architecture,
													 fs=sample_rate)
print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')


# Write logs
with open(exp_dir + '/logs-' + data_format  + '.txt', 'a+') as f:
	f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')
	for keys, dicts in train_output.items():
		f.write(str(keys)+':\n')
		for key, value in dicts.items():
			f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
	f.write('\n'+str(summary))

