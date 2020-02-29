'''
Preprocesses and stores data for a WiFi experiment.

Use the 'preprocess_type' argument to switch on fractionally spaced equalization
'''

import numpy as np
import os
import argparse                     

from .preproc.preproc_wifi import parse_input_files, read_wifi, preprocess_wifi
from .preproc.fading_model  import add_fading_channel

parser = argparse.ArgumentParser(description='Trains network on WiFi data')
parser.add_argument('-d', '--exp_dir', type=str, default='/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2', help='Base directory for experiment')
# Available experiments:
# exp19
# exp19_S1
# exp19_S2
# exp100_S1
# exp100_S2
# NSWC_Crane_Experiments/Test3/converted_3Av2
# NSWC_Crane_Experiments/Test3/converted_3Bv2
# NSWC_Crane_Experiments/Test3/converted_3Cv2
# NSWC_Crane_Experiments/Test3/converted_3Dv2 
parser.add_argument('-fs', '--sample_rate', type=np.float, default=200., help='Sampling rate in MHz (default 200 MHz)')
parser.add_argument('-l', '--sample_duration', type=np.float, default=16., help='Length of pruned packet in us (default 16 us)')
parser.add_argument('-pp', '--preprocess_type', type=int, default=1, help='Type of preprocessing (1-3)')
# 1 - No preamble detection -> Uses the first 'sample_duration' us of packet
# 2 - Preamble detection -> Finds beginning of preamble and then uses the next 'sample_duration' us of packet
# 3 - Fractionally spaced equalization -> Detects and equalizes the preamble. (This is slower than the other 2 types of preprocessing.)
parser.add_argument('-ch', '--channel', action='store_true', help='Add channel')
parser.add_argument('-dd', '--diff_day', action='store_true', help='Training and test data from different days')
parser.add_argument('-n', '--num_ch', type=np.int, default=1, help='Number of channels (default 1)')
parser.add_argument('-snr', '--snr', type=np.float, default=20., help='SNR in dB (default 20 dB)')
parser.add_argument('-b', '--beta', type=np.float, default=0.5, help='Roll-off factor for RC pulse (default 0.5)')
parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed for channel (default 0)')
parser.add_argument('-p', '--progress', action='store_false', help='Switch off progress bars')
args = parser.parse_args()
progress = True

# Get experiment and data directories
experiment_directory = args.exp_dir
data_directory = os.environ.get('TDYRADIO_DATA')
assert os.path.isdir(experiment_directory)
assert os.path.isdir(data_directory)
devices_csv = os.path.join(experiment_directory, 'Devices.csv')
training_csv = os.path.join(experiment_directory, 'Training_Signals.csv')
testing_csv = os.path.join(experiment_directory, 'Testing_Signals.csv')

# Parse csv files
train_files = parse_input_files(training_csv, devices_csv)
test_files = parse_input_files(testing_csv, devices_csv)
device_map = train_files['device_map'] # Device label map: eg. {'crane-gfi_3_dataset-7515': 88}
device_list = train_files['device_list'] # List of devices
num_devices = len(device_list)
print('Number of devices: {:}'.format(num_devices))

# Read data using parsed csv files
print('Reading WiFi data')
read_params = {'base_data_directory': data_directory,
			   'device_map': device_map,
			   'progress': args.progress}
train_dict = read_wifi(train_files, **read_params)
test_dict = read_wifi(test_files, **read_params)

print('-----------------------\nExperiment:\n' + experiment_directory + '\n-----------------------')

# Print some analytics
num_train = len(train_dict['data_file'])
num_test = len(test_dict['data_file'])
fs_train = np.array(train_dict['capture_sample_rate'].values())
fs_test = np.array(test_dict['capture_sample_rate'].values())
pct_train_200 = 100.*(fs_train == np.int(200e6)).sum() / num_train
pct_train_30 = 100.*(fs_train == np.int(30e6)).sum() / num_train
pct_test_200 = 100.*(fs_test == np.int(200e6)).sum() / num_test
pct_test_30 = 100.*(fs_test == np.int(30e6)).sum() / num_test
print('\nTraining set size = {} \nTest set size = {}\n'.format(num_train, num_test))
print('Capture Sampling Rate composition:')
print('Training set: {:.2f}% 200MHz, {:.2f}% 30MHz \nTest set: {:.2f}% 200MHz, {:.2f}% 30MHz\n'.format(pct_train_200, pct_train_30, pct_test_200, pct_test_30))

# Add channel and noise
if args.channel is True:
	print('Adding channel to WiFi data')
	channel_params = {'snr': args.snr,
					  'beta': args.beta,
  					  'num_ch': args.num_ch,
					  'progress': args.progress}
	seed_train = args.seed
	if args.diff_day is False:
		seed_test = seed_train
	else:
		seed_test = seed_train + 1
	train_dict = add_fading_channel(train_dict, seed=seed_train, **channel_params)
	test_dict = add_fading_channel(test_dict, seed=seed_test, **channel_params)

# Preprocess data
print('Preprocessing WiFi data')
preproc_params = {'sample_duration': args.sample_duration*1e-6,
				  'sample_rate': args.sample_rate*1e6,
				  'preprocess_type': args.preprocess_type,
				  'progress': args.progress}
train_dict = preprocess_wifi(train_dict, **preproc_params)
test_dict = preprocess_wifi(test_dict, **preproc_params)

# Reshape complex data so that Keras can understand it
x_train = np.array(train_dict['signal'].values())
x_train = np.concatenate((x_train.real[..., None], x_train.imag[..., None]), axis=2)
x_test = np.array(test_dict['signal'].values())
x_test = np.concatenate((x_test.real[..., None], x_test.imag[..., None]), axis=2)

# Do one hot encoding of labels
y_train = np.zeros([num_train, num_devices], dtype=np.int)
y_test = np.zeros([num_test, num_devices], dtype=np.int)
I = np.eye(num_devices)
for i in range(num_train):
	y_train[i] = I[train_dict['device_key'][i]]
for i in range(num_test):
	y_test[i] = I[test_dict['device_key'][i]]

# Store data
if args.channel is False:
	outfile = experiment_directory + '/sym-{:.0f}-pp-{:.0f}-fs-{:.0f}.npz'.format(args.sample_duration, args.preprocess_type, args.sample_rate)
elif args.num_ch==1:
	outfile = experiment_directory + '/sym-{:.0f}-pp-{:.0f}-fs-{:.0f}-dd-{:}-snr-{:.0f}-b-{:.0f}-s-{:}.npz'.format(args.sample_duration, args.preprocess_type, args.sample_rate, int(args.diff_day), args.snr, 100*args.beta, args.seed)
else:
	outfile = experiment_directory + '/sym-{:.0f}-pp-{:.0f}-fs-{:.0f}-dd-{:}-snr-{:.0f}-b-{:.0f}-s-{:}-n-{:}.npz'.format(args.sample_duration, args.preprocess_type, args.sample_rate, int(args.diff_day), args.snr, 100*args.beta, args.seed, args.num_ch)
np.savez_compressed(outfile, x_train, y_train, x_test, y_test)