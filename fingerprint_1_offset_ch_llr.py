'''
Trains data for a WiFi experiment with carrier freq offset augmentation.

Data is read from npz files.
'''
import numpy as np
import numpy.random as random
from timeit import default_timer as timer

from .preproc.preproc_wifi import rms
from .preproc.fading_model  import normalize, add_custom_fading_channel, add_freq_offset

# from .cxnn.train_network _small import train
# from .cxnn.train_llr  import train
# from .cxnn.train_llr  import train_200 as train
from .cxnn.train_llr  import train_20 as train

from tqdm import tqdm
import ipdb

##  EXPERIMENT DIRECTORIES TO READ FROM
exp_dirs = []
# exp_dirs += ['/home/rfml/wifi/experiments/exp19']
exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3E']

use_smaller_dataset = True
new_num_classes = 19

# -------------------------------------------------------------------- #
# DATA PREPROCESSING TYPE
# PP = 1 means no detection of preamble,
# PP = 2 means detect and start from beginniong of preamble
# One needs to specify all 3 following parameters

preprocess_type = 1
# preprocess_type = 2

sampling_rate = 20
sample_duration = 16
# -------------------------------------------------------------------- #



# -------------------------------------------------------------------- #
# CHANNEL

# Augmetation method for channel augmentation
# CH_AUG_TYPE = 0  # simple channel augmentation
CH_AUG_TYPE = 1 # channels are different inside a class, but same for i'th packet in each class

# CHANNEL AUGMENTATION
NUM_CH_AUG_TRAIN = 20 # Number of channel augmentations that will be done on train set
NUM_CH_AUG_TEST = 0 # Number of channel augmentations that will be done on test set

# Number of channel filters per augmentation for train data used (-1 corresponds to use different channel for every packet)
NUM_CH_PER_AUG_TRAIN = -1
# Number of channel filters per augmentation for test data used (-1 corresponds to use different channel for every packet)
NUM_CH_PER_AUG_TEST = -1

'''
channel type:
	1 - Extended Pedestrian A (410 ns, 7 taps)
	2 - Extended Vehicular A (2510 ns, 9 taps)
	3 - Extended Typical Urban (5000 ns, 9 taps)
'''
CHANNEL_TYPE_TRAIN = 1
# CHANNEL_TYPE_TRAIN = 2
# CHANNEL_TYPE_TRAIN = 3

CHANNEL_TYPE_TEST = 1

CHANNEL_METHOD = 'FFT' 
# CHANNEL_METHOD = 'RC' # Raised-cosine

NOISE_METHOD = 'reg' # Regular
# NOISE_METHOD = 'bl' # Bandlimited via root raised-cosine

# Random seed for delay perturbation
# Set to False for fixed delays
# DELAY_SEED = None
DELAY_SEED = False
# -------------------------------------------------------------------- #



# -------------------------------------------------------------------- #
# CARRIER FREQUENCY OFFSET
# Number of CFO augmentations will be done on train set
NUM_CFO_AUG_TRAIN = 1 # aug_train = 5
NUM_CFO_AUG_TEST = 0 # aug_train = 5


# Offset for testing
df_test = 40e-6 # 20 ppm
rand_test = 'unif' # Random uniform offset
# rand_test = 'ber' # Random bernoulli offset
# rand_test = False # Fixed offset of -df_test

df_train = 40e-6 
rand_train = 'unif'
# rand_train = 'ber'
# rand_train = False

# -------------------------------------------------------------------- #



# -------------------------------------------------------------------- #
# KEEP ORIGINAL DATA
# Whether you want to keep original set or not
KEEP_ORIG_TRAIN = False
KEEP_ORIG_TEST = False
# -------------------------------------------------------------------- #



# -------------------------------------------------------------------- #
# IN CASE OF NOISE INJECTION INSIDE CHANNEL AUGMENTATION
# SNR values for noise injection in channel augmentation 500>= corresponds to no noise injection
snr_trains = [500]
snr_tests = [500]
# -------------------------------------------------------------------- #



# for NUM_CFO_AUG_TRAIN in [5]:
# 	for rand_train in ['unif', 'ber']:
# 		if NUM_CFO_AUG_TRAIN == 0 and rand_train == 'unif':
# 			continue

print('\n Channel Augmentation')
print('\t Channel type: Train: {}, Test: {} (EPA, EVA, ETU)'.format( CHANNEL_TYPE_TRAIN, CHANNEL_TYPE_TEST ))
print('\t Channel augmentation style (0: simple augmentation) {}'.format( CH_AUG_TYPE ))
print('\t Number of Augmentations (training): {}, (testing): {} '.format(NUM_CH_AUG_TRAIN, NUM_CH_AUG_TEST))
print('\t Number of channels per augmentation (training): {}, (testing): {} '.format(NUM_CH_PER_AUG_TRAIN, NUM_CH_PER_AUG_TEST))
print('\t Channel method : {}, noise method : {} '.format(CHANNEL_METHOD, NOISE_METHOD))
print('\t Delay seed for taps: {} \n'.format(DELAY_SEED))

print('Carrier Frequency Augmentation')
print('\t Randomness of Train CFO: {}, Test CFO: {} (uniform, bernouili, False: (fixed) )'.format( rand_train, rand_test ))
print('\t PPM train : {}, PPM test : {}'.format( df_train, df_test ))
print('\t Number of Augmentations (training): {}, (testing): {} \n'.format(NUM_CFO_AUG_TRAIN, NUM_CFO_AUG_TEST))

print('Keep Original Dataset and Noise Addition')
print('\t Keep Original Data (Train) : {}, (Test) : {}'.format( KEEP_ORIG_TRAIN, KEEP_ORIG_TEST ))
print('\t SNR values for (training): {}, (testing): {} \n'.format(snr_trains[0], snr_tests[0]))


# GET FIRST EXPERIMENT DIRECTION, for multiple directories please add for loop
exp_dir = exp_dirs[0]

# Read Data from npz file
data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sampling_rate)
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





def get_smaller_dataset(dict_wifi, new_num_classes = 19):

	old_num_classes = dict_wifi['num_classes']

	assert old_num_classes >= new_num_classes 

	y_train_orig = dict_wifi['y_train'].copy()
	y_test_orig = dict_wifi['y_test'].copy()

	n_list = np.arange(new_num_classes)

	for n in n_list:
		
		ind_n = np.where(y_train_orig.argmax(axis=1)==n)[0]

		if n == n_list[0]:
			x_train_new = dict_wifi['x_train'][ind_n].copy()
			y_train_new = dict_wifi['y_train'][ind_n].copy()
			fc_train_new = dict_wifi['fc_train'][ind_n].copy()

		else:
			x_train_new = np.concatenate((x_train_new, dict_wifi['x_train'][ind_n].copy()), axis = 0)
			y_train_new = np.concatenate((y_train_new, dict_wifi['y_train'][ind_n].copy()), axis = 0)
			fc_train_new = np.concatenate((fc_train_new, dict_wifi['fc_train'][ind_n].copy()), axis = 0)

	for n in n_list:
		
		ind_n = np.where(y_test_orig.argmax(axis=1)==n)[0]

		if n == n_list[0]:
			x_test_new = dict_wifi['x_test'][ind_n].copy()
			y_test_new = dict_wifi['y_test'][ind_n].copy()
			fc_test_new = dict_wifi['fc_test'][ind_n].copy()

		else:
			x_test_new = np.concatenate((x_test_new, dict_wifi['x_test'][ind_n].copy()), axis = 0)
			y_test_new = np.concatenate((y_test_new, dict_wifi['y_test'][ind_n].copy()), axis = 0)
			fc_test_new = np.concatenate((fc_test_new, dict_wifi['fc_test'][ind_n].copy()), axis = 0)


	dict_wifi['x_train'] = x_train_new
	dict_wifi['y_train'] = y_train_new
	dict_wifi['x_test'] = x_test_new
	dict_wifi['y_test'] = y_test_new
	dict_wifi['fc_train'] = fc_train_new
	dict_wifi['fc_test'] = fc_test_new
	
	dict_wifi['y_train'] = np.delete(dict_wifi['y_train'],np.arange(new_num_classes,old_num_classes),axis=1)
	dict_wifi['y_test'] = np.delete(dict_wifi['y_test'],np.arange(new_num_classes,old_num_classes),axis=1)

	dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]

	assert new_num_classes == dict_wifi['num_classes']

	return dict_wifi

if use_smaller_dataset is True:
	dict_wifi = get_smaller_dataset(dict_wifi, new_num_classes)


num_train = dict_wifi['x_train'].shape[0]
num_test = dict_wifi['x_test'].shape[0]
fc_train = dict_wifi['fc_train']
fc_test = dict_wifi['fc_test']

# Sampling Frequency
fs = sampling_rate * 1e+6


#--------------------------------------------------------------------------------------------
# Train and Test Channel Augmentation
#--------------------------------------------------------------------------------------------

x_train = dict_wifi['x_train'].copy()
y_train = dict_wifi['y_train'].copy()
fc_train = dict_wifi['fc_train'].copy()


if KEEP_ORIG_TRAIN:
	x_train_orig = x_train.copy()
	y_train_orig = y_train.copy()

x_test = dict_wifi['x_test'].copy()
y_test = dict_wifi['y_test'].copy()
fc_test = dict_wifi['fc_test'].copy()

if KEEP_ORIG_TEST:
	x_test_orig = x_test.copy()
	y_test_orig = y_test.copy()

for snr_train in snr_trains:
	for snr_test in snr_tests:
		# data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sampling_rate)
		x_train_aug = x_train.copy()
		y_train_aug = y_train.copy()

		channel_dict = {}
		for i in range(401):
			channel_dict[i] = 0

		if NUM_CH_PER_AUG_TRAIN < -1:
			raise ValueError('NUM_CH_PER_AUG_TRAIN')
		elif NUM_CH_PER_AUG_TRAIN != 0:
			for k in tqdm(range(NUM_CH_AUG_TRAIN)):
				augmented_signal = np.zeros(x_train.shape)
				for i in tqdm(range(num_train)):
					signal = x_train[i][:,0]+1j * x_train[i][:,1]
					if NUM_CH_PER_AUG_TRAIN==-1:
						signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, 
																			seed=(i + k * num_train) % (num_train * NUM_CH_AUG_TRAIN),
																			beta=0, 
																			delay_seed=DELAY_SEED, 
																			channel_type=CHANNEL_TYPE_TRAIN,
																			channel_method=CHANNEL_METHOD,
																			noise_method=NOISE_METHOD)
					elif CH_AUG_TYPE==1:
						signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, 
																			seed=channel_dict[np.argmax(y_train[i])],
																			beta=0, 
																			delay_seed=DELAY_SEED,
																			channel_type=CHANNEL_TYPE_TRAIN,
																			channel_method=CHANNEL_METHOD,
																			noise_method=NOISE_METHOD)
						channel_dict[np.argmax(y_train[i])] += 1
					elif CH_AUG_TYPE==0:
						signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, 
																			# seed = 0, 
																			seed = k * num_ch_train + (i % num_ch_train), 
																			beta=0, 
																			delay_seed=DELAY_SEED,
																			channel_type=CHANNEL_TYPE_TRAIN,
																			channel_method=CHANNEL_METHOD,
																			noise_method=NOISE_METHOD)

					signal_faded = normalize(signal_faded)
					augmented_signal[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))

				# if keep_orig_train is False:
				if k==0:
					x_train_aug = augmented_signal.copy()
					y_train_aug = y_train.copy()
					fc_train_aug = fc_train.copy()
				else:
					x_train_aug = np.concatenate((x_train_aug, augmented_signal), axis=0)
					y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)
					fc_train_aug = np.concatenate((fc_train_aug, fc_train), axis=0)
				# else:
				# 	x_train_aug = np.concatenate((x_train_aug, augmented_signal), axis=0)
				# 	y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)					

		if NUM_CH_AUG_TRAIN>0:
			dict_wifi['x_train'] = x_train_aug.copy()
			dict_wifi['y_train'] = y_train_aug.copy()
			dict_wifi['fc_train'] = fc_train_aug.copy()

		x_test_aug = x_test.copy()
		y_test_aug = y_test.copy()


		if NUM_CH_PER_AUG_TEST < -1:
			raise ValueError(' NUM_CH_PER_AUG_TEST')
		elif NUM_CH_PER_AUG_TEST!=0:
			for k in tqdm(range(NUM_CH_AUG_TEST)):
				augmented_signal = np.zeros(x_test.shape)
				for i in tqdm(range(num_test)):
					signal = dict_wifi['x_test'][i][:,0]+1j*dict_wifi['x_test'][i][:,1]
					if NUM_CH_PER_AUG_TEST==-1:
						signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, 
																			seed=num_train*NUM_CH_AUG_TRAIN + 1 + (i + k*num_test) % (num_test*NUM_CH_AUG_TEST), 
																			beta=0, 
																			delay_seed=DELAY_SEED,
																			channel_type=CHANNEL_TYPE_TEST,
																			channel_method=CHANNEL_METHOD,
																			noise_method=NOISE_METHOD)
					else:
						signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, 
																			# seed = 1, 
																			seed = num_train*NUM_CH_AUG_TRAIN + 1 + (i %  NUM_CH_PER_AUG_TEST) + k *  NUM_CH_PER_AUG_TEST, 
																			beta=0, 
																			delay_seed=DELAY_SEED,
																			channel_type=CHANNEL_TYPE_TEST,
																			channel_method=CHANNEL_METHOD,
																			noise_method=NOISE_METHOD)
					
					signal_faded = normalize(signal_faded)
					augmented_signal[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1)
					# dict_wifi['x_test'][i] = augmented_signal
				# if keep_orig_test is False:
				if k==0:
					x_test_aug = augmented_signal.copy()
					y_test_aug = y_test.copy()
					fc_test_aug = fc_test.copy()
				else:
					x_test_aug = np.concatenate((x_test_aug, augmented_signal), axis=0)
					y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)
					fc_test_aug = np.concatenate((fc_test_aug, fc_test), axis=0)
				# else:
				# 	x_test_aug = np.concatenate((x_test_aug, augmented_signal), axis=0)
				# 	y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)		

		if NUM_CH_AUG_TEST>0:
			dict_wifi['x_test'] = x_test_aug.copy()
			dict_wifi['y_test'] = y_test_aug.copy()
			dict_wifi['fc_test'] = fc_test_aug.copy()

x_test = dict_wifi['x_test'].copy()
y_test = dict_wifi['y_test'].copy()
fc_test = dict_wifi['fc_test'].copy()

x_test_aug = x_test.copy()
y_test_aug = y_test.copy()

#--------------------------------------------------------------------------------------------
# Adding offset to test
#--------------------------------------------------------------------------------------------
for k in tqdm(range(NUM_CFO_AUG_TEST)):
	augmented_signal = x_test.copy()
	augmented_signal = add_freq_offset(augmented_signal, 
									   rand = rand_test,
									   df = df_test,
									   fc = fc_test, 
									   fs = fs)
	if k==0:
		x_test_aug = augmented_signal.copy()
		y_test_aug = y_test.copy()
	else:
		x_test_aug = np.concatenate((x_test_aug, augmented_signal), axis=0)
		y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)

dict_wifi['x_test'] = x_test_aug.copy()
dict_wifi['y_test'] = y_test_aug.copy()

#--------------------------------------------------------------------------------------------
# Train offset augmentation
#--------------------------------------------------------------------------------------------

x_train = dict_wifi['x_train'].copy()
y_train = dict_wifi['y_train'].copy()
fc_train = dict_wifi['fc_train'].copy()

x_train_aug = x_train.copy()
y_train_aug = y_train.copy()

for k in tqdm(range(NUM_CFO_AUG_TRAIN)):
	augmented_signal = x_train.copy()
	saugmented_signal = add_freq_offset(augmented_signal, rand = rand_train,
										   df = df_train,
										   fc = fc_train, 
										   fs = fs)
	# if keep_orig is False:
	if k==0:
		x_train_aug = augmented_signal.copy()
		y_train_aug = y_train.copy()
	else:
		x_train_aug = np.concatenate((x_train_aug, augmented_signal), axis=0)
		y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)
	# else:
	# 	x_train_aug = np.concatenate((x_train_aug, augmented_signal), axis=0)
	# 	y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)					


dict_wifi['x_train'] = x_train_aug.copy()
dict_wifi['y_train'] = y_train_aug.copy()


if KEEP_ORIG_TRAIN is True:
	x_train = dict_wifi['x_train'].copy()
	y_train = dict_wifi['y_train'].copy()
	
	x_train = np.concatenate((x_train, x_train_orig), axis=0)
	y_train = np.concatenate((y_train, y_train_orig), axis=0)

	dict_wifi['x_train'] = x_train.copy()
	dict_wifi['y_train'] = y_train.copy()

if KEEP_ORIG_TEST is True:
	x_test = dict_wifi['x_test'].copy()
	y_test = dict_wifi['y_test'].copy()
	
	x_test = np.concatenate((x_test, x_test_orig), axis=0)
	y_test = np.concatenate((y_test, y_test_orig), axis=0)

	dict_wifi['x_test'] = x_test.copy()
	dict_wifi['y_test'] = y_test.copy()


data_format = 'aug-{}-ty-{}-nch-{}-{}-snr-{:.0f}-{:.0f}-'.format(NUM_CH_AUG_TRAIN, CH_AUG_TYPE, NUM_CH_PER_AUG_TRAIN, NUM_CH_PER_AUG_TEST, snr_train, snr_test)
data_format += 'offset-{}-{}-ko-{}-'.format(NUM_CFO_AUG_TRAIN, rand_train, KEEP_ORIG_TRAIN)
data_format += '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sampling_rate)

# Checkpoint path
checkpoint = exp_dirs[0] + '/ckpt-' + data_format +'.h5'

end = timer()
print('Load time: {:} sec'.format(end - start))

if NUM_CFO_AUG_TEST == 0 and NUM_CH_AUG_TEST == 0 :
	aug_test_num = 1
elif NUM_CFO_AUG_TEST == 0 and NUM_CH_AUG_TEST > 0 :
	aug_test_num = NUM_CH_AUG_TEST 
elif NUM_CFO_AUG_TEST > 0 and NUM_CH_AUG_TEST == 0 :
	aug_test_num = NUM_CFO_AUG_TEST 
elif NUM_CFO_AUG_TEST > 0 and NUM_CH_AUG_TEST > 0 :
	aug_test_num = NUM_CFO_AUG_TEST * NUM_CH_AUG_TEST 
if KEEP_ORIG_TEST:
	aug_test_num += 1

print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
train_output, model_name, summary = train(dict_wifi, checkpoint_in=None, checkpoint_out=checkpoint, num_aug_test = aug_test_num)
print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

# Write logs
with open(exp_dir + '/logs-' + data_format  + '.txt', 'a+') as f:
	f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')


	f.write('\n Channel Augmentation')
	f.write('\t Channel type: Train: {}, Test: {} (EPA, EVA, ETU)'.format( CHANNEL_TYPE_TRAIN, CHANNEL_TYPE_TEST ))
	f.write('\t Channel augmentation style (0: simple augmentation) {}'.format( CH_AUG_TYPE ))
	f.write('\t Number of Augmentations (training): {}, (testing): {} '.format(NUM_CH_AUG_TRAIN, NUM_CH_AUG_TEST))
	f.write('\t Number of channels per augmentation (training): {}, (testing): {} '.format(NUM_CH_PER_AUG_TRAIN, NUM_CH_PER_AUG_TEST))
	f.write('\t Channel method : {}, noise method : {} '.format(CHANNEL_METHOD, NOISE_METHOD))
	f.write('\t Delay seed for taps: {} \n'.format(DELAY_SEED))

	f.write('Carrier Frequency Augmentation')
	f.write('\t Randomness of Train CFO: {}, Test CFO: {} (uniform, bernouili, False: (fixed) )'.format( rand_train, rand_test ))
	f.write('\t PPM train : {}, PPM test : {}'.format( df_train, df_test ))
	f.write('\t Number of Augmentations (training): {}, (testing): {} \n'.format(NUM_CFO_AUG_TRAIN, NUM_CFO_AUG_TEST))

	f.write('Keep Original Dataset and Noise Addition')
	f.write('\t Keep Original Data (Train) : {}, (Test) : {}'.format( KEEP_ORIG_TRAIN, KEEP_ORIG_TEST ))
	f.write('\t SNR values for (training): {}, (testing): {} \n'.format(snr_train, snr_test))

	for keys, dicts in train_output.items():
		f.write(str(keys)+':\n')
		for key, value in dicts.items():
			f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
	f.write('\n'+str(summary))

