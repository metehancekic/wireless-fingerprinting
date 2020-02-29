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


new_num_classes = 19

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

dict_wifi = get_smaller_dataset(dict_wifi, new_num_classes)





