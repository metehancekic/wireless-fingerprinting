from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import numpy.random as random
from collections import OrderedDict as odict
from timeit import default_timer as timer
from tqdm import tqdm, trange
from sklearn import metrics

import keras
from keras import backend as K
from keras.utils import plot_model
from keras import optimizers, regularizers, losses
from keras.layers import Dense, Input, Activation, Conv1D, Dropout, GlobalAveragePooling1D, Lambda, Average
from keras.models import Model, load_model
from keras.regularizers import l2

# import complexnn
from .complexnn import ComplexDense, ComplexConv1D, utils

from .models_wifi import *
from .models_adsb import Modrelu

from ..preproc.preproc_wifi import rms
from ..preproc.fading_model import normalize, add_custom_fading_channel, add_freq_offset

exp_dir = '/home/rfml/wifi/experiments/exp19'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2'

preprocess_type = 1
sample_rate = 200
sample_duration = 16

#-------------------------------------------------
# Model name
#-------------------------------------------------

# Regular model
model_name = '-modrelu-100-100-new-'
# model_name = '-100C200x100-ModReLU-100C10x1-ModReLU-Abs-100shared_D-100shared_D-Avg'

# Early abs
# model_name = '-100C200x100-ModReLU-Abs-100shared_C10x1-100shared_D-100shared_D-Avg'

# Short stride
# model_name = '-100C200x10-ModReLU-100C10x1-ModReLU-Abs-100shared_D-100shared_D-Avg'

# Short conv
# model_name = '-100C40x10-ModReLU-100C10x1-ModReLU-Abs-100shared_D-100shared_D-Avg'

# Short conv, but no modrelu
# model_name = '-100C40x10-ModReLU-100C10x1-Abs-100shared_D-100shared_D-Avg'

# Short conv and early abs
# model_name = '-100C40x10-ModReLU-Abs-100shared_C10x1-100shared_D-100shared_D-Avg'

# Short conv and early abs, but no modrelu
# model_name = '-100C40x10-Abs-100shared_C10x1-100shared_D-100shared_D-Avg'

#-------------------------------
# Physical offset params
#-------------------------------
df_phy_train = 40e-6
df_phy_test = 40e-6

# seed_phy_pairs = [(0, 20), (40, 60), (80, 100), (120, 140), (160, 180)]
seed_phy_pairs = [(0, 20)]

#-------------------------------
# Augmentation offset params
#-------------------------------
df_aug_train = df_phy_train 
rand_aug_train = 'unif'
# rand_aug_train = 'ber'
# rand_aug_train = 'False'

df_aug_test = df_aug_train 
rand_aug_test = 'unif'
# rand_aug_test = 'ber'
# rand_aug_test = 'False'

num_aug_train = 0
# num_aug_train = 5
# num_aug_train = 20

# num_aug_test = 0
num_aug_test = 5
# num_aug_test = 20
# num_aug_test = 100

keep_orig_train = False
# keep_orig_train = True

keep_orig_test = False
# keep_orig_test = True
'''
aug_type:
	0 - usual offset aug
	1 - same offset for ith example in each class
'''
aug_type = 0
# aug_type = 1


# from IPython import embed; embed()
# ipdb.set_trace()


data_format_plain = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

npz_filename = exp_dir + '/sym-' + data_format_plain + '.npz'

start = timer()
np_dict = np.load(npz_filename)
dict_wifi = {}
dict_wifi['x_train'] = np_dict['arr_0']
dict_wifi['y_train'] = np_dict['arr_1']
dict_wifi['x_test'] = np_dict['arr_2']
dict_wifi['y_test'] = np_dict['arr_3']
dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]
dict_wifi['fc_train'] = np_dict['arr_4']
dict_wifi['fc_test'] = np_dict['arr_5']
end = timer()
print('Load time: {:} sec'.format(end - start))

fs = sample_rate * 1e+6
sampling_rate = sample_rate * 1e+6

x_train = dict_wifi['x_train'].copy()
y_train = dict_wifi['y_train'].copy()
x_test = dict_wifi['x_test'].copy()
y_test = dict_wifi['y_test'].copy()
num_train = x_train.shape[0]
num_test = x_test.shape[0]
num_features = x_train.shape[1]
num_classes = dict_wifi['num_classes']

x_train_orig = dict_wifi['x_train'].copy()
y_train_orig = dict_wifi['y_train'].copy()
num_classes = y_train_orig.shape[1]

x_test_orig = dict_wifi['x_test'].copy()
y_test_orig = dict_wifi['y_test'].copy()

fc_train = dict_wifi['fc_train']
fc_test = dict_wifi['fc_test']

for seed_phy_train, seed_phy_test in seed_phy_pairs:

	#--------------------------------------------------------------------------------------------
	# Physical offset simulation (different days)
	#--------------------------------------------------------------------------------------------

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
	x_test = signal_ch.copy()


	data_format = 'offset-phy-{}-s-{}-aug-{}-df-{}-rand-{}-ty-{}-{}-t-'.format(df_phy_train*1e6, seed_phy_train, num_aug_train, df_aug_train*1e6, rand_aug_train, aug_type, keep_orig_train)
	data_format += '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

	# checkpoint_in = exp_dir + '/ckpt-' + data_format +'.h5'
	# checkpoint_in += '-modrelu-100-100'+'-new-'
	# checkpoint_in += '-new.h5'

	checkpoint_in = exp_dir + '/ckpt-' + data_format +'.h5'
	checkpoint_in +=  model_name
	checkpoint_in += '-new.h5'

	print('\n-------------------------------')
	print("Loading model from checkpoint")
	print('Model name: {}'.format(model_name[1:]))
	batch_size = 100

	#--------------------------------------------------------------------------------------------
	# Load network
	#--------------------------------------------------------------------------------------------

	print("========================================") 
	print("== BUILDING MODEL... ==")
	batch_size = 100

	if checkpoint_in is None:
		raise ValueError('Cannot test without a checkpoint')

	densenet = load_model(checkpoint_in, 
						  custom_objects={'ComplexConv1D':complexnn.ComplexConv1D,
						  				  'GetAbs': utils.GetAbs,
						  				  'Modrelu': Modrelu})


	#--------------------------------------------------------------------------------------------
	# Offset augmentation
	#--------------------------------------------------------------------------------------------

	x_test_aug = dict_wifi['x_test'].copy()
	y_test_aug = dict_wifi['y_test'].copy()

	for k in tqdm(range(num_aug_test)):
		signal_ch = dict_wifi['x_test'].copy()
		signal_ch = add_freq_offset(signal_ch, rand = rand_aug_test,
											   df = df_aug_test,
											   fc = fc_test, 
											   fs = fs)

		if keep_orig_train is False:
			if k==0:
				x_test_aug = signal_ch.copy()
				y_test_aug = y_test.copy()
			else:
				x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
				y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)
		else:
			x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
			y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)	

	num_test = x_test_aug.shape[0]

	#--------------------------------------------------------------------------------------------
	# Add LLRs
	#--------------------------------------------------------------------------------------------

	# print('Training data size: {}'.format(x_train.shape))
	print('\nTest data size before aug: {}'.format(x_test.shape))
	print('Test data size after aug: {}\n'.format(x_test_aug.shape))

	output_dict = odict(acc=odict(), comp=odict(), loss=odict())

	if num_aug_test!=0:
		num_test_per_aug = num_test // num_aug_test

		logits = densenet.layers[-1].output

		model2 = Model(densenet.input, logits)

		logits_test = model2.predict(x=x_test_aug,
									 batch_size=batch_size,
								 	 verbose=0)		
		logits_test_new = np.zeros((num_test_per_aug, num_classes))
		for i in range(num_aug_test):
			# list_x_test.append(x_test_aug[i*num_test:(i+1)*num_test])

			logits_test_new += logits_test[i*num_test_per_aug:(i+1)*num_test_per_aug]

		# Adding LLRs for num_aug_test test augmentations
		label_pred_llr = logits_test_new.argmax(axis=1)
		label_act = y_test_aug[:num_test_per_aug].argmax(axis=1) 
		ind_correct = np.where(label_pred_llr==label_act)[0] 
		ind_wrong = np.where(label_pred_llr!=label_act)[0] 
		assert (num_test_per_aug == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
		test_acc_llr = 100.*ind_correct.size / num_test_per_aug

		# 1 test augmentation
		probs = densenet.predict(x=x_test_aug[:num_test_per_aug],
								 batch_size=batch_size,
								 verbose=0)
		label_pred = probs.argmax(axis=1)
		ind_correct = np.where(label_pred==label_act)[0] 
		ind_wrong = np.where(label_pred!=label_act)[0] 
		assert (num_test_per_aug == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
		test_acc = 100.*ind_correct.size / num_test_per_aug

		# No test augmentations
		probs = densenet.predict(x=x_test,
								 batch_size=batch_size,
								 verbose=0)
		label_pred = probs.argmax(axis=1)
		label_act = y_test.argmax(axis=1) 
		ind_correct = np.where(label_pred==label_act)[0] 
		ind_wrong = np.where(label_pred!=label_act)[0] 
		assert (x_test.shape[0] == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
		test_acc_no_aug = 100.*ind_correct.size / x_test.shape[0]

		# print("\n========================================") 
		print('Test accuracy (0 aug): {:.2f}%'.format(test_acc_no_aug))
		print('Test accuracy (1 aug): {:.2f}%'.format(test_acc))
		print('Test accuracy ({} aug): {:.2f}%'.format(num_aug_test, test_acc_llr))
		output_dict['acc']['test'] = test_acc_llr

	else:
		probs = densenet.predict(x=x_test,
								 batch_size=batch_size,
								 verbose=0)
		label_pred = probs.argmax(axis=1)
		label_act = y_test.argmax(axis=1) 
		ind_correct = np.where(label_pred==label_act)[0] 
		ind_wrong = np.where(label_pred!=label_act)[0] 
		assert (x_test.shape[0]== ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
		test_acc_no_aug = 100.*ind_correct.size / x_test.shape[0]

		# print("\n========================================") 
		print('Test accuracy (no aug): {:.2f}%'.format(test_acc_no_aug))
		output_dict['acc']['test'] = test_acc_no_aug

	print('Freq offset:\n')
	print('\tPhysical offsets: Train: {}, Test: {} ppm\n'.format(df_phy_train*1e6, df_aug_train*1e6))
	print('\tAugmentations: Train: {}, Test: {}, Keep orig: Train: {}, Test: {} \n'.format(num_aug_train, num_aug_test, keep_orig_train, keep_orig_train, keep_orig_test))
	print('\tAugmentation offsets: Train: {}, {}, Test: {}, {} ppm\n'.format(rand_aug_train, df_aug_train*1e6, rand_aug_test, df_aug_test*1e6))

	print("\n========================================\n") 

del densenet
