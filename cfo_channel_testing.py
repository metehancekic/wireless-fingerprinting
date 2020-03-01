'''
Real life channel and CFO experiments are done in this code.


 - Physical Layer Channel Simulation
 - Physical Layer CFO Simulation
 - Channel Equalization
 - CFO Compensation
 - Channel Augmentation
 - CFO Augmentation
'''

import numpy as np
from timeit import default_timer as timer
import argparse
from tqdm import trange, tqdm
import json
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict as odict
import copy

from simulators import physical_layer_channel, physical_layer_cfo, cfo_compansator, equalize_channel, augment_with_channel_test, augment_with_cfo_test, get_residual

from cxnn.complexnn import ComplexDense, ComplexConv1D, utils, Modrelu

import keras
from keras import backend as K
from keras.utils import plot_model
from keras import optimizers, regularizers, losses
from keras.layers import Dense, Input, Activation, Conv1D, Dropout, GlobalAveragePooling1D, Lambda, Average
from keras.models import Model, load_model
from keras.regularizers import l2

def test_experiments(architecture, config, num_days, seed_days, seed_test_day, experiment_setup, testing_setup):

	# print(architecture)

	#-------------------------------------------------
	# Analysis
	#-------------------------------------------------

	plot_signal = False
	check_signal_power_effect = False

	#-------------------------------------------------
	# Data configuration
	#-------------------------------------------------

	exp_dir = config['exp_dir']
	sample_duration = config['sample_duration']
	preprocess_type = config['preprocess_type']
	sample_rate = config['sample_rate']

	#-------------------------------------------------
	# Training configuration
	#-------------------------------------------------
	epochs = config['epochs']


	#-------------------------------------------------
	# Equalization before any preprocessing
	#-------------------------------------------------
	equalize_train_before = experiment_setup['equalize_train_before']
	equalize_test_before = experiment_setup['equalize_test_before']


	#-------------------------------------------------
	# Physical Channel Parameters
	#-------------------------------------------------
	add_channel = experiment_setup['add_channel']

	phy_method = num_days
	seed_phy_train = seed_days
	# seed_phy_test = config['seed_phy_test']
	seed_phy_test = seed_test_day
	channel_type_phy_train = config['channel_type_phy_train']
	channel_type_phy_test = config['channel_type_phy_test']
	phy_noise = config['phy_noise']
	snr_train_phy = config['snr_train_phy']
	snr_test_phy = config['snr_test_phy']

	#-------------------------------------------------
	# Physical CFO parameters
	#-------------------------------------------------

	add_cfo = experiment_setup['add_cfo']
	remove_cfo = experiment_setup['remove_cfo']

	phy_method_cfo = phy_method  # config["phy_method_cfo"]
	df_phy_train = config['df_phy_train']
	df_phy_test = config['df_phy_test']
	seed_phy_train_cfo = seed_phy_train # config['seed_phy_train_cfo']
	seed_phy_test_cfo = seed_phy_test # config['seed_phy_test_cfo']

	#-------------------------------------------------
	# Equalization params
	#-------------------------------------------------
	equalize_train = experiment_setup['equalize_train']
	equalize_test = testing_setup['equalize_test']
	verbose_train = False
	verbose_test = False

	#-------------------------------------------------
	# Augmentation channel parameters
	#-------------------------------------------------
	augment_channel = experiment_setup['augment_channel']

	channel_type_aug_train = config['channel_type_aug_train']
	channel_type_aug_test = config['channel_type_aug_test']
	num_aug_train = config['num_aug_train']
	num_aug_test = config['num_aug_test']
	aug_type = config['aug_type']
	num_ch_train = config['num_ch_train']
	num_ch_test =  config['num_ch_test']
	channel_method = config['channel_method']
	noise_method = config['noise_method']
	delay_seed_aug_train = False
	delay_seed_aug_test = False
	keep_orig_train = False
	keep_orig_test = False
	snr_train = config['snr_train']
	snr_test = config['snr_test']
	beta = config['beta']

	#-------------------------------------------------
	# Augmentation CFO parameters
	#-------------------------------------------------
	augment_cfo = experiment_setup['augment_cfo']

	df_aug_train = df_phy_train 
	rand_aug_train = config['rand_aug_train']
	num_aug_train_cfo = config['num_aug_train_cfo']
	keep_orig_train_cfo = config['keep_orig_train_cfo']
	aug_type_cfo = config['aug_type_cfo']

	keep_orig_test_cfo = config["keep_orig_test_cfo"]
	num_aug_test_cfo = config[ "num_aug_test_cfo"]
	df_aug_test = config["df_aug_test"]

	#-------------------------------------------------
	# Residuals
	#-------------------------------------------------

	obtain_residuals = experiment_setup['obtain_residuals']

	#-------------------------------------------------
	# Loading Data
	#-------------------------------------------------

	data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)
	outfile = exp_dir + '/sym-' + data_format + '.npz'

	np_dict = np.load(outfile)
	dict_wifi = {}
	dict_wifi['x_train'] = np_dict['arr_0']
	dict_wifi['y_train'] = np_dict['arr_1']
	dict_wifi['x_test'] = np_dict['arr_2']
	dict_wifi['y_test'] = np_dict['arr_3']
	dict_wifi['fc_train'] = np_dict['arr_4']
	dict_wifi['fc_test'] = np_dict['arr_5']
	dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]

	x_test_orig = dict_wifi['x_test']
	y_test_orig = dict_wifi['y_test']

	data_format += '_{}'.format(architecture)

	num_train = dict_wifi['x_train'].shape[0]
	num_test = dict_wifi['x_test'].shape[0]
	num_classes = dict_wifi['y_train'].shape[1]

	sampling_rate = sample_rate * 1e+6
	fs = sample_rate * 1e+6


	if equalize_train_before or equalize_test_before:
		print('\nEqualization Before')
		print('\tTrain: {}, Test: {}'.format(equalize_train_before, equalize_test_before))

		data_format = data_format + '-eq'

	if equalize_test_before is True:
		dict_wifi, data_format = equalize_channel(dict_wifi = dict_wifi, 
												  sampling_rate = sampling_rate, 
												  data_format = data_format, 
												  verbosity = verbose_test, 
												  which_set = 'x_test')

	#--------------------------------------------------------------------------------------------
	# Physical channel simulation (different days)
	#--------------------------------------------------------------------------------------------
	if add_channel:
		dict_wifi, data_format = physical_layer_channel(dict_wifi = dict_wifi, 
														phy_method = phy_method, 
														channel_type_phy_train = channel_type_phy_train, 
														channel_type_phy_test = channel_type_phy_test, 
														channel_method = channel_method, 
														noise_method = noise_method, 
														seed_phy_train = seed_phy_train, 
														seed_phy_test = seed_phy_test, 
														sampling_rate = sampling_rate, 
														data_format = data_format)

	#--------------------------------------------------------------------------------------------
	# Physical offset simulation (different days)
	#--------------------------------------------------------------------------------------------
	if add_cfo:

		dict_wifi, data_format = physical_layer_cfo(dict_wifi = dict_wifi,
													df_phy_train = df_phy_train,
													df_phy_test = df_phy_test, 
													seed_phy_train_cfo = seed_phy_train_cfo, 
													seed_phy_test_cfo = seed_phy_test_cfo, 
													sampling_rate = sampling_rate, 
													phy_method_cfo = phy_method_cfo, 
													data_format = data_format)
	if remove_cfo:
		data_format = data_format + '[_comp]-'

	#--------------------------------------------------------------------------------------------
	# Physical offset compensation 
	#--------------------------------------------------------------------------------------------
	if testing_setup['remove_test_cfo']:
		dict_wifi, _ = cfo_compansator(dict_wifi = dict_wifi, 
								       sampling_rate = sampling_rate, 
								       data_format = data_format)
		

	#--------------------------------------------------------------------------------------------
	# Equalization
	#--------------------------------------------------------------------------------------------
	if equalize_train or equalize_test:
		print('\nEqualization')
		print('\tTrain: {}, Test: {}'.format(equalize_train, equalize_test))

		data_format = data_format + '-eq'

	if equalize_train is True:
		dict_wifi, data_format = equalize_channel(dict_wifi = dict_wifi, 
												  sampling_rate = sampling_rate, 
												  data_format = data_format, 
												  verbosity = verbose_train, 
												  which_set = 'x_train')

	if equalize_test is True:
		dict_wifi, data_format = equalize_channel(dict_wifi = dict_wifi, 
												  sampling_rate = sampling_rate, 
												  data_format = data_format, 
												  verbosity = verbose_test, 
												  which_set = 'x_test')


	if augment_channel:
		data_format = data_format + '[aug-{}-art-{}-ty-{}-nch-{}-snr-{:.0f}]-'.format(num_aug_train, channel_type_aug_train, aug_type, num_ch_train, snr_train)

	if augment_cfo:
		data_format = data_format + '[augcfo-{}-df-{}-rand-{}-ty-{}-{}-t-]-'.format(num_aug_train_cfo, df_aug_train*1e6, rand_aug_train, aug_type_cfo, keep_orig_train_cfo)


	if obtain_residuals is True:
		print('Residuals are being obtained.')

		dict_wifi, data_format = get_residual(dict_wifi = dict_wifi, 
											  sampling_rate = sampling_rate, 
											  data_format = data_format, 
											  verbosity = verbose_test, 
											  which_set = 'x_test')

	print(data_format)

	# Checkpoint path
	exp_dir += "/CFO_channel_experiments"
	checkpoint = str(exp_dir + '/ckpt-' + data_format)

	if augment_channel is False:
		num_aug_test = 0

	print(checkpoint)

	dict_wifi_no_aug = copy.deepcopy(dict_wifi)

	if testing_setup['augment_test_channel']:

		seed_aug = np.max(seed_phy_train) + seed_phy_test + num_classes + 1

		dict_wifi, data_format = augment_with_channel_test(dict_wifi = dict_wifi, 
														  aug_type = aug_type, 
														  channel_method = channel_method, 
														  num_aug_train = num_aug_train, 
														  num_aug_test = num_aug_test, 
														  keep_orig_train = keep_orig_train, 
														  keep_orig_test = keep_orig_test, 
														  num_ch_train = num_ch_train, 
														  num_ch_test = num_ch_test, 
														  channel_type_aug_train = channel_type_aug_train, 
														  channel_type_aug_test = channel_type_aug_test, 
														  delay_seed_aug_test = delay_seed_aug_test, 
														  snr_test = snr_test, 
														  noise_method = noise_method, 
														  seed_aug = seed_aug, 
														  sampling_rate = sampling_rate,
														  data_format = data_format)

	if testing_setup['augment_test_cfo']:
		
		dict_wifi = augment_with_cfo_test(dict_wifi = dict_wifi,
										  aug_type_cfo = aug_type_cfo,
										  df_aug_test = df_aug_test,
										  num_aug_test_cfo = num_aug_test_cfo, 
										  keep_orig_test_cfo = keep_orig_test_cfo, 
										  rand_aug_test = False, 
										  sampling_rate = sampling_rate)

	print("========================================") 
	print("== BUILDING MODEL... ==")

	checkpoint_in = checkpoint

	if checkpoint_in is None:
		raise ValueError('Cannot test without a checkpoint')
		# data_input = Input(batch_shape=(batch_size, num_features, 2))
		# output, model_name = network_20_2(data_input, num_classes, weight_decay)
		# densenet = Model(data_input, output)

	checkpoint_in = checkpoint_in + '.h5'
	densenet = load_model(checkpoint_in, 
						  custom_objects={'ComplexConv1D':ComplexConv1D,
						  				  'GetAbs': utils.GetAbs,
						  				  'Modrelu': Modrelu})

	batch_size = 100

	num_test_aug = dict_wifi['x_test'].shape[0]

	# probs = densenet.predict(x=x_test, batch_size=batch_size, verbose=0)
	# label_pred = probs.argmax(axis=1) 
	# label_act = y_test.argmax(axis=1) 
	# ind_correct = np.where(label_pred==label_act)[0] 
	# ind_wrong = np.where(label_pred!=label_act)[0] 
	# assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
	# test_acc = 100.*ind_correct.size / num_test

	# acc_class = np.zeros([num_classes])
	# for class_idx in range(num_classes):
	# 	idx_inclass = np.where(label_act==class_idx)[0]
	# 	ind_correct = np.where(label_pred[idx_inclass]==label_act[idx_inclass])[0] 
	# 	acc_class[class_idx] = 100*ind_correct.size / idx_inclass.size

	# print("\n========================================") 
	# print('Test accuracy: {:.2f}%'.format(test_acc))

	# # print(densenet.summary())
	# # for layer in densenet.layers:
	# # 	print(layer.name)
	# # densenet = ...  # create the original model

	# ######################################
	# # Mean and cov_train
	# ######################################

	# x_test_classes = [None]*19

	# for n in range(19):
	# 	ind_n = np.where(y_test.argmax(axis=1)==n)[0]
	# 	x_test_classes[n] = x_test[ind_n]

	# x_test_classes = np.array(x_test_classes)


	# layer_name = densenet.layers[-1].name
	# print(layer_name)
	# model_2 = Model(inputs=densenet.input,
	#                 outputs=densenet.get_layer(layer_name).input)
	# weight, bias = densenet.get_layer(layer_name).get_weights()





	# logits_test = model_2.predict(x=x_test, batch_size=batch_size, verbose=0)
	# logits_test = logits_test.dot(weight) + bias

	output_dict = odict(acc=odict(), comp=odict(), loss=odict())

	if num_test_aug != num_test:
		
		num_test_per_aug = num_test_aug // num_test

		embeddings = densenet.layers[-2].output

		model2 = Model(densenet.input, embeddings)


		logits_test = model2.predict(x=dict_wifi['x_test'],
									 batch_size=batch_size,
								 	 verbose=0)		

		softmax_test = densenet.predict(x=dict_wifi['x_test'],
									 batch_size=batch_size,
								 	 verbose=0)	

		layer_name = densenet.layers[-1].name
		weight, bias = densenet.get_layer(layer_name).get_weights()

		logits_test = logits_test.dot(weight) + bias


		logits_test_new = np.zeros((num_test, num_classes))
		softmax_test_new = np.zeros((num_test, num_classes))
		for i in range(num_test_per_aug):
			# list_x_test.append(x_test_aug[i*num_test:(i+1)*num_test])

			logits_test_new += logits_test[i*num_test:(i+1)*num_test]
			softmax_test_new += softmax_test[i*num_test:(i+1)*num_test]

		# Adding LLRs for num_channel_aug_test test augmentations
		label_pred_llr = logits_test_new.argmax(axis=1)
		label_act = dict_wifi['y_test'][:num_test].argmax(axis=1) 
		ind_correct = np.where(label_pred_llr==label_act)[0] 
		ind_wrong = np.where(label_pred_llr!=label_act)[0] 
		assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
		test_acc_llr = 100.*ind_correct.size / num_test


		# Adding LLRs for num_channel_aug_test test augmentations
		label_pred_soft = softmax_test_new.argmax(axis=1)
		label_act = dict_wifi['y_test'][:num_test].argmax(axis=1) 
		ind_correct = np.where(label_pred_soft==label_act)[0] 
		ind_wrong = np.where(label_pred_soft!=label_act)[0] 
		assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
		test_acc_soft = 100.*ind_correct.size / num_test

		# 1 test augmentation
		probs = densenet.predict(x=dict_wifi['x_test'][:num_test],
								 batch_size=batch_size,
								 verbose=0)
		label_pred = probs.argmax(axis=1)
		ind_correct = np.where(label_pred==label_act)[0] 
		ind_wrong = np.where(label_pred!=label_act)[0] 
		assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
		test_acc = 100.*ind_correct.size / num_test

		# No test augmentations
		probs = densenet.predict(x=dict_wifi_no_aug['x_test'],
								 batch_size=batch_size,
								 verbose=0)
		label_pred = probs.argmax(axis=1)
		label_act = y_test_orig.argmax(axis=1) 
		ind_correct = np.where(label_pred==label_act)[0] 
		ind_wrong = np.where(label_pred!=label_act)[0] 
		assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
		test_acc_no_aug = 100.*ind_correct.size / num_test

		# print("\n========================================") 
		print('Test accuracy (0 aug): {:.2f}%'.format(test_acc_no_aug))
		print('Test accuracy (1 aug): {:.2f}%'.format(test_acc))
		print('Test accuracy ({} aug) llr: {:.2f}%'.format(num_test_per_aug, test_acc_llr))
		print('Test accuracy ({} aug) softmax avg: {:.2f}%'.format(num_test_per_aug, test_acc_soft))



		output_dict['acc']['test_zero_aug'] = test_acc_no_aug
		output_dict['acc']['test_one_aug'] = test_acc
		output_dict['acc']['test_many_aug'] = test_acc_llr
		output_dict['acc']['test_many_aug_soft_avg'] = test_acc_soft

	else:
		probs = densenet.predict(x=dict_wifi['x_test'],
								 batch_size=batch_size,
								 verbose=0)
		label_pred = probs.argmax(axis=1)
		label_act = y_test_orig.argmax(axis=1) 
		ind_correct = np.where(label_pred==label_act)[0] 
		ind_wrong = np.where(label_pred!=label_act)[0] 
		assert (dict_wifi['x_test'].shape[0]== ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
		test_acc_no_aug = 100.*ind_correct.size / dict_wifi['x_test'].shape[0]

		# print("\n========================================") 
		print('Test accuracy (no aug): {:.2f}%'.format(test_acc_no_aug))
		output_dict['acc']['test'] = test_acc_no_aug

	return output_dict, num_test_aug // num_test


if __name__=='__main__':

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

	n_val = 5


	with open('/home/rfml/wifi/scripts/config_cfo_channel.json') as config_file:
	    config = json.load(config_file, encoding='utf-8')

	experiment_setup = {'equalize_train_before': False,
						'equalize_test_before':  False,

						'add_channel':           True,

						'add_cfo':               True,
						'remove_cfo':            True,

						'equalize_train':        False,

						'augment_channel':       True,

						'augment_cfo':           False,

						'obtain_residuals':      False}

	testing_setup = {'augment_test_channel':     True,
					 'augment_test_cfo':         False,
					 'remove_test_cfo':          True,
					 'equalize_test':            False}

	log_name = 'aa_test_'
	if experiment_setup['equalize_train_before']:
		log_name += 'eq_'
	if experiment_setup['add_channel']:
		log_name += 'ch_'
	if experiment_setup['add_cfo']:
		log_name += 'cfo_'
	if experiment_setup['remove_cfo']:
		log_name += 'rmcfo_'
	if experiment_setup['equalize_train']:
		log_name += 'eqtr_'
	# if experiment_setup['equalize_test']:
	# 	log_name += 'eqte_'
	if experiment_setup['augment_channel']:
		log_name += 'augch_type{}_'.format(config['aug_type'])
	if experiment_setup['augment_cfo']:
		log_name += 'augcfo_type{}_'.format(config['aug_type_cfo'])
	if experiment_setup['obtain_residuals']:
		log_name += 'residual_'

	if testing_setup['augment_test_channel']:
		log_name += 'chaugtest_' + str(config["num_aug_test"])
	if testing_setup['augment_test_cfo']:
		log_name += 'cfoaugtest_' + str(config["num_aug_test_cfo"])
	if testing_setup['remove_test_cfo']:
		log_name += 'comptest_'
	if testing_setup['equalize_test']:
		log_name += 'eqtest_'

	
	num_experiments = 5
	for exp_i in range(num_experiments):
		days_multi = [2,5,10,20]
		# days_multi = [10]
		# days_multi = [1]
		# days_multi = [1,5,10,15,20]
		# days_multi = [1, 2, 3]
		# max_seed = (max(days_multi)+1) * 20
		max_seed = 21*20
		

		seed_test = exp_i * max_seed + 60
		exp_list = [1, 2, 3, 4, 5]
		seeds_train_multi = [[exp_i * max_seed + s*20 if exp_i * max_seed + s*20<seed_test else exp_i * max_seed + (s+1)*20 for s in range(days)] for days in days_multi]
		for i in range(len(seeds_train_multi)):
			assert seed_test not in seeds_train_multi[i]

		for ind_train in [20]:
			config["num_aug_train_cfo"] = ind_train
			config["num_aug_train"] = ind_train
			config["num_aug_test_cfo"] = 10
			config["num_aug_test"] = 100

			with open(config['exp_dir'] + "/CFO_channel_experiments/" + log_name + '.txt', 'a+') as f:
				f.write(json.dumps(config))

			for indexx, day_count in enumerate(days_multi):
				test_output, total_aug_test = test_experiments(architecture, config, num_days = day_count, seed_days = seeds_train_multi[indexx], seed_test_day = seed_test, experiment_setup = experiment_setup,
					testing_setup = testing_setup)

				with open(config['exp_dir'] + "/CFO_channel_experiments/" + log_name + '.txt', 'a+') as f:

					f.write('Number of training days: {:}\n'.format(day_count))
					if experiment_setup['augment_cfo']:
						f.write('Number of training cfo aug: {:}\n'.format(config["num_aug_train_cfo"]))
					if experiment_setup['augment_channel']:
						f.write('Number of training channel aug: {:}\n'.format(config["num_aug_train"]))

					f.write('Number of test aug: {:}\n'.format(config["num_aug_test"]))
					f.write('\tExperiment: {:}\n'.format(exp_i + 1))
					f.write('\tSeed train: {:}\n'.format(seeds_train_multi[indexx]))
					f.write('\tSeed test: {:}\n'.format(seed_test))
					f.write('\tNumber of augmentation test: {:}\n'.format(total_aug_test))

					for keys, dicts in test_output.items():
						f.write(str(keys)+':\n')
						for key, value in dicts.items():
							f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')

					if day_count == days_multi[-1]:
						f.write("#------------------------------------------------------------------------------------------#")


	# _ = multiple_day_fingerprint(architecture, config, num_days = 2, seed_days = [20, 40], seed_test_day = 60, experiment_setup = experiment_setup, n_val=n_val)











