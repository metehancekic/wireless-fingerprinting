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
import json

from .CFO_channel_experiments import multiple_day_fingerprint

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

						'add_channel':           False,

						'add_cfo':               True,
						'remove_cfo':            False,

						'equalize_train':        False,
						'equalize_test':         False,

						'augment_channel':       False,

						'augment_cfo':           True,

						'obtain_residuals':      True}

	log_name = ''
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
	if experiment_setup['equalize_test']:
		log_name += 'eqte_'
	if experiment_setup['augment_channel']:
		log_name += 'augch_type{}_'.format(config['aug_type'])
	if experiment_setup['augment_cfo']:
		log_name += 'augcfo_type{}_'.format(config['aug_type_cfo'])
	if experiment_setup['obtain_residuals']:
		log_name += 'residual'

	num_experiments = 5
	for exp_i in range(num_experiments):
		days_multi = [1]
		number_aug_cfo = [10,20,30,40,50]
		# days_multi = [1, 2, 3]
		# max_seed = (max(days_multi)+1) * 20
		max_seed = 21*20
		seed_test = exp_i * max_seed + 60
		exp_list = [1, 2, 3, 4, 5]
		seeds_train_multi = [[exp_i * max_seed + s*20 if exp_i * max_seed + s*20<seed_test else exp_i * max_seed + (s+1)*20 for s in range(days)] for days in days_multi]
		for i in range(len(seeds_train_multi)):
			assert seed_test not in seeds_train_multi[i]

		# config["df_aug_train"] = 80e-6

		with open(config['exp_dir'] + "/CFO_channel_experiments/" + log_name + '.txt', 'a+') as f:
			f.write(json.dumps(config))

		for indexx, day_count in enumerate(days_multi):
			for ind_aug in number_aug_cfo:
				config['num_aug_train_cfo'] = ind_aug
				train_output = multiple_day_fingerprint(architecture, config, num_days = day_count, seed_days = seeds_train_multi[indexx], seed_test_day = seed_test, experiment_setup = experiment_setup,
					n_val=n_val)

				with open(config['exp_dir'] + "/CFO_channel_experiments/" + log_name + '.txt', 'a+') as f:

					f.write('Number of training days: {:}\n'.format(day_count))
					f.write('Number of augs: {:}\n'.format(ind_aug))
					if experiment_setup['obtain_residuals']:
						f.write('Residuals obtained')
					f.write('\tExperiment: {:}\n'.format(exp_i + 1))
					f.write('\tSeed train: {:}\n'.format(seeds_train_multi[indexx]))
					f.write('\tSeed test: {:}\n'.format(seed_test))

					for keys, dicts in train_output.items():
						f.write(str(keys)+':\n')
						for key, value in dicts.items():
							f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')

					if day_count == days_multi[-1]:
						f.write("#------------------------------------------------------------------------------------------#")


	# _ = multiple_day_fingerprint(architecture, config, num_days = 2, seed_days = [20, 40], seed_test_day = 60, experiment_setup = experiment_setup, n_val=n_val)


