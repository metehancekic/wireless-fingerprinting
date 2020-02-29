'''
Trains data for a WiFi experiment.

Data is read from npz files.
'''

import numpy as np
import argparse
import json
import os

from .fingerprint_cfo_channel import multiple_day_fingerprint

# from freq_offset import estimate_freq_offset !!


with open('/home/rfml/wifi/scripts/config_cfo_channel_multi.json') as config_file:
    config = json.load(config_file, encoding='utf-8')

num_days = [1,2,3,4,5,6,7,8,9,10,15,20]


for days in range(len(num_days)):
	for experiment_i in range(5):

		seed_days = [0 + experiment_i*400,
	 			  	 [0 + experiment_i*400, 20 + experiment_i*400],
	 			  	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400],
				  	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400],
				  	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400],
	  			  	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400],
				  	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400, 140 + experiment_i*400],
				  	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400, 140 + experiment_i*400, 160 + experiment_i*400],
	             	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400, 140 + experiment_i*400, 160 + experiment_i*400, 180 + experiment_i*400],
				 	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400, 140 + experiment_i*400, 160 + experiment_i*400, 180 + experiment_i*400, 200 + experiment_i*400],
				 	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400, 140 + experiment_i*400, 160 + experiment_i*400, 180 + experiment_i*400, 200 + experiment_i*400, 220 + experiment_i*400, 240 + experiment_i*400, 260 + experiment_i*400, 280 + experiment_i*400, 300 + experiment_i*400],
				 	 [0 + experiment_i*400, 20 + experiment_i*400, 40 + experiment_i*400, 80 + experiment_i*400, 100 + experiment_i*400, 120 + experiment_i*400, 140 + experiment_i*400, 160 + experiment_i*400, 180 + experiment_i*400, 200 + experiment_i*400, 220 + experiment_i*400, 240 + experiment_i*400, 260 + experiment_i*400, 280 + experiment_i*400, 300 + experiment_i*400, 320 + experiment_i*400, 340 + experiment_i*400, 360 + experiment_i*400, 380 + experiment_i*400, 400 + experiment_i*400]]

		train_output = multiple_day_fingerprint(architecture='modrelu', config=config, num_days=num_days[days], seed_days=seed_days[days])

		
		with open(config['exp_dir'] + '/logs-' + 'multi_days_cfo_comp'  + '.txt', 'a+') as f:
			f.write('\n\n----------{:}-days------------\n'.format(days)+'\n\n')
			f.write('\n\nSeed : {:}\n'.format(seed_days[days])+'\n\n')
			f.write('\n\nExperiment : {:}\n'.format(experiment_i+1)+'\n\n')
			for keys, dicts in train_output.items():
				f.write(str(keys)+':\n')
				for key, value in dicts.items():
					f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')


