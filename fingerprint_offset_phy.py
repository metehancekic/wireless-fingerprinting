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
# from .cxnn.train_network _aug import train
# from .cxnn.train_llr  import train
from .cxnn.train_llr  import train_200 as train

from tqdm import tqdm, trange
import ipdb


exp_dirs = []
exp_dirs += ['/home/rfml/wifi/experiments/exp19']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2']
# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2']

preprocess_type = 1
sample_rate = 200
sample_duration = 16

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

# num_aug_train = 0
# num_aug_train = 5
num_aug_train = 20

keep_orig_train = False
# keep_orig_train = True

'''
aug_type:
	0 - usual offset aug
	1 - same offset for ith example in each class
'''
# aug_type = 0
aug_type = 1

data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

exp_dir = exp_dirs[0]

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

num_train = dict_wifi['x_train'].shape[0]
num_test = dict_wifi['x_test'].shape[0]
fc_train = dict_wifi['fc_train']
fc_test = dict_wifi['fc_test']

fs = sample_rate * 1e+6

x_train_orig = dict_wifi['x_train'].copy()
y_train_orig = dict_wifi['y_train'].copy()
num_classes = y_train_orig.shape[1]

x_test_orig = dict_wifi['x_test'].copy()
y_test_orig = dict_wifi['y_test'].copy()

for seed_phy_train, seed_phy_test in seed_phy_pairs:

	#--------------------------------------------------------------------------------------------
	# Physical offset simulation (different days)
	#--------------------------------------------------------------------------------------------
	signal_ch = x_train_orig.copy()
	for n in trange(num_classes):
		ind_n = np.where(y_train_orig.argmax(axis=1)==n)[0]
		seed_phy_train_n = seed_phy_train + n
		for i in ind_n:
			rv_n = np.random.RandomState(seed=seed_phy_train_n)
			signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																 df = rv_n.uniform(low=-df_phy_train, high=df_phy_train),
																 fc = fc_train[i:i+1], 
																 fs = fs)
	dict_wifi['x_train'] = signal_ch.copy()


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


	#--------------------------------------------------------------------------------------------
	# Offset augmentation
	#--------------------------------------------------------------------------------------------

	x_train = dict_wifi['x_train'].copy()
	y_train = dict_wifi['y_train'].copy()

	x_train_aug = x_train.copy()
	y_train_aug = y_train.copy()

	if aug_type == 0:
		for k in tqdm(range(num_aug_train)):
			signal_ch = x_train.copy()
			signal_ch = add_freq_offset(signal_ch, rand = rand_aug_train,
												   df = df_aug_train,
												   fc = fc_train, 
												   fs = fs)
			if keep_orig_train is False:
				if k==0:
					x_train_aug = signal_ch.copy()
					y_train_aug = y_train.copy()
				else:
					x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
					y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)
			else:
				x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
				y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)		
	elif aug_type == 1:
		offset_dict = {}
		for i in range(401):
			offset_dict[i] = seed_phy_train+seed_phy_test+num_classes+1			
		for k in tqdm(range(num_aug_train)):
			signal_ch = x_train.copy()
			for i in tqdm(range(num_train)):
				rv_n = np.random.RandomState(seed=offset_dict[np.argmax(y_train[i])])
				if rand_aug_train=='unif':
					df_n = rv_n.uniform(low=-df_aug_train, high=df_aug_train)
				elif rand_aug_train=='ber':
					df_n = rv_n.choice(a=np.array([-df_aug_train, df_aug_train]))
				signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																	 df = df_n,
																	 fc = fc_train[i:i+1], 
																	 fs = fs)
				offset_dict[np.argmax(y_train[i])] += 1
			if keep_orig_train is False:
				if k==0:
					x_train_aug = signal_ch.copy()
					y_train_aug = y_train.copy()
				else:
					x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
					y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)
			else:
				x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
				y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)			


	dict_wifi['x_train'] = x_train_aug.copy()
	dict_wifi['y_train'] = y_train_aug.copy()


	data_format = 'offset-phy-{}-s-{}-aug-{}-df-{}-rand-{}-ty-{}-{}-t-'.format(df_phy_train*1e6, seed_phy_train, num_aug_train, df_aug_train*1e6, rand_aug_train, aug_type, keep_orig_train)
	data_format += '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

	# Checkpoint path
	checkpoint = exp_dirs[0] + '/ckpt-' + data_format +'.h5'

	end = timer()
	print('Load time: {:} sec'.format(end - start))

	print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
	train_output, model_name, summary = train(dict_wifi, checkpoint_in=None, checkpoint_out=checkpoint)
	print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

	# Write logs
	with open(exp_dir + '/logs-' + data_format  + '.txt', 'a+') as f:
		f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')
		f.write('Freq offset augmentation')
		f.write('Physical offsets: Train: {}, Test:{} ppm\n'.format(df_phy_train*1e6, df_aug_train*1e6))
		f.write('Physical seeds: Train: {}, Test:{}\n'.format(seed_phy_train, seed_phy_test))
		f.write('Augmentations: {}, Keep orig train: {} \n'.format(num_aug_train, keep_orig_train))
		f.write('Augmentation offset: Train: {}, {} ppm\n'.format(rand_aug_train, df_aug_train*1e6))
		for keys, dicts in train_output.items():
			f.write(str(keys)+':\n')
			for key, value in dicts.items():
				f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
		f.write('\n'+str(summary))