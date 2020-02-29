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
from .cxnn.train_network _aug import train

from tqdm import tqdm
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

# Offset for testing
df_test = 20e-6 # 20 ppm
# rand_test = 'unif' # Random uniform offset
rand_test = 'ber' # Random bernoulli offset
# rand_test = False # Fixed offset of -df_test

df_train = 20e-6 
rand_train = 'unif'
# rand_train = 'ber'
# rand_train = 'False'

# aug_train = 0
# aug_train = 5
aug_train = 20
keep_orig = False
# keep_orig = True


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

#--------------------------------------------------------------------------------------------
# Adding offset to test
#--------------------------------------------------------------------------------------------

signal_ch = dict_wifi['x_test'].copy()
signal_ch = add_freq_offset(signal_ch, rand = rand_test,
									   df = df_test,
									   fc = fc_test, 
									   fs = fs)
dict_wifi['x_test'] = signal_ch.copy()


#--------------------------------------------------------------------------------------------
# Train offset augmentation
#--------------------------------------------------------------------------------------------

x_train = dict_wifi['x_train'].copy()
y_train = dict_wifi['y_train'].copy()

x_train_aug = x_train.copy()
y_train_aug = y_train.copy()

for k in tqdm(range(aug_train)):
	signal_ch = x_train.copy()
	signal_ch = add_freq_offset(signal_ch, rand = rand_train,
										   df = df_train,
										   fc = fc_train, 
										   fs = fs)
	if keep_orig is False:
		if k==0:
			x_train_aug = signal_ch.copy()
			y_train_aug = y_train.copy()
		else:
			x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
			y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)
	else:
		x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
		y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)					


dict_wifi['x_train'] = x_train_aug
dict_wifi['y_train'] = y_train_aug

data_format = 'offset-{}-{}-'.format(aug_train, rand_train)
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
	f.write('Augmentations: {}, Keep_Orig: {} \n'.format(aug_train, keep_orig))
	f.write('Train offset: {}, {} ppm\n'.format(rand_train, df_train*1e6))
	f.write('Test offset: {}, {} ppm\n'.format(rand_test, df_test*1e6))
	for keys, dicts in train_output.items():
		f.write(str(keys)+':\n')
		for key, value in dicts.items():
			f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
	f.write('\n'+str(summary))