'''
Trains data for a WiFi experiment.

Data is read from npz files.
'''

import numpy as np
from timeit import default_timer as timer

# from .cxnn.train_network  import train
# from .cxnn.train_network _aug import train
from .cxnn.train_llr  import train_200 as train

import matplotlib as mpl
from matplotlib import pyplot as plt


mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble = r'\usepackage{amsmath}, \usepackage{sfmath}, \usepackage{amssymb}, \usepackage{bm}, \DeclareMathOperator*{\supp}{supp}, \DeclareMathOperator*{\proj}{\mathcal{P}_K}, \newcommand{\be}{{\bm e}}, \newcommand{\bx}{{\bm x}}, \DeclareMathOperator*{\support}{\mathcal{S}_K}')

exp_dir = '/home/rfml/wifi/experiments/exp19'
# exp_dir = '/home/rfml/wifi/experiments/exp19_S1'
# exp_dir = '/home/rfml/wifi/experiments/exp19_S2'
# exp_dir = '/home/rfml/wifi/experiments/exp100_S1'
# exp_dir = '/home/rfml/wifi/experiments/exp100_S2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3E'

# sample_rate = 20
sample_rate = 200

preprocess_type = 1
sample_duration = 16

noise = False
snr_train = 500
# snr_train = 10
# snr_train = 15
# snr_train = 20

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

# Checkpoint path
checkpoint = exp_dir + '/ckpt-' + data_format + '.h5'

print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
# train_output, model_name, summary, conf_matrix_test = train(dict_wifi, checkpoint_in=None, checkpoint_out=checkpoint)
train_output, model_name, summary = train(dict_wifi, checkpoint_in=None, checkpoint_out=checkpoint)
print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')


# plt.figure(figsize=[13.0, 4.8])
# plt.imshow(conf_matrix_test, vmin=0, vmax=100)
# plt.title('Confusion Matrix')
# plt.colorbar()
# fig_name = 'conf_matrix' + exp_dir[-4:] + '.pdf'
# plt.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')




# Write logs
with open(exp_dir + '/logs-' + data_format  + '.txt', 'a+') as f:
	f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')
	for keys, dicts in train_output.items():
		f.write(str(keys)+':\n')
		for key, value in dicts.items():
			f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
	f.write('\n'+str(summary))

