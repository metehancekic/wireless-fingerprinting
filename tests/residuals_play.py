'''
Trains data for a WiFi experiment.

Data is read from npz files.
'''

import numpy as np
from timeit import default_timer as timer
import argparse
from tqdm import trange, tqdm
from scipy.fftpack import fft, ifft, fftshift, ifftshift
import json
import os

from .freq_offset import estimate_freq_offset_ 
from ..preproc.preproc_wifi import basic_equalize_preamble
from ..preproc.fading_model import normalize, add_custom_fading_channel, add_freq_offset
from ..cxnn.train_globecom import train_20, train_200
from ..preproc.preproc_wifi import get_residuals_preamble

from ..simulators import signal_power_effect, plot_signals, physical_layer_channel, physical_layer_cfo, cfo_compansator, equalize_channel, augment_with_channel, augment_with_cfo, get_residual

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt


with open(os.path.join('/home/rfml/wifi/scripts', 'config_cfo_channel.json')) as config_file:
    config = json.load(config_file, encoding='utf-8')



def plot_constellation(preamble, title, name_fig):

	signals_directory = "signal_images/"
	if not os.path.exists(signals_directory):
		os.makedirs(signals_directory)

	n_short = 1600
	n_long = 1600

	L = 160
	N = 640

	Stf_1 = fftshift(fft(preamble[n_short-2*N:n_short-N]))
	Stf_2 = fftshift(fft(preamble[n_short-N:n_short]))
	Ltf_1 = fftshift(fft(preamble[n_short+n_long-2*N:n_short+n_long-N]))
	Ltf_2 = fftshift(fft(preamble[n_short+n_long-N:n_short+n_long]))

	ind_guard = np.concatenate((np.arange(-32, -26), np.arange(27, 32))) + (N//2)
	# ind_null = np.concatenate((np.array([0]), np.arange(-(N//2), -32), np.arange(32, (N//2)) )) + (N//2)
	ind_null = np.array([0]) + (N//2)
	ind_pilots = np.array([-21, -7, 7, 21]) + (N//2)

	mask_data = np.ones(N)
	mask_data_pilots = np.ones(N)
	mask_data[list(np.concatenate((ind_guard, ind_null, ind_pilots)))] = 0
	mask_data_pilots[list(np.concatenate((ind_guard, ind_null)))] = 0
	ind_all_all = np.arange(-(N//2), (N//2)) + N//2
	ind_data = ind_all_all[mask_data==1]
	ind_data_pilots = ind_all_all[mask_data_pilots==1]


	preamble_eq = np.concatenate((Stf_1[ind_data_pilots], Stf_2[ind_data_pilots], Ltf_1[ind_data_pilots], Ltf_2[ind_data_pilots]))
	# preamble_eq = Ltf_2[ind_data_pilots]

	# print(Stf_1[ind_data_pilots].shape)
	plt.scatter(preamble_eq.real, preamble_eq.imag)
	plt.grid(True)
	fig_name = os.path.join(signals_directory, name_fig + '.pdf')
	plt.title(title)
	plt.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')

	plt.clf()

def plot_histogram(preamble, title, name_fig):

	signals_directory = "signal_images/"
	if not os.path.exists(signals_directory):
		os.makedirs(signals_directory)


	plt.subplot(2, 1, 1)
	plt.hist(preamble.real, 100)
	plt.subplot(2, 1, 2)
	plt.hist(preamble.imag, 100)
	fig_name = os.path.join(signals_directory, name_fig + '.pdf')
	plt.title(title)
	plt.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')

	plt.clf()





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
preprocess_type = 2
sample_rate = 200

#-------------------------------------------------
# Training configuration
#-------------------------------------------------
epochs = config['epochs']

#-------------------------------------------------
# Physical Channel Parameters
#-------------------------------------------------
add_channel = False

phy_method = config['phy_method']
seed_phy_train = config['seed_phy_train']
seed_phy_test = config['seed_phy_test']
channel_type_phy_train = config['channel_type_phy_train']
channel_type_phy_test = config['channel_type_phy_test']
phy_noise = config['phy_noise']
snr_train_phy = config['snr_train_phy']
snr_test_phy = config['snr_test_phy']

#-------------------------------------------------
# Physical CFO parameters
#-------------------------------------------------

add_cfo = False
remove_cfo = False

phy_method_cfo = config["phy_method_cfo"]
df_phy_train = config['df_phy_train']
df_phy_test = config['df_phy_test']
seed_phy_train_cfo = config['seed_phy_train_cfo']
seed_phy_test_cfo = config['seed_phy_test_cfo']

#-------------------------------------------------
# Equalization params
#-------------------------------------------------
equalize_train = False
equalize_test = False
verbose_train = True
verbose_test = False

#-------------------------------------------------
# Augmentation channel parameters
#-------------------------------------------------
augment_channel = False

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
augment_cfo = False

df_aug_train = df_phy_train 
rand_aug_train = config['rand_aug_train']
num_aug_train_cfo = config['num_aug_train_cfo']
keep_orig_train_cfo = config['keep_orig_train_cfo']
aug_type_cfo = config['aug_type_cfo']

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

num_train = dict_wifi['x_train'].shape[0]
num_test = dict_wifi['x_test'].shape[0]
num_classes = dict_wifi['y_train'].shape[1]

sampling_rate = sample_rate * 1e+6
fs = sample_rate * 1e+6

x_train_orig = dict_wifi['x_train'].copy()
y_train_orig = dict_wifi['y_train'].copy()

x_test_orig = dict_wifi['x_test'].copy()
y_test_orig = dict_wifi['y_test'].copy()

#--------------------------------------------------------------------------------------------
# Physical channel simulation (different days)
#--------------------------------------------------------------------------------------------
add_channel = True
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
add_cfo = False
if add_cfo:
	dict_wifi, data_format = physical_layer_cfo(dict_wifi = dict_wifi,
												df_phy_train = df_phy_train,
												df_phy_test = df_phy_test, 
												seed_phy_train_cfo = seed_phy_train_cfo, 
												seed_phy_test_cfo = seed_phy_test_cfo, 
												sampling_rate = sampling_rate, 
												phy_method_cfo = phy_method_cfo, 
												data_format = data_format)




preamble = dict_wifi['x_test'][0][:,0] + dict_wifi['x_test'][0][:,1]*1j

preamble_orig = x_test_orig[0][:,0] + x_test_orig[0][:,1]*1j

residuals, preamble_constructed = get_residuals_preamble(preamble, fs, method = 'subtraction', channel_method='time')


# plt.figure()
# plt.scatter(np.arange(h_hat.size),np.abs(h_hat))
# plt.title('Channel')

# plt.figure()
# plt.scatter(np.arange(H_hat.size),np.abs(H_hat))
# plt.title('Channel Freq')


plt.figure()
plt.plot(np.abs(preamble_constructed))
plt.title('Constructed preamble')

plt.figure()
plt.plot(np.abs(preamble))
plt.title('Original preamble')

plt.figure()
plt.plot(np.abs(preamble_orig))
plt.title('Source preamble')

plt.figure()
plt.plot(np.abs(residuals))
plt.title('Residuals')

plt.show()



# class_indices = np.zeros([19,200])
# for i in range(19):
# 	class_indices[i,:] = np.where(np.argmax(dict_wifi['y_train'],axis=-1)==i)[0]


# num_transmitters = 3
# num_packets = 3
# histograms = np.zeros([num_transmitters,num_packets,10000])
# for transmitter in trange(num_transmitters):
# 	for packet in range(num_packets):

# 		signal_num = int(class_indices[transmitter,packet])

# 		signal = dict_wifi['x_train'][signal_num][:,0]+1j*dict_wifi['x_train'][signal_num][:,1]
# 		fc = dict_wifi['fc_train'][signal_num]


# 		signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
# 															seed=0, 
# 															beta=0, 
# 															delay_seed=False, 
# 															channel_type=channel_type_phy_train,
# 															channel_method=channel_method,
# 															noise_method=noise_method)
# 		# signal_faded = normalize(signal_faded)

# 		signal_faded_cartesian = np.concatenate((signal_faded.real[..., None], signal_faded.imag[..., None]),axis = -1)


# 		rv_n = np.random.RandomState(seed=0)
# 		signal_faded_cfo_cartesian = add_freq_offset(signal_faded_cartesian.copy().reshape([1,-1,2]), fc=fc, fs=fs, df = rv_n.binomial(n = 1, p=0.5) * 2 * df_phy_train - df_phy_train)[0]

# 		signal_faded_cfo = signal_faded_cfo_cartesian[:,0]+1j*signal_faded_cfo_cartesian[:,1]


# 		signal_faded_eq = basic_equalize_preamble(signal_faded, 
# 											   fs=fs, 
# 											   verbose=False,
# 											   label=' -  Class {}'.format(transmitter))

# 		signal_eq = basic_equalize_preamble(signal, 
# 											   fs=fs, 
# 											   verbose=verbose_train,
# 											   label=' -  Class {}'.format(transmitter))

# 		# signal_faded_eq = normalize(signal_faded_eq)

# 		signal_eq_comp, freq_train = estimate_freq_offset_(signal_eq.copy(), Fs = fs, verbose=False)

# 		signal_faded_cfo_comp, freq_train = estimate_freq_offset_(signal_faded_cfo.copy(), Fs = fs, verbose=False)

# 		signal_faded_cfo_comp_eq = basic_equalize_preamble(signal_faded_cfo_comp, 
# 											   fs=fs, 
# 											   verbose=verbose_train)

# 		signals_directory = "signal_images/"
# 		if not os.path.exists(signals_directory):
# 			os.makedirs(signals_directory)

		
# 		# plt.subplot(num_transmitters, num_packets, transmitter*num_packets + packet+1).set_title("Transmitter {:}".format(transmitter))
# 		# histograms[transmitter, packet,:] = plt.hist(signal.real, bins = 10000 )[0]
# 		# plt.xlim(-0.017, 0.017)
# 		plt.ylim(0, 10)
		
# ipdb.set_trace()

# plt.tight_layout()
# fig_name = os.path.join(signals_directory, "Histogram_comparison_real_raw" + '.pdf')		
# plt.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')

# plt.clf()

# # plot_histogram(signal, 'Signal Histogram', 'histogram_sig')
# # plot_histogram(signal_eq, 'Signal Eq Histogram', 'histogram_sig_eq')

# plot_constellation(signal, 'Received Preamble', 'constellation_map_orig')
# plot_constellation(signal_faded, 'Preamble + Channel', 'constellation_map_ch')
# plot_constellation(signal_faded_cfo, 'Preamble + Channel + CFO', 'constellation_map_ch_cfo')
# plot_constellation(signal_faded_cfo_comp_eq, 'Preamble + Channel + CFO + Compensation + Equalization', 'constellation_map_ch_cfo_comp_eq')
# plot_constellation(signal_faded_eq, 'Preamble + Channel + Equalization', 'constellation_map_ch_eq')
# plot_constellation(signal_eq, 'Preamble +  Equalization', 'constellation_map_eq')
# plot_constellation(signal_eq_comp, 'Preamble + Equalization + Compensation', 'constellation_map_eq_comp')


