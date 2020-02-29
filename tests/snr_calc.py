'''
Uses freq offset as a feature
'''
import numpy as np
from tqdm import tqdm
from tqdm import trange
from timeit import default_timer as timer
from scipy.fftpack import fft, ifft, fftshift, ifftshift
from scipy.signal import resample
from scipy.linalg import circulant, toeplitz

from ..preproc.preproc_wifi import rms
from ..preproc.fading_model import normalize, add_custom_fading_channel, add_fading_channel, add_noise

import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rc('text', usetex=True)
# mpl.rc('text.latex', preamble = r'\usepackage{amsmath}, \usepackage{sfmath}, \usepackage{amssymb}, \usepackage{bm}, \DeclareMathOperator*{\supp}{supp}, \DeclareMathOperator*{\proj}{\mathcal{P}_K}, \newcommand{\be}{{\bm e}}, \newcommand{\bx}{{\bm x}}, \DeclareMathOperator*{\support}{\mathcal{S}_K}')

def estimate_snr(complex_signal, verbose=False):
	"""
	Function to read WiFi data from file and do the following pre-processing operations
	 	1. Filter out out-of-band noise
	 	2. Subsample to 20 Megasamples/sec (which is the standard OFDM sampling rate)
	 	3. Detect beginning of preamble (i.e. remove guard band)
	 	4. Remove frequency offsets
	This works best when the original sampling rate is an integer multiple of 20 Msps. 

	Inputs:
		data_file	 - Path to file with extension .sigmf-data
		meta_file	 - Path to file with extension .sigmf-meta
		frame_index	 - Frame index (an integer in the range [0, total no. of packets in file - 1])
					   e.g., frame_index=0 corresponds to the first packet in the file.

	Output:
		frame 		- Frame (array of complex numbers) containing effects of channel 
					  and Tx nonlinearities
		freq_offset - dict with info about frequency offset
	"""

	Fs = 20e6
	n_short = 160
	n_long = 160

	# # ----------------------------------------------------
	# # Filter out-of-band noise
	# # ----------------------------------------------------

	# N = complex_signal.shape[0]
	# if N % 2 != 0:
	# 	complex_signal = complex_signal[:-1]
	# 	N -= 1
	# low_ind = np.int((lowFreq-Fc)*(N/Fs) + N/2) 
	# up_ind = np.int((upFreq-Fc)*(N/Fs) + N/2) 
	# lag = np.int(( -Fc + (lowFreq+upFreq)/2 )*(N/Fs) + N/2) - np.int(N/2)
	# X = fftshift(fft(complex_signal))
	# X[:low_ind] = 0 + 0j
	# X[up_ind:] = 0 + 0j
	# X = np.roll(X, -lag)
	# complex_signal = ifft(ifftshift(X))

	# ----------------------------------------------------
	# Energy detection
	# ----------------------------------------------------

	n_win = 16
	lag = n_win
	a = np.zeros((complex_signal.size-n_win-lag, 1))
	b = np.zeros((complex_signal.size-n_win-lag, 1))
	for n in range(complex_signal.size-n_win-lag):
		sig1 = complex_signal[n:n+n_win].reshape(1,-1)
		sig2 = complex_signal[n+lag:n+n_win+lag].conj().reshape(1,-1)
		a[n] = (np.abs(sig1)**2).sum()
		b[n] = (np.abs(sig2)**2).sum()
	energy = b/a

	snr = 10*np.log10(energy[energy.argmax()][0]-1)
	# print('SNR = {:.2f} dB'.format(10*np.log10(energy[energy.argmax()][0]-1)))

	return snr


# exp_dir = '/home/rfml/wifi/experiments/exp19'
# exp_dir = '/home/rfml/wifi/experiments/exp19_S1'
# exp_dir = '/home/rfml/wifi/experiments/exp19_S2'
# exp_dir = '/home/rfml/wifi/experiments/exp100_S1'
# exp_dir = '/home/rfml/wifi/experiments/exp100_S2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2'
exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2'

sample_duration = 16
preprocess_type = 1
sample_rate = 20

file_name_1 = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

outfile_1 = exp_dir + '/sym-' + file_name_1 + '.npz'
np_dict_1 = np.load(outfile_1)
dict_wifi = {}
dict_wifi['x_train'] = np_dict_1['arr_0']
dict_wifi['y_train'] = np_dict_1['arr_1']
dict_wifi['x_test'] = np_dict_1['arr_2']
dict_wifi['y_test'] = np_dict_1['arr_3']
dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]

x_train = dict_wifi['x_train'].copy()
y_train = dict_wifi['y_train'].copy()
x_test = dict_wifi['x_test'].copy()
y_test = dict_wifi['y_test'].copy()
num_train = x_train.shape[0]
num_test = x_test.shape[0]

complex_train = x_train[..., 0] + 1j* x_train[..., 1]
complex_test = x_test[..., 0] + 1j* x_test[..., 1]

snr_train = np.zeros([num_train])
snr_test = np.zeros([num_test])

# for i in trange(num_train):
# # for i in range(2):
# 	snr_train[i] = estimate_snr(complex_train[i], verbose=False)

for i in trange(num_test):
# for i in range(2):
	snr_test[i] = estimate_snr(complex_test[i], verbose=False)

num_classes_plot = 20

plt.figure(figsize=[15, 6])
num_rows = 4
num_cols = 5
for i in range(num_classes_plot):
	plt.subplot(num_rows, num_cols, i+1)
	ind_n = np.where(y_train.argmax(axis=1)==i)[0]
	plt.hist(snr_train[ind_n], density=True, bins=25)
	# plt.axvline(thresh[i], color='r', linestyle='dashed', linewidth=1)
	plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
	plt.title('Class {}'.format(i+1), fontsize=12)
plt.suptitle('Train SNR histogram')	

plt.figure(figsize=[15, 6])
num_rows = 4
num_cols = 5
for i in range(num_classes_plot):
	plt.subplot(num_rows, num_cols, i+1)
	ind_n = np.where(y_test.argmax(axis=1)==i)[0]
	plt.hist(snr_test[ind_n], density=True, bins=25)
	# plt.axvline(thresh[i], color='r', linestyle='dashed', linewidth=1)
	plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
	plt.title('Class {}'.format(i+1), fontsize=12)
plt.suptitle('Test SNR histogram')	


