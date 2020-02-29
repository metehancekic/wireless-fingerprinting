'''
Compares real and artifical channels
'''
import numpy as np
from timeit import default_timer as timer
from scipy.fftpack import fft, ifft, fftshift, ifftshift
from scipy.signal import resample
from scipy.linalg import circulant, toeplitz
import matplotlib as mpl
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..preproc.preproc_wifi import rms
from ..preproc.fading_model import normalize, add_custom_fading_channel, add_fading_channel, add_noise


def plot_channel(frame, label=' '):
	'''
	LTI channel estimation, assuming delay spread <= length of cyclic prefix.
	'''
	n_short = 160
	n_long = 160
	L = 16
	N = 64

	# Received ltf
	ltf_rx = frame[n_short + 2*L - np.int(L/2):n_short + 2*L+N - np.int(L/2)]
	Ltf_rx = fftshift(fft(ltf_rx))

	# Actual ltf
	Ltf = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
	ltf = ifft(ifftshift(Ltf))

	# ------------------------------------------------------------------------
	# Coarse estimation
	# ------------------------------------------------------------------------

	H_hat = Ltf_rx*Ltf
	h_hat = ifft(ifftshift(H_hat))
	h_hat[L+1:] = 0
	H_hat = fftshift(fft(h_hat))

	freq = np.arange(-32, 32)
	plt.figure(figsize=[10, 3])
	plt.subplot(1,2,1)
	plt.stem(freq, np.abs(H_hat))
	plt.grid(True)
	plt.title('Magnitude')
	plt.xlabel('Frequency bin')
	plt.subplot(1,2,2)
	# plt.stem(freq, np.unwrap(np.angle(H_hat)))
	plt.stem(freq, np.angle(H_hat))
	plt.title('Phase')
	plt.xlabel('Frequency bin')
	plt.suptitle(label + '\n Coarse estimation + smoothing')
	plt.grid(True)
	plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.93])

	plt.figure(figsize=[10, 3])
	plt.subplot(1,2,1)
	plt.stem(np.abs(h_hat))
	plt.title('Magnitude')
	plt.xlabel('Time (in samples)')
	plt.grid(True)
	plt.subplot(1,2,2)
	# plt.stem(np.unwrap(np.angle(h_hat)))
	plt.stem(np.angle(h_hat))
	plt.title('Phase')
	plt.xlabel('Time (in samples)')
	plt.grid(True)
	plt.suptitle(label + '\n Coarse estimation + smoothing')
	plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.93])


	# ------------------------------------------------------------------------
	# Least squares estimation in frequency domain (no guard band)
	# ------------------------------------------------------------------------
	AA = np.zeros((N, N)) + 0j
	for m in range(N):
		for n in range(L+1):
			AA[m, n] = Ltf[m] * np.exp(-1j*2*np.pi*m*n/N)
	A = AA[:, :L+1] * np.exp(1j*np.pi*np.arange(L+1)).reshape(1, -1)
	ind_all = np.arange(-32, 32) + 32
	ind_guard = np.concatenate((np.arange(-32, -26), np.arange(27, 32))) + 32
	ind_null = np.array([0]) + 32
	mask_data_pilots = np.ones(64)
	mask_data_pilots[list(np.concatenate((ind_guard, ind_null)))] = 0
	ind_data_pilots = ind_all[mask_data_pilots==1]

	cond = np.linalg.cond(A[ind_data_pilots,:])
	print('Condition number = {}'.format(cond))
	h_hat_small, residuals, rank, singular_values = np.linalg.lstsq(A[ind_data_pilots,:], Ltf_rx[ind_data_pilots], rcond=None)
	h_hat = np.zeros(N)+0j
	h_hat[:L+1] = h_hat_small
	# h_hat = np.roll(h_hat, -np.int(L/2))
	H_hat = fftshift(fft(h_hat))

	freq = np.arange(-32, 32)
	plt.figure(figsize=[10, 3])
	plt.subplot(1,2,1)
	plt.stem(freq, np.abs(H_hat))
	plt.grid(True)
	plt.title('Magnitude')
	plt.xlabel('Frequency bin')
	plt.subplot(1,2,2)
	# plt.stem(freq, np.unwrap(np.angle(H_hat)))
	plt.stem(freq, np.angle(H_hat))
	plt.title('Phase')
	plt.xlabel('Frequency bin')
	plt.suptitle(label + '\n Frequency domain least squares estimation (without guard band)')
	plt.grid(True)
	plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.9])

	plt.figure(figsize=[10, 3])
	plt.subplot(1,2,1)
	plt.stem(np.abs(h_hat))
	plt.title('Magnitude')
	plt.xlabel('Time (in samples)')
	plt.grid(True)
	plt.subplot(1,2,2)
	# plt.stem(np.unwrap(np.angle(h_hat)))
	plt.stem(np.angle(h_hat))
	plt.title('Phase')
	plt.xlabel('Time (in samples)')
	plt.grid(True)
	plt.suptitle(label + '\n Frequency domain least squares estimation (without guard band)')
	plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.9])

	# ------------------------------------------------------------------------
	# Least squares estimation in frequency domain (all subcarriers)
	# ------------------------------------------------------------------------

	h_hat_small, residuals, rank, singular_values = np.linalg.lstsq(A, Ltf_rx, rcond=None)
	h_hat = np.zeros(N)+0j
	h_hat[:L+1] = h_hat_small
	# h_hat = np.roll(h_hat, -np.int(L/2))
	H_hat = fftshift(fft(h_hat))

	freq = np.arange(-32, 32)
	plt.figure(figsize=[10, 3])
	plt.subplot(1,2,1)
	plt.stem(freq, np.abs(H_hat))
	plt.grid(True)
	plt.title('Magnitude')
	plt.xlabel('Frequency bin')
	plt.subplot(1,2,2)
	# plt.stem(freq, np.unwrap(np.angle(H_hat)))
	plt.stem(freq, np.angle(H_hat))
	plt.title('Phase')
	plt.xlabel('Frequency bin')
	plt.suptitle(label + '\n Frequency domain least squares estimation (all subcarriers)')
	plt.grid(True)
	plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.9])

	plt.figure(figsize=[10, 3])
	plt.subplot(1,2,1)
	plt.stem(np.abs(h_hat))
	plt.title('Magnitude')
	plt.xlabel('Time (in samples)')
	plt.grid(True)
	plt.subplot(1,2,2)
	# plt.stem(np.unwrap(np.angle(h_hat)))
	plt.stem(np.angle(h_hat))
	plt.title('Phase')
	plt.xlabel('Time (in samples)')
	plt.grid(True)
	plt.suptitle(label + '\n Frequency domain least squares estimation (all subcarriers)')
	plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.9])

	# ------------------------------------------------------------------------
	# Least squares estimation in time domain
	# ------------------------------------------------------------------------
	A_time = circulant(ltf)[:,:L]
	h_hat, residuals, rank, singular_values = np.linalg.lstsq(A_time, ltf_rx, rcond=None)
	h_hat = np.concatenate((h_hat, np.zeros(N-L)+0j))
	H_hat = fftshift(fft(h_hat))

	freq = np.arange(-32, 32)
	plt.figure(figsize=[10, 3])
	plt.subplot(1,2,1)
	plt.stem(freq, np.abs(H_hat))
	plt.grid(True)
	plt.title('Magnitude')
	plt.xlabel('Frequency bin')
	plt.subplot(1,2,2)
	# plt.stem(freq, np.unwrap(np.angle(H_hat)))
	plt.stem(freq, np.angle(H_hat))
	plt.title('Phase')
	plt.xlabel('Frequency bin')
	plt.suptitle(label + '\n Time domain least squares estimation')
	plt.grid(True)
	plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.9])

	plt.figure(figsize=[10, 3])
	plt.subplot(1,2,1)
	plt.stem(np.abs(h_hat))
	plt.title('Magnitude')
	plt.xlabel('Time (in samples)')
	plt.grid(True)
	plt.subplot(1,2,2)
	# plt.stem(np.unwrap(np.angle(h_hat)))
	plt.stem(np.angle(h_hat))
	plt.title('Phase')
	plt.xlabel('Time (in samples)')
	plt.grid(True)
	plt.suptitle(label + '\n Time domain least squares estimation')
	plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.9])

# mpl.rc('text', usetex=True)
# mpl.rc('text.latex', preamble = r'\usepackage{amsmath}, \usepackage{sfmath}, \usepackage{amssymb}, \usepackage{bm}, \DeclareMathOperator*{\supp}{supp}, \DeclareMathOperator*{\proj}{\mathcal{P}_K}, \newcommand{\be}{{\bm e}}, \newcommand{\bx}{{\bm x}}, \DeclareMathOperator*{\support}{\mathcal{S}_K}')

exp_dir = '/home/rfml/wifi/experiments/exp19'
# exp_dir = '/home/rfml/wifi/experiments/exp19_S1'
# exp_dir = '/home/rfml/wifi/experiments/exp19_S2'
# exp_dir = '/home/rfml/wifi/experiments/exp100_S1'
# exp_dir = '/home/rfml/wifi/experiments/exp100_S2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2'
# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2'

########################################################
# Clean data
########################################################
sample_duration = 16
preprocess_type = 2
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

i = 1

frame = dict_wifi['x_train'][i][:,0] + 1j*dict_wifi['x_train'][i][:,1]
# plot_channel(frame, label='Before aug, train data ')
plot_channel(frame, label='')

# frame = dict_wifi['x_test'][i][:,0] + 1j*dict_wifi['x_test'][i][:,1]
# plot_channel(frame, label='Before aug, test data ')

channel = False
# channel = True

if channel is True:
	########################################################
	#  Data with channel
	########################################################

	# aug_type:
	# 0 - usual channel aug
	# 1 - same channel for ith example in each class
	aug_type = 1

	num_ch_train = 1
	num_ch_test = 1

	aug_train = 1

	snr_train = 500
	snr_test = 500

	# snr_trains = [20]
	# snr_tests = [25]

	# from IPython import embed; embed()
	# ipdb.set_trace()

	sampling_rate = sample_rate * 1e+6

	x_train = dict_wifi['x_train'].copy()
	y_train = dict_wifi['y_train'].copy()

	x_train_aug = x_train.copy()
	y_train_aug = y_train.copy()

	channel_dict = {}
	for i in range(50):
		channel_dict[i]=0

	num_train = x_train.shape[0]
	if num_ch_train < -1:
		raise ValueError('num_ch_train')
	elif num_ch_train != 0:
		signal_ch = np.zeros(x_train.shape)
		for i in tqdm(range(num_train)):
			signal = x_train[i][:,0]+1j*x_train[i][:,1]
			if num_ch_train==-1:
				signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, seed=None, beta=0)
			elif aug_type==1:
				signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, seed=channel_dict[np.argmax(y_train[i])], beta=0)
				channel_dict[np.argmax(y_train[i])] += 1
			elif aug_type==0:
				signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate, seed = k * num_ch_train + (i % num_ch_train) , beta=0)

			signal_faded = normalize(signal_faded)
			signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))

		x_train_aug = signal_ch

	# 		x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
	# 		y_train_aug = np.concatenate((y_train_aug, y_train), axis=0)


	# dict_wifi['x_train'] = x_train_aug
	# dict_wifi['y_train'] = y_train_aug

	num_test = dict_wifi['x_test'].shape[0]
	if num_ch_test < -1:
		raise ValueError('num_ch_test')
	elif num_ch_test!=0:
		for i in tqdm(range(num_test)):
			signal = dict_wifi['x_test'][i][:,0]+1j*dict_wifi['x_test'][i][:,1]
			if num_ch_test==-1:
				signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, seed=None, beta=0)
			else:
				signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate, seed = num_train + (i % num_ch_test) , beta=0)
			
			signal_faded = normalize(signal_faded)
			signal_ch = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1)
			dict_wifi['x_test'][i] = signal_ch

	i = 2

	frame = x_train_aug[i][:,0] + 1j*x_train_aug[i][:,1]
	plot_channel(frame, label='After aug, train data ')

	frame = dict_wifi['x_test'][i][:,0] + 1j*dict_wifi['x_test'][i][:,1]
	plot_channel(frame, label='After aug, test data ')


	# rv_channel = random.RandomState(seed=seed)
	# # rv_noise = random.RandomState(seed=None)
	# # epa_delay = np.array([0, 30, 70, 90, 110, 190, 410])
	# # epa_power = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])
	# # h_epa = rv_channel.normal(loc=0, scale=1, size=(7, 2)).dot(np.array([1, 1j]))
	# # h_epa *= np.sqrt((10**(epa_power/10))/2)

	# delay_ns = np.array([0, 30, 70, 90, 110, 190, 410])
	# power_dB = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])
	# # A0 = (np.random.randn(len(self.power_dB)) + 1j*np.random.randn(len(self.power_dB)))/np.sqrt(2)
	# # A = (10**(np.array(np.array(power_dB)/20)))*A0

	# A = rv_channel.normal(loc=0, scale=1, size=(len(power_dB), 2)).dot(np.array([1, 1j]))
	# A *= np.sqrt((10**(power_dB/10))/2)


plt.show()