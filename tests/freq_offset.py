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
import os
import matplotlib as mpl
from matplotlib import pyplot as plt

from ..preproc.preproc_wifi import rms
from ..preproc.fading_model import normalize, add_custom_fading_channel, add_freq_offset


mpl.use('Agg')
mpl.rc('text', usetex=True)
# mpl.rc('text.latex', preamble = r'\usepackage{amsmath}, \usepackage{sfmath}, \usepackage{amssymb}, \usepackage{bm}, \DeclareMathOperator*{\supp}{supp}, \DeclareMathOperator*{\proj}{\mathcal{P}_K}, \newcommand{\be}{{\bm e}}, \newcommand{\bx}{{\bm x}}, \DeclareMathOperator*{\support}{\mathcal{S}_K}')

def estimate_freq_offset(frame, verbose=False):
	"""
	Function to read WiFi data from file and do the following pre-processing operations
	 	1. Filter out out
	 	-of-band noise
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

	# ----------------------------------------------------
	# Coarse frequency offset correction (using STF)
	# ----------------------------------------------------

	sig3 = frame[n_short//2 : n_short-16].conj().copy()
	sig4 = frame[n_short//2 + 16 : n_short].copy()
	df1 = 1/16.0 * np.angle(sig3.dot(sig4.T))
	frame *= np.exp(-1j*np.arange(0, frame.size) *df1).flatten()

	# if verbose==True:
	# 	print('Coarse freq offset = {:.2f} KHz'.format(df1* 2e8 / (2*np.pi*1e3)))

	# ----------------------------------------------------
	# Fine frame alignment (using LTF cross-correlation)
	# ----------------------------------------------------

	Ltf = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
	ltf = ifft(ifftshift(Ltf))
	var_ltf = np.sum(np.abs(ltf)**2)

	guard_band = 0

	# search_length_ltf = frame.size - 128
	# corr_ltf = np.zeros(search_length_ltf)
	# for n in range(search_length_ltf):	
	# 	corr_ltf[n] = np.abs(frame[n:n+64].dot(ltf.conj().T))
	# 	corr_ltf[n] /= np.sqrt(np.sum(np.abs(frame[n:n+64])**2)*var_ltf)
	# ind_ltf_1 = corr_ltf.argmax()
	# ind_ltf_2 = np.concatenate((corr_ltf[:ind_ltf_1-60], np.zeros(120), corr_ltf[ind_ltf_1+60:])).argmax()
	# if (np.abs(ind_ltf_1 - ind_ltf_2) > 60) and (np.abs(ind_ltf_1 - ind_ltf_2) < 70):
	# 	ind_ltf = max(ind_ltf_1, ind_ltf_2)
	# 	frame_start = ind_ltf.max() + 64 - n_long - n_short
	# 	if frame_start <= 0:
	# 		frame_start = guard_band # Use guess from guard band size
	# else:
	# 	frame_start = guard_band # Use guess from guard band size

	# frame = frame[frame_start:]
	frame_length = frame.shape[0]

	# ----------------------------------------------------
	# Fine frequency offset correction (using LTF)
	# ----------------------------------------------------

	sig5 = frame[n_short+32:n_short+32+64].conj().copy()
	sig6 = frame[n_short+64+32:n_short+n_long].copy()
	df2 = 1/64.0 * np.angle(sig5.dot(sig6.T))
	frame *= np.exp(-1j*np.arange(0,frame_length)*df2).flatten()

	# if df1<0:
	# 	df1 += 2*np.pi/16.0
	# if df2<0:
	# 	df2 += 2*np.pi/64.0

	df1 /= (2*np.pi)
	df2 /= (2*np.pi)

	df1 *= 100
	df2 *= 100

	if verbose==True:
		# print('Coarse freq offset = {:.2f} KHz'.format(df1 * Fs / (2*np.pi*1e3)))
		# print('Fine freq offset = {:.2f} KHz'.format(df2 * Fs / (2*np.pi*1e3)))
		print('Coarse freq offset = {:.3f} %'.format(df1))
		print('Fine freq offset = {:.3f} %'.format(df2))

	# # -------------------------------------------------------------
	# # Sampling frequency offset correction
	# # This is done in frequency domain, using pilot subcarriers
	# # -------------------------------------------------------------

	# epsilon_0 = (df1 + df2) / (2*np.pi*Fc)
	# Pilots = np.array([1, 1, 1, -1])
	# P_i = np.array([1,1,1,1, -1,-1,-1,1, -1,-1,-1,-1, 1,1,-1,1, -1,-1,1,1, -1,1,1,-1, 1,1,1,1,1,1,-1,1, 1,1,-1,1, 1,-1,-1,1, 1,1,-1,1, -1,-1,-1,1, -1,1,-1,-1, 1,-1,-1,1, 1,1,1,1,-1,-1,1,1,-1,-1,1,-1, 1,-1,1,1, -1,-1,-1,1, 1,-1,-1,-1, -1,1,-1,-1, 1,-1,1,1, 1,1,-1,1, -1,1,-1,1, -1,-1,-1,-1, -1,1,-1,1, 1,-1,1,-1, 1,1,1,-1, -1,1,-1,-1, -1,1,1,1, -1,-1,-1,-1, -1,-1,-1])

	# ind_all = np.arange(-32, 32) + 32
	# ind_guard = np.concatenate((np.arange(-32, -26), np.arange(27, 32))) + 32
	# ind_null = np.array([0]) + 32
	# ind_pilots = np.array([-21, -7, 7, 21]) + 32
	# mask_data = np.ones(64)
	# mask_data_pilots = np.ones(64)
	# mask_data[list(np.concatenate((ind_guard, ind_null, ind_pilots)))] = 0
	# mask_data_pilots[list(np.concatenate((ind_guard, ind_null)))] = 0
	# ind_data = ind_all[mask_data==1]
	# ind_data_pilots = ind_all[mask_data_pilots==1]

	# Signal = fftshift(fft(frame[n_short + n_long + 16 : n_short + n_long + 80]))
	# n_symbols_est = np.int(np.floor((frame_length-n_short- n_long)/80)) - 1
	# Data = np.zeros((n_symbols_est,64)) + 0j
	# for i in range(n_symbols_est):
	# 	Data[i] = fftshift(fft(frame[n_short + n_long + 16 + 80*(i+1):n_short + n_long + 80*(i+2)]))
	# 	Data[i][ind_pilots] = Data[i][ind_pilots]*P_i[i+1]	# Remove randomness in Pilots

	# # Sampling freq offset correction
	# Signal[ind_data_pilots] *= np.exp(1j*2*np.pi*(80/64) * epsilon_0 * (ind_data_pilots-32))

	# # Fine estimation of sampling frequency offset
	# W = Signal[ind_pilots].dot(Pilots.T)
	# epsilon = np.zeros(n_symbols_est+2)
	# epsilon[0] = epsilon_0
	# dr = 2*np.pi*4*Fc*1e-6
	# epsilon[1] = epsilon[0] + np.angle(W)/(4*dr)

	# for i in range(n_symbols_est):
	# 	# Sampling freq offset correction
	# 	Data[i][ind_data_pilots] *= np.exp(1j*2*np.pi*(80/64) * epsilon[i+1] * (ind_data_pilots-32))

	# 	# Fine estimation of sampling frequency offset
	# 	W += Data[i][ind_pilots].dot(Pilots.T)
	# 	epsilon[i+2] = epsilon[i+1] + np.angle(W)/(4*(i+2)*dr)

	# 	# Restore pilot multiplier
	# 	Data[i][ind_pilots] = Data[i][ind_pilots]*P_i[i+1]

	# # Go back to time domain
	# frame[n_short + n_long + 16 : n_short + n_long + 80] = ifft(ifftshift(Signal))
	# for i in range(n_symbols_est):
	# 	frame[n_short + n_long + 16 + 80*(i+1):n_short + n_long + 80*(i+2)] = ifft(ifftshift(Data[i]))

	# Dictionary containing frequency offsets
	# freq_offset = {}
	# freq_offset['carrier_coarse'] = df1
	# freq_offset['carrier_fine'] = df2
	# freq_offset['sampling'] = epsilon
	# freq_offset = np.array([df1, df2, epsilon])
	# freq_offset = np.array([df1, df2])
	freq_offset = np.array([df1, df2])

	return frame, freq_offset

def estimate_freq_offset_(frame, Fs = 20e6, verbose=False):
	"""
	Function to read WiFi data from file and do the following pre-processing operations
	 	1. Filter out out
	 	-of-band noise
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

	if Fs == 20e6:

		n_short = 160
		n_long = 160

		# ----------------------------------------------------
		# Coarse frequency offset correction (using STF)
		# ----------------------------------------------------

		sig3 = frame[n_short//2 : n_short-16].conj().copy()
		sig4 = frame[n_short//2 + 16 : n_short].copy()
		df1 = 1/16.0 * np.angle(sig3.dot(sig4.T))
		frame *= np.exp(-1j*np.arange(0, frame.size) *df1).flatten()

		# if verbose==True:
		# 	print('Coarse freq offset = {:.2f} KHz'.format(df1* 2e8 / (2*np.pi*1e3)))

		# ----------------------------------------------------
		# Fine frame alignment (using LTF cross-correlation)
		# ----------------------------------------------------

		Ltf = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
		ltf = ifft(ifftshift(Ltf))
		var_ltf = np.sum(np.abs(ltf)**2)

		guard_band = 0

		search_length_ltf = frame.size - 128
		corr_ltf = np.zeros(search_length_ltf)
		for n in range(search_length_ltf):	
			corr_ltf[n] = np.abs(frame[n:n+64].dot(ltf.conj().T))
			corr_ltf[n] /= np.sqrt(np.sum(np.abs(frame[n:n+64])**2)*var_ltf)
		ind_ltf_1 = corr_ltf.argmax()
		ind_ltf_2 = np.concatenate((corr_ltf[:ind_ltf_1-60], np.zeros(120), corr_ltf[ind_ltf_1+60:])).argmax()
		if (np.abs(ind_ltf_1 - ind_ltf_2) > 60) and (np.abs(ind_ltf_1 - ind_ltf_2) < 70):
			ind_ltf = max(ind_ltf_1, ind_ltf_2)
			frame_start = ind_ltf.max() + 64 - n_long - n_short
			if frame_start <= 0:
				frame_start = guard_band # Use guess from guard band size
		else:
			frame_start = guard_band # Use guess from guard band size

		frame = frame[frame_start:]
		frame_length = frame.shape[0]

		# ----------------------------------------------------
		# Fine frequency offset correction (using LTF)
		# ----------------------------------------------------

		sig5 = frame[n_short+32:n_short+32+64].conj().copy()
		sig6 = frame[n_short+64+32:n_short+n_long].copy()
		df2 = 1/64.0 * np.angle(sig5.dot(sig6.T))
		frame *= np.exp(-1j*np.arange(0,frame_length)*df2).flatten()

		# if df1<0:
		# 	df1 += 2*np.pi/16.0
		# if df2<0:
		# 	df2 += 2*np.pi/64.0

		df1 /= (2*np.pi)
		df2 /= (2*np.pi)

		df1 *= 100
		df2 *= 100

		if verbose==True:
			# print('Coarse freq offset = {:.2f} KHz'.format(df1 * Fs / (2*np.pi*1e3)))
			# print('Fine freq offset = {:.2f} KHz'.format(df2 * Fs / (2*np.pi*1e3)))
			print('Coarse freq offset = {:.3f} %'.format(df1))
			print('Fine freq offset = {:.3f} %'.format(df2))

		# # -------------------------------------------------------------
		# # Sampling frequency offset correction
		# # This is done in frequency domain, using pilot subcarriers
		# # -------------------------------------------------------------

		# epsilon_0 = (df1 + df2) / (2*np.pi*Fc)
		# Pilots = np.array([1, 1, 1, -1])
		# P_i = np.array([1,1,1,1, -1,-1,-1,1, -1,-1,-1,-1, 1,1,-1,1, -1,-1,1,1, -1,1,1,-1, 1,1,1,1,1,1,-1,1, 1,1,-1,1, 1,-1,-1,1, 1,1,-1,1, -1,-1,-1,1, -1,1,-1,-1, 1,-1,-1,1, 1,1,1,1,-1,-1,1,1,-1,-1,1,-1, 1,-1,1,1, -1,-1,-1,1, 1,-1,-1,-1, -1,1,-1,-1, 1,-1,1,1, 1,1,-1,1, -1,1,-1,1, -1,-1,-1,-1, -1,1,-1,1, 1,-1,1,-1, 1,1,1,-1, -1,1,-1,-1, -1,1,1,1, -1,-1,-1,-1, -1,-1,-1])

		# ind_all = np.arange(-32, 32) + 32
		# ind_guard = np.concatenate((np.arange(-32, -26), np.arange(27, 32))) + 32
		# ind_null = np.array([0]) + 32
		# ind_pilots = np.array([-21, -7, 7, 21]) + 32
		# mask_data = np.ones(64)
		# mask_data_pilots = np.ones(64)
		# mask_data[list(np.concatenate((ind_guard, ind_null, ind_pilots)))] = 0
		# mask_data_pilots[list(np.concatenate((ind_guard, ind_null)))] = 0
		# ind_data = ind_all[mask_data==1]
		# ind_data_pilots = ind_all[mask_data_pilots==1]

		# Signal = fftshift(fft(frame[n_short + n_long + 16 : n_short + n_long + 80]))
		# n_symbols_est = np.int(np.floor((frame_length-n_short- n_long)/80)) - 1
		# Data = np.zeros((n_symbols_est,64)) + 0j
		# for i in range(n_symbols_est):
		# 	Data[i] = fftshift(fft(frame[n_short + n_long + 16 + 80*(i+1):n_short + n_long + 80*(i+2)]))
		# 	Data[i][ind_pilots] = Data[i][ind_pilots]*P_i[i+1]	# Remove randomness in Pilots

		# # Sampling freq offset correction
		# Signal[ind_data_pilots] *= np.exp(1j*2*np.pi*(80/64) * epsilon_0 * (ind_data_pilots-32))

		# # Fine estimation of sampling frequency offset
		# W = Signal[ind_pilots].dot(Pilots.T)
		# epsilon = np.zeros(n_symbols_est+2)
		# epsilon[0] = epsilon_0
		# dr = 2*np.pi*4*Fc*1e-6
		# epsilon[1] = epsilon[0] + np.angle(W)/(4*dr)

		# for i in range(n_symbols_est):
		# 	# Sampling freq offset correction
		# 	Data[i][ind_data_pilots] *= np.exp(1j*2*np.pi*(80/64) * epsilon[i+1] * (ind_data_pilots-32))

		# 	# Fine estimation of sampling frequency offset
		# 	W += Data[i][ind_pilots].dot(Pilots.T)
		# 	epsilon[i+2] = epsilon[i+1] + np.angle(W)/(4*(i+2)*dr)

		# 	# Restore pilot multiplier
		# 	Data[i][ind_pilots] = Data[i][ind_pilots]*P_i[i+1]

		# # Go back to time domain
		# frame[n_short + n_long + 16 : n_short + n_long + 80] = ifft(ifftshift(Signal))
		# for i in range(n_symbols_est):
		# 	frame[n_short + n_long + 16 + 80*(i+1):n_short + n_long + 80*(i+2)] = ifft(ifftshift(Data[i]))

		# Dictionary containing frequency offsets
		# freq_offset = {}
		# freq_offset['carrier_coarse'] = df1
		# freq_offset['carrier_fine'] = df2
		# freq_offset['sampling'] = epsilon
		# freq_offset = np.array([df1, df2, epsilon])
		# freq_offset = np.array([df1, df2])
		freq_offset = np.array([df1, df2])

	elif Fs == 200e6:
		n_short = 1600
		n_long = 1600
		L = 160
		N = 640

		# ----------------------------------------------------
		# Coarse frequency offset correction (using STF)
		# ----------------------------------------------------

		sig3 = frame[n_short//2 : n_short-L].conj().copy()
		sig4 = frame[n_short//2 + L : n_short].copy()
		df1 = 1.0/L * np.angle(sig3.dot(sig4.T))
		frame *= np.exp(-1j*np.arange(0, frame.size) *df1).flatten()

		# if verbose==True:
		# 	print('Coarse freq offset = {:.2f} KHz'.format(df1* 2e8 / (2*np.pi*1e3)))

		# ----------------------------------------------------
		# Fine frame alignment (using LTF cross-correlation)
		# ----------------------------------------------------

		# Ltf = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
		# ltf = ifft(ifftshift(Ltf))
		# var_ltf = np.sum(np.abs(ltf)**2)

		# guard_band = 0

		# search_length_ltf = frame.size - 128
		# corr_ltf = np.zeros(search_length_ltf)
		# for n in range(search_length_ltf):	
		# 	corr_ltf[n] = np.abs(frame[n:n+64].dot(ltf.conj().T))
		# 	corr_ltf[n] /= np.sqrt(np.sum(np.abs(frame[n:n+64])**2)*var_ltf)
		# ind_ltf_1 = corr_ltf.argmax()
		# ind_ltf_2 = np.concatenate((corr_ltf[:ind_ltf_1-60], np.zeros(120), corr_ltf[ind_ltf_1+60:])).argmax()
		# if (np.abs(ind_ltf_1 - ind_ltf_2) > 60) and (np.abs(ind_ltf_1 - ind_ltf_2) < 70):
		# 	ind_ltf = max(ind_ltf_1, ind_ltf_2)
		# 	frame_start = ind_ltf.max() + 64 - n_long - n_short
		# 	if frame_start <= 0:
		# 		frame_start = guard_band # Use guess from guard band size
		# else:
		# 	frame_start = guard_band # Use guess from guard band size

		# frame = frame[frame_start:]
		# frame_length = frame.shape[0]

		# ----------------------------------------------------
		# Fine frequency offset correction (using LTF)
		# ----------------------------------------------------

		sig5 = frame[n_short+2*L:n_short+2*L+N].conj().copy()
		sig6 = frame[n_short+N+2*L:n_short+n_long].copy()
		df2 = 1.0/N * np.angle(sig5.dot(sig6.T))
		frame *= np.exp(-1j*np.arange(0,frame.shape[0])*df2).flatten()

		# if df1<0:
		# 	df1 += 2*np.pi/16.0
		# if df2<0:
		# 	df2 += 2*np.pi/64.0

		df1 /= (2*np.pi)
		df2 /= (2*np.pi)

		df1 *= 100
		df2 *= 100

		if verbose==True:
			# print('Coarse freq offset = {:.2f} KHz'.format(df1 * Fs / (2*np.pi*1e3)))
			# print('Fine freq offset = {:.2f} KHz'.format(df2 * Fs / (2*np.pi*1e3)))
			print('Coarse freq offset = {:.3f} %'.format(df1))
			print('Fine freq offset = {:.3f} %'.format(df2))

		freq_offset = np.array([df1, df2])

	else:
		raise NotImplementedError()


	return frame, freq_offset

def main():

	exp_dir = '/home/rfml/wifi/experiments/exp19'


	sample_duration = 16
	preprocess_type = 1
	sample_rate = 20


	#-------------------------------------------------
	# Physical offset params
	#-------------------------------------------------
	df_phy_train = 40e-6
	df_phy_test = 40e-6

	seed_phy_train = 40
	seed_phy_test = 60

	#-------------------------------------------------
	# Physical channel params
	#-------------------------------------------------
	'''
	phy_method:
		0 - same channel for all packets
		1 - different channel for each class, same channel for all packets in a class
	'''
	# phy_method = 0
	phy_method = 1

	'''
	channel type:
		1 - Extended Pedestrian A
		2 - Extended Vehicular A
		3 - Extended Typical Urban
	'''
	channel_type_phy_train = 1
	channel_type_phy_test = 1

	data_format_plain = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

	npz_filename = exp_dir + '/sym-' + data_format_plain + '.npz'

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

	x_train_orig = dict_wifi['x_train'].copy()
	y_train_orig = dict_wifi['y_train'].copy()
	num_classes = y_train_orig.shape[1]

	x_test_orig = dict_wifi['x_test'].copy()
	y_test_orig = dict_wifi['y_test'].copy()

	fc_train_orig = dict_wifi['fc_train']
	fc_test_orig = dict_wifi['fc_test']

	fs = sample_rate * 1e+6
	sampling_rate = sample_rate * 1e+6

	#--------------------------------------------------------------------------------------------
	# Physical offset simulation (different days)
	#--------------------------------------------------------------------------------------------

	print('\n---------------------------------------------')
	print('Physical offset simulation (different days)')
	print('---------------------------------------------')
	print('Physical offsets: Train: {}, Test: {} ppm'.format(df_phy_train*1e6, df_phy_test*1e6))
	print('Physical seeds: Train: {}, Test: {}\n'.format(seed_phy_train, seed_phy_test))

	signal_ch = x_train_orig.copy()
	for n in trange(num_classes):
		ind_n = np.where(y_train_orig.argmax(axis=1)==n)[0]
		seed_phy_train_n = seed_phy_train + n
		for i in ind_n:
			rv_n = np.random.RandomState(seed=seed_phy_train_n)
			signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
																 df = rv_n.uniform(low=-df_phy_train, high=df_phy_train),
																 fc = fc_train_orig[i:i+1], 
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
																 fc = fc_test_orig[i:i+1], 
																 fs = fs)
	dict_wifi['x_test'] = signal_ch.copy()


#--------------------------------------------------------------------------------------------
# Physical channel simulation (different days)
#--------------------------------------------------------------------------------------------

# print('\n---------------------------------------------')
# print('Physical channel simulation (different days)')
# print('---------------------------------------------')
# print('Physical channel types: Train: {}, Test: {}'.format(channel_type_phy_train, channel_type_phy_test))
# print('Physical seeds: Train: {}, Test: {}\n'.format(seed_phy_train, seed_phy_test))


# if phy_method == 0: # Same channel for all packets
# 	signal_ch = dict_wifi['x_train'].copy()
# 	for i in tqdm(range(num_train)):
# 		signal = dict_wifi['x_train'][i][:,0] + 1j*dict_wifi['x_train'][i][:,1]
# 		signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
# 															seed=seed_phy_train, 
# 															beta=0, 
# 															delay_seed=False, 
# 															channel_type=channel_type_phy_train,
# 															channel_method=channel_method,
# 															noise_method=noise_method)
# 		signal_faded = normalize(signal_faded)
# 		signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
# 	dict_wifi['x_train'] = signal_ch.copy()

# 	signal_ch = dict_wifi['x_test'].copy()
# 	for i in tqdm(range(num_test)):
# 		signal = dict_wifi['x_test'][i][:,0] + 1j*dict_wifi['x_test'][i][:,1]
# 		signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
# 															seed=seed_phy_test, 
# 															beta=0, 
# 															delay_seed=False,
# 															channel_type=channel_type_phy_test,
# 															channel_method=channel_method,
# 															noise_method=noise_method)
# 		signal_faded = normalize(signal_faded)
# 		signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
# 	dict_wifi['x_test'] = signal_ch.copy()

# else: # Different channel for each class, same channel for all packets in a class
# 	signal_ch = dict_wifi['x_train'].copy()
# 	for n in trange(num_classes):
# 		ind_n = np.where(y_train_orig.argmax(axis=1)==n)[0]
# 		seed_phy_train_n = seed_phy_train + n
# 		# print('{}: {}'.format(n, ind_n))
# 		for i in ind_n:
# 			signal = dict_wifi['x_train'][i][:,0] + 1j*dict_wifi['x_train'][i][:,1]
# 			signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
# 																seed=seed_phy_train_n, 
# 																beta=0, 
# 																delay_seed=False, 
# 																channel_type=channel_type_phy_train,
# 																channel_method=channel_method,
# 																noise_method=noise_method)
# 			signal_faded = normalize(signal_faded)
# 			signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
# 	dict_wifi['x_train'] = signal_ch.copy()

# 	signal_ch = dict_wifi['x_test'].copy()
# 	for n in trange(num_classes):
# 		ind_n = np.where(y_test_orig.argmax(axis=1)==n)[0]
# 		seed_phy_test_n = seed_phy_test + n
# 		for i in ind_n:
# 			signal = dict_wifi['x_test'][i][:,0] + 1j*dict_wifi['x_test'][i][:,1]
# 			signal_faded = add_custom_fading_channel(signal, 500, sampling_rate, 
# 																seed=seed_phy_test_n, 
# 																beta=0, 
# 																delay_seed=False,
# 																channel_type=channel_type_phy_test,
# 																channel_method=channel_method,
# 																noise_method=noise_method)
# 			signal_faded = normalize(signal_faded)
# 			signal_ch[i] = np.concatenate((signal_faded.real.reshape([-1,1]),signal_faded.imag.reshape([-1,1])),axis=1).reshape((1,-1,2))
# 	dict_wifi['x_test'] = signal_ch.copy()


	x_train = dict_wifi['x_train'].copy()
	y_train = dict_wifi['y_train'].copy()
	x_test = dict_wifi['x_test'].copy()
	y_test = dict_wifi['y_test'].copy()
	
	num_train = x_train.shape[0]
	num_test = x_test.shape[0]

	complex_train = x_train[..., 0] + 1j* x_train[..., 1]
	complex_test = x_test[..., 0] + 1j* x_test[..., 1]

	complex_train_removed_cfo = complex_train.copy()
	complex_test_removed_cfo = complex_test.copy()

	freq_train = np.zeros([num_train, 2])
	for i in trange(num_train):
	# for i in range(2):
		complex_train_removed_cfo[i], freq_train[i] = estimate_freq_offset(complex_train[i], verbose=False)

	freq_train_removed_cfo = np.zeros([num_train, 2])
	for i in range(num_train):
	# for i in range(2):
		_, freq_train_removed_cfo[i] = estimate_freq_offset(complex_train_removed_cfo[i], verbose=False)


	num_classes_plot = 20
	histogram_directory = "cfo_images/"
	if not os.path.exists(histogram_directory):
		os.makedirs(histogram_directory)

	plt.figure(figsize=[15, 6])
	num_rows = 4
	num_cols = 5
	for i in range(num_classes_plot):
		plt.subplot(num_rows, num_cols, i+1)
		ind_n = np.where(y_train.argmax(axis=1)==i)[0]
		plt.hist(freq_train[ind_n, 0], density=True, bins=25)   #,range = (-6,6)
		# plt.axvline(thresh[i], color='r', linestyle='dashed', linewidth=1)
		plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
		plt.title('Class {}'.format(i+1), fontsize=12)
	plt.suptitle('Histogram of coarse frequency offset')
	fig_name = os.path.join(histogram_directory,'coarse_freq_offset.pdf') 
	plt.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')	

	plt.figure(figsize=[15, 6])
	num_rows = 4
	num_cols = 5
	for i in range(num_classes_plot):
		plt.subplot(num_rows, num_cols, i+1)
		ind_n = np.where(y_train.argmax(axis=1)==i)[0]
		plt.hist(freq_train[ind_n, 1], density=True, bins=25)
		# plt.axvline(thresh[i], color='r', linestyle='dashed', linewidth=1)
		plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
		plt.title('Class {}'.format(i+1), fontsize=12)
	plt.suptitle('Histogram of fine frequency offset')	
	fig_name = os.path.join(histogram_directory,'fine_freq_offset.pdf') 
	plt.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')	

	plt.figure(figsize=[15, 6])
	num_rows = 4
	num_cols = 5
	for i in range(num_classes_plot):
		plt.subplot(num_rows, num_cols, i+1)
		ind_n = np.where(y_train.argmax(axis=1)==i)[0]
		plt.hist(freq_train[ind_n, 0] + freq_train[ind_n, 1], density=True, bins=25)
		# plt.axvline(thresh[i], color='r', linestyle='dashed', linewidth=1)
		plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
		plt.title('Class {}'.format(i+1), fontsize=12)
	plt.suptitle('Histogram of total frequency offset')	
	fig_name = os.path.join(histogram_directory,'total_freq_offset.pdf') 
	plt.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')	




	print('Bitti')



	plt.figure(figsize=[15, 6])
	num_rows = 4
	num_cols = 5
	for i in range(num_classes_plot):
		plt.subplot(num_rows, num_cols, i+1)
		ind_n = np.where(y_train.argmax(axis=1)==i)[0]
		plt.hist(freq_train_removed_cfo[ind_n, 0], density=True, bins=25)
		# plt.axvline(thresh[i], color='r', linestyle='dashed', linewidth=1)
		plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
		plt.title('Class {}'.format(i+1), fontsize=12)
	plt.suptitle('Histogram of coarse frequency offset after cfo correction')
	fig_name = os.path.join(histogram_directory,'coarse_freq_offset_corrected.pdf') 
	plt.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')	

	plt.figure(figsize=[15, 6])
	num_rows = 4
	num_cols = 5
	for i in range(num_classes_plot):
		plt.subplot(num_rows, num_cols, i+1)
		ind_n = np.where(y_train.argmax(axis=1)==i)[0]
		plt.hist(freq_train_removed_cfo[ind_n, 1], density=True, bins=25)
		# plt.axvline(thresh[i], color='r', linestyle='dashed', linewidth=1)
		plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
		plt.title('Class {}'.format(i+1), fontsize=12)
	plt.suptitle('Histogram of fine frequency offset after cfo correction')	
	fig_name = os.path.join(histogram_directory,'fine_freq_offset_corrected.pdf') 
	plt.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')	

	plt.figure(figsize=[15, 6])
	num_rows = 4
	num_cols = 5
	for i in range(num_classes_plot):
		plt.subplot(num_rows, num_cols, i+1)
		ind_n = np.where(y_train.argmax(axis=1)==i)[0]
		plt.hist(freq_train_removed_cfo[ind_n, 0] + freq_train_removed_cfo[ind_n, 1], density=True, bins=25)
		# plt.axvline(thresh[i], color='r', linestyle='dashed', linewidth=1)
		plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
		plt.title('Class {}'.format(i+1), fontsize=12)
	plt.suptitle('Histogram of total frequency offset after cfo correction')	
	fig_name = os.path.join(histogram_directory,'total_freq_offset_corrected.pdf') 
	plt.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')	


if __name__ == '__main__':
	main()

