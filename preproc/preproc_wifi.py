'''
Contains code for fractionally spaced equalization, preamble detection
Also includes a modified version of Teledyne's data read and preprocessing code
'''

import numpy as np
import os
import json
import csv
import math
import fractions
import resampy
from tqdm import tqdm, trange
import matplotlib
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, ifftshift
import ipdb
from sklearn.preprocessing import normalize


def preprocess_wifi(data_dict, sample_duration, sample_rate, preprocess_type=1, progress=True):
    '''
    Detects preamble and extract its
    '''

    signal_indices = range(len(data_dict['data_file']))
    if progress is True:
        signal_indices = tqdm(signal_indices)

    flag = 0

    for i in signal_indices:
        signal = data_dict['signal'][i]
        orig_sample_rate = data_dict['capture_sample_rate'][i]
        start_index = 0
        end_index = math.ceil(sample_duration * orig_sample_rate)

        if orig_sample_rate == np.int(200e6):
            if (preprocess_type == 2) or (preprocess_type == 3):
                lowFreq = data_dict['freq_lower_edge'][i]
                upFreq = data_dict['freq_upper_edge'][i]
                Fc = data_dict['capture_frequency'][i]
                signal, flag_i = detect_frame(signal, lowFreq, upFreq, Fc, verbose=False)
                flag = flag + flag_i
            if preprocess_type == 3:
                signal = frac_eq_preamble(signal)

        start_index = np.int(start_index)
        end_index = np.int(end_index)

        if (preprocess_type == 1) or (preprocess_type == 2) or (orig_sample_rate != np.int(200e6)):
            signal = signal[start_index:end_index]     # extract needed section of signal

        with np.errstate(all='raise'):
            try:
                signal = signal / rms(signal)  # normalize signal
            except FloatingPointError:
                # print('data_file = '+str(data_dict['data_file'][i]) + ',\t reference_number = '+str(data_dict['reference_number'][i]))
                try:
                    # print('Normalization error. RMS = {}, Max = {}, Min = {}, Data size = {}'.format(rms(signal), np.abs(signal).min(), np.abs(signal).max(), signal.shape))
                    signal += 1.0/np.sqrt(2*signal.size) + 1.0/np.sqrt(2*signal.size)*1j
                except FloatingPointError:
                    # print('i = {}, signal.shape = {}'.format(i, signal.shape))
                    # print('start_index = {}, end_index = {}'.format(start_index, end_index))
                    signal_size = end_index - start_index
                    signal = np.ones([signal_size]) * (1.0 + 1.0*1j)/np.sqrt(2*signal_size)

        if (preprocess_type == 1) or (orig_sample_rate != np.int(200e6)):
            freq_shift = (data_dict['freq_upper_edge'][i] +
                          data_dict['freq_lower_edge'][i])/2 - data_dict['capture_frequency'][i]
            # baseband signal w.r.t. center frequency
            signal = shift_frequency(signal, freq_shift, orig_sample_rate)
            # filter and downsample signal
            signal = resample(signal, orig_sample_rate, sample_rate)

        if (preprocess_type == 2):
            signal = resample(signal, orig_sample_rate, sample_rate)

        data_dict['signal'][i] = signal
        # data_dict['freq_lower_edge'][i] = -sample_rate/2.
        # data_dict['freq_upper_edge'][i] = sample_rate/2.
        # data_dict['sample_start'][i] = 0
        # data_dict['sample_count'][i] = len(signal)
        data_dict['center_frequency'][i] = (
            data_dict['freq_upper_edge'][i] + data_dict['freq_lower_edge'][i])/2.
        data_dict['sample_rate'][i] = sample_rate

    if (preprocess_type == 2) or (preprocess_type == 3):
        print('Successful frame detection on {:.2f}% of signals'.format(
            100.0-flag*100.0/len(data_dict['data_file'])))

    return data_dict


def frac_eq_preamble(rx, verbose=False):
    '''
    Fractionally equalize preamble
    https://ieeexplore.ieee.org/document/489269
    '''

    # print('Hello!')

    Stf_64 = np.sqrt(13/6)*np.array([0, 0, 0, 0, 0, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0,
                                     1+1j, 0, 0, 0, 0, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 0, 0, 0, 0])
    stf_64 = ifft(ifftshift(Stf_64))
    # stf = stf_64[:16]

    Ltf = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1,
                    1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    ltf = ifft(ifftshift(Ltf))

    tx = np.concatenate((stf_64[:-32], stf_64, stf_64, ltf[-32:], ltf, ltf))
    L = 160
    N = 320

    rx = rx.reshape([-1, 1])

    R = np.zeros([L, L]) + 0j
    p = np.zeros([L, 1]) + 0j
    for i in range(N):
        j = 10*i
        R += rx[j:j+L].dot(rx[j:j+L].conj().T)
        p += rx[j:j+L] * tx[i].conj()

    c, residuals, rank, sing = np.linalg.lstsq(R, p)
    # h = c[::-1].conj()
    # rx_eq = np.convolve(h, rx, mode='full')[np.int(L/2):-np.int(L/2)]
    # signal_eq = rx_eq[::10][:1600]

    signal_eq = np.zeros([N, 1]) + 0j
    for i in range(N):
        j = 10*i
        signal_eq[i] = rx[j:j+L].T.dot(c.conj())

    return signal_eq.flatten()


def detect_frame(complex_signal, lowFreq, upFreq, Fc, verbose=False):
    '''
    Detects preamble and extract its
    '''

    Fs = 200e6
    flag = 0

    # ----------------------------------------------------
    # Filter out-of-band noise
    # ----------------------------------------------------

    N = complex_signal.shape[0]
    if N % 2 != 0:
        complex_signal = complex_signal[:-1]
        N -= 1
    low_ind = np.int((lowFreq-Fc)*(N/Fs) + N/2)
    up_ind = np.int((upFreq-Fc)*(N/Fs) + N/2)
    lag = np.int((-Fc + (lowFreq+upFreq)/2)*(N/Fs) + N/2) - np.int(N/2)
    X = fftshift(fft(complex_signal))
    X[:low_ind] = 0 + 0j
    X[up_ind:] = 0 + 0j
    X = np.roll(X, -lag)
    complex_signal = ifft(ifftshift(X))

    # ----------------------------------------------------
    # Coarse frame detection (using STF)
    # ----------------------------------------------------

    guard_band_upsamp = np.int(2e-6*Fs)  # 2 usec
    n_win = 1600-160                        # ?
    lag = 160
    search_length_stf_upsamp = min(2*guard_band_upsamp+1, np.int(complex_signal.size))
    autocorr_stf_upsamp = np.zeros(search_length_stf_upsamp)
    a = np.zeros(search_length_stf_upsamp)+0j
    p = np.zeros(search_length_stf_upsamp)
    for n in range(search_length_stf_upsamp):
        sig1 = complex_signal[n:n+n_win].reshape(1, -1)
        sig2 = complex_signal[n+lag:n+n_win+lag].conj().reshape(1, -1)
        a[n] = sig1.dot(sig2.T)
        # p[n] = np.sum(np.abs(sig1)**2)
        p[n] = np.sqrt(np.sum(np.abs(sig1)**2)*np.sum(np.abs(sig2)**2))
    autocorr_stf_upsamp = np.abs(a)/p
    frame_start_autocorr_upsamp = np.argmax(autocorr_stf_upsamp)

    # ----------------------------------------------------
    # Guard band sanity check
    # ----------------------------------------------------

    n_short_upsamp = 1600

    if frame_start_autocorr_upsamp <= 2*guard_band_upsamp:
        # sig3 = complex_signal[frame_start_autocorr_upsamp+np.int(n_short_upsamp/2):frame_start_autocorr_upsamp+n_short_upsamp-160].conj().copy()
        # sig4 = complex_signal[frame_start_autocorr_upsamp+np.int(n_short_upsamp/2)+160:frame_start_autocorr_upsamp+n_short_upsamp].copy()
        # df1_upsamp = 1/160 * np.angle(sig3.dot(sig4.T))
        # complex_signal[frame_start_autocorr_upsamp:] *= np.exp(-1j*np.arange(0,complex_signal.size - frame_start_autocorr_upsamp)*df1_upsamp).flatten()
        if verbose == True:
            print('Autocorr prediction = {}'.format(frame_start_autocorr_upsamp))
            # print('Freq offset_upsamp = {:.2f} KHz'.format(df1_upsamp* 2e8 / (2*np.pi*1e3)))
    else:
        if verbose == True:
            print('Autocorr detection failed\n Prediction = {}'.format(frame_start_autocorr_upsamp))
        frame_start_autocorr_upsamp = guard_band_upsamp
        # df1_upsamp = 0
        flag = 1

    return complex_signal[frame_start_autocorr_upsamp:], flag


def offset_compensate_preamble(preamble_in, fs=200e6, verbose=False, option=1):
    """
    Function that strips out the effect of the offset from the preamble. 

    df = 1/16 arg(sum_{n=0}^{N_short - 1 - 16} s[n]* s'[n+16] )
    s[n] <---- s[n]* e^(j.n.df)

    Inputs:
        preamble  - Preamble containing effects of the channel and Tx nonlinearities
                    (320 samples)
        fs        - Sampling frequency
        [Verbose] - Verbose
        ### NotImplemented: freq_offset - Dict containing freq offset

    Output: 
        preamble_eq - Preamble with the channel stripped out (320 samples)
        ### NotImplemented: preamble_eq_offset - Equalized preamble with frequency offset

    """

    # if fs!=20e6:
    #   raise NotImplementedError

    preamble = preamble_in.copy()

    if fs == 200e6:
        if preamble.size != 3200:
            raise Exception('Size of preamble is {}, but it should be 3200.'.format(preamble.size))

        n_short = 1600  # Length of short preamble
        n_long = 1600   # Length of long preamble

        L = 160  # length of single short sequence
        N = 640  # length of single long sequnce

        # ----------------------------------------------------
        # Frequency offset correction
        # ----------------------------------------------------
        # Coarse estimation
        # sig3 = preamble[n_short//2: n_short-L].conj().copy()
        # sig4 = preamble[n_short//2 + L: n_short].copy()
        sig3 = preamble[: n_short-L].conj().copy()
        sig4 = preamble[L: n_short].copy()
        df1 = 1./L * np.angle(sig3.dot(sig4.T))
        preamble *= np.exp(-1j*np.arange(0, preamble.size)*df1).flatten()

        # Fine estimation
        sig5 = preamble[n_short + 2*L: n_short + 2*L + N].conj().copy()
        sig6 = preamble[n_short + N+2*L: n_short + n_long].reshape(1, -1).copy()
        df2 = 1./N * np.angle(sig5.dot(sig6.T))
        preamble *= np.exp(-1j*np.arange(0, preamble.size)*df2).flatten()
        freq_offset = np.array([df1, df2])

    elif fs == 20e6:

        if preamble.size != 320:
            raise Exception('Size of preamble is {}, but it should be 320.'.format(preamble.size))

        n_short = 160  # Length of short preamble
        n_long = 160   # Length of long preamble

        L = 16  # length of single short sequence
        N = 64  # length of single long sequence

        # ----------------------------------------------------
        # Frequency offset correction
        # ----------------------------------------------------
        # Coarse estimation
        sig3 = preamble[np.int(n_short/2):n_short-L].conj().copy()
        sig4 = preamble[np.int(n_short/2)+L:n_short].copy()
        df1 = 1./L * np.angle(sig3.dot(sig4.T))
        preamble *= np.exp(-1j*np.arange(0, preamble.size)*df1).flatten()

        # Fine estimation
        sig5 = preamble[n_short+32:n_short+32+N].conj().copy()
        sig6 = preamble[n_short+N+32:n_short+n_long].reshape(1, -1).copy()
        df2 = 1./N * np.angle(sig5.dot(sig6.T))
        preamble *= np.exp(-1j*np.arange(0, preamble.size)*df2).flatten()
        freq_offset = np.array([df1, df2])

    if option == 1:
        return preamble
    elif option == 2:
        return preamble, freq_offset
    else:
        raise NotImplementedError


def get_residuals_preamble(preamble_in, fs, method='subtraction', channel_method='frequency', verbose=False, label=''):
    """
    Function that reconstructs the preamble fed into this function with the channel and CFO effects
    and returns the difference between original preamble and reconstructed one (residuals):
    Inputs:
        preamble  - Preamble containing effects of the channel and Tx nonlinearities
                    (3200 samples)
        ### NotImplemented: freq_offset - Dict containing freq offset

    Output: 
        preamble_eq - Preamble with the channel stripped out (320 samples)
        ### NotImplemented: preamble_eq_offset - Equalized preamble with frequency offset

    """
    # if fs!=20e6:
    #   raise NotImplementedError

    preamble = preamble_in.copy()
    preamble_orig = preamble_in.copy()

    if fs == 200e6:
        if preamble.size != 3200:
            raise Exception('Size of preamble is {}, but it should be 3200.'.format(preamble.size))

        n_short = 1600
        n_long = 1600

        L = 160
        N = 640

        # ----------------------------------------------------
        # Frequency offset correction
        # ----------------------------------------------------
        sig3 = preamble[: n_short-L].conj().copy()
        sig4 = preamble[L: n_short].copy()
        df1 = 1./L * np.angle(sig3.dot(sig4.T))
        preamble *= np.exp(-1j*np.arange(0, preamble.size)*df1).flatten()

        # Fine estimation
        sig5 = preamble[n_short + 2*L: n_short + 2*L + N].conj().copy()
        sig6 = preamble[n_short + N+2*L: n_short + n_long].reshape(1, -1).copy()
        df2 = 1./N * np.angle(sig5.dot(sig6.T))
        preamble *= np.exp(-1j*np.arange(0, preamble.size)*df2).flatten()
        freq_offset = np.array([df1, df2])

        cfo_total = np.multiply(np.exp(1j*np.arange(0, preamble.size)*df1).flatten(),
                                np.exp(1j*np.arange(0, preamble.size)*df2).flatten())

        # ------------------------------------------------------------------------
        # LTI channel estimation (with delay spread <= length of cyclic prefix)
        # ------------------------------------------------------------------------

        Stf_64 = np.sqrt(13/6)*np.array([0, 0, 0, 0, 0, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0,
                                         1+1j, 0, 0, 0, 0, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 0, 0, 0, 0])

        Ltf = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1,
                        1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        Ltf1_rx = fftshift(
            fft(preamble[n_short+np.int(n_long/5):n_short+np.int(n_long/5 + n_long*2/5)]))
        Ltf2_rx = fftshift(fft(preamble[n_short+np.int(n_long/5 + n_long*2/5):n_short+n_long]))
        Ltf_mid_rx = fftshift(
            fft(preamble[n_short + 2*L - np.int(L/2):n_short + 2*L+N - np.int(L/2)]))

        Ltf_avg_rx = (Ltf1_rx + Ltf2_rx)/2

        ind_all = np.arange(-32, 32) + (N//2)

        H_hat = np.zeros((N)) + 1j*np.zeros((N))

        # ipdb.set_trace()

        Ltf_interpolated = np.concatenate(
            (np.zeros(32*9) + 1j * np.zeros(32*9), Ltf, np.zeros(32*9) + 1j * np.zeros(32*9)))

        H_hat[ind_all] = Ltf_avg_rx[ind_all]*Ltf  # because Ltf is 1's and 0's

        h_hat = np.roll(ifft(ifftshift(H_hat)), -N//2)
        # H_1_hat[ind_all] = Ltf_1_rx[ind_all]*Ltf
        # H_2_hat[ind_all] = Ltf_2_rx[ind_all]*Ltf

        # H_hat[ind_all] = Ltf/Ltf_avg_rx[ind_all]

        # ltf_1_interpolated = ifft(ifftshift(H_1_hat*Ltf_interpolated))
        # ltf_2_interpolated = ifft(ifftshift(H_2_hat*Ltf_interpolated))
        # ltf_total = np.concatenate((ltf_1_interpolated[-N//2:], ltf_1_interpolated, ltf_2_interpolated))

        # ltf_interpolated = ifft(ifftshift(H_hat * Ltf_interpolated))

        if channel_method == 'time':
            ltf_interpolated = ifft(ifftshift(Ltf_interpolated))
            ltf_total = np.concatenate(
                (ltf_interpolated[-N//2:], ltf_interpolated, ltf_interpolated))

            Stf_64_interpolated = np.concatenate(
                (np.zeros(32*9) + 1j * np.zeros(32*9), Stf_64, np.zeros(32*9) + 1j * np.zeros(32*9)))
            stf_64_interpolated = ifft(ifftshift(Stf_64_interpolated))
            stf_total = np.concatenate(
                (stf_64_interpolated[-N//2:], stf_64_interpolated, stf_64_interpolated))

            preamble_constructed = cfo_total * (np.convolve(np.concatenate((stf_total, ltf_total)), h_hat)[
                                                N//2-1:-N//2])/rms(np.convolve(np.concatenate((stf_total, ltf_total)), h_hat)[N//2-1:-N//2])
        elif channel_method == 'frequency':
            ltf_interpolated = ifft(ifftshift(H_hat * Ltf_interpolated))
            ltf_total = np.concatenate(
                (ltf_interpolated[-N//2:], ltf_interpolated, ltf_interpolated))

            Stf_64_interpolated = np.concatenate(
                (np.zeros(32*9) + 1j * np.zeros(32*9), Stf_64, np.zeros(32*9) + 1j * np.zeros(32*9)))
            stf_64_interpolated = ifft(ifftshift(H_hat * Stf_64_interpolated))
            stf_total = np.concatenate(
                (stf_64_interpolated[-N//2:], stf_64_interpolated, stf_64_interpolated))

            preamble_constructed = cfo_total * np.concatenate((stf_total, ltf_total))

        # stf_ch_cfo = ifft(ifftshift(fftshift(fft(preamble_constructed[N//2:N+N//2]))*H_hat))
        # ltf_ch_cfo = ifft(ifftshift(fftshift(fft(preamble_constructed[n_short+N//2:n_short+N//2+N]))*H_hat))

        # stf_total_cfo_ch_added = np.concatenate((stf_ch_cfo[-N//2:], stf_ch_cfo, stf_ch_cfo))
        # ltf_total_cfo_ch_added = np.concatenate((ltf_ch_cfo[-N//2:], ltf_ch_cfo, ltf_ch_cfo))
        # preamble_constructed = np.concatenate((stf_total_cfo_ch_added, ltf_total_cfo_ch_added))

        if method == 'division':
            residuals = preamble_orig/(preamble_constructed+0.001)
        elif method == 'subtraction':
            residuals = preamble_orig - preamble_constructed

        # # ----------------------------------------------------
        # # Preamble equalization
        # # ----------------------------------------------------
        # ind_guard = np.concatenate((np.arange(-32, -26), np.arange(27, 32))) + (N//2)
        # ind_null = np.concatenate((np.array([0]), np.arange(-(N//2), -32), np.arange(32, (N//2)) )) + (N//2)
        # ind_pilots = np.array([-21, -7, 7, 21]) + (N//2)

        # mask_data = np.ones(N)
        # mask_data_pilots = np.ones(N)
        # mask_data[list(np.concatenate((ind_guard, ind_null, ind_pilots)))] = 0
        # mask_data_pilots[list(np.concatenate((ind_guard, ind_null)))] = 0
        # ind_all_all = np.arange(-(N//2), (N//2)) + N//2
        # ind_data = ind_all_all[mask_data==1]
        # ind_data_pilots = ind_all_all[mask_data_pilots==1]

        # h_hat = ifft(ifftshift(H_hat))

        # Stf_1_eq = fftshift(fft(preamble[n_short-2*N:n_short-N]))
        # Stf_2_eq = fftshift(fft(preamble[n_short-N:n_short]))
        # Ltf_1_eq = fftshift(fft(preamble[n_short+n_long-2*N:n_short+n_long-N]))
        # Ltf_2_eq = fftshift(fft(preamble[n_short+n_long-N:n_short+n_long]))

        # Stf_1_eq[ind_data_pilots] /= (H_hat[ind_data_pilots]+0.001)
        # Stf_2_eq[ind_data_pilots] /= (H_hat[ind_data_pilots]+0.001)
        # Ltf_1_eq[ind_data_pilots] /= (H_hat[ind_data_pilots]+0.001)
        # Ltf_2_eq[ind_data_pilots] /= (H_hat[ind_data_pilots]+0.001)

        # Stf_1_eq[ind_guard] = 0
        # Stf_2_eq[ind_guard] = 0
        # Ltf_1_eq[ind_guard] = 0
        # Ltf_2_eq[ind_guard] = 0

        # Stf_1_eq[ind_null] = 0
        # Stf_2_eq[ind_null] = 0
        # Ltf_1_eq[ind_null] = 0
        # Ltf_2_eq[ind_null] = 0

        # # Sanity check
        # Ltf_1_eq = Ltf
        # Ltf_2_eq = Ltf
        # Stf_1_eq = Stf_64
        # Stf_2_eq = Stf_64

        # stf_1_eq = ifft(ifftshift(Stf_1_eq))
        # stf_2_eq = ifft(ifftshift(Stf_2_eq))
        # ltf_1_eq = ifft(ifftshift(Ltf_1_eq))
        # ltf_2_eq = ifft(ifftshift(Ltf_2_eq))

        # preamble_eq = np.concatenate((stf_1_eq[:-(N//4)], stf_1_eq, stf_2_eq[:-(N//4)], stf_2_eq, ltf_1_eq[:-(N//2)], ltf_1_eq, ltf_2_eq))

    return residuals, preamble_constructed  # , h_hat, H_hat


def basic_equalize_preamble(preamble_in, fs, verbose=False, label=''):
    """
    Function that strips out the effect of the channel from the preamble. 
    It does the following:
        1. LTI channel estimation (with delay spread <= length of cyclic prefix)
        2. Remove the channel estimate from the preamble

    Inputs:
        preamble  - Preamble containing effects of the channel and Tx nonlinearities
                    (320 samples)
        ### NotImplemented: freq_offset - Dict containing freq offset

    Output: 
        preamble_eq - Preamble with the channel stripped out (320 samples)
        ### NotImplemented: preamble_eq_offset - Equalized preamble with frequency offset

    """
    # if fs!=20e6:
    #   raise NotImplementedError

    preamble = preamble_in.copy()

    if fs == 200e6:
        if preamble.size != 3200:
            raise Exception('Size of preamble is {}, but it should be 3200.'.format(preamble.size))

        n_short = 1600
        n_long = 1600

        L = 160
        N = 640

        # ----------------------------------------------------
        # Frequency offset correction
        # ----------------------------------------------------
        # sig3 = preamble[np.int(n_short/2):n_short-L].conj().copy()
        # sig4 = preamble[np.int(n_short/2)+L:n_short].copy()
        # df1 = 1/L * np.angle(sig3.dot(sig4.T))
        # preamble *= np.exp(-1j*np.arange(0, preamble.size)*df1).flatten()

        # sig5 = preamble[n_short+2*L:n_short+2*L+N].conj().copy()
        # sig6 = preamble[n_short+N+2*L:n_short+n_long].reshape(1,-1).copy()
        # df2 = 1/N * np.angle(sig5.dot(sig6.T))
        # preamble *= np.exp(-1j*np.arange(0, preamble.size)*df2).flatten()

        # ------------------------------------------------------------------------
        # LTI channel estimation (with delay spread <= length of cyclic prefix)
        # ------------------------------------------------------------------------

        Stf_64 = np.sqrt(13/6)*np.array([0, 0, 0, 0, 0, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0,
                                         1+1j, 0, 0, 0, 0, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 0, 0, 0, 0])

        Ltf = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1,
                        1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        Ltf1_rx = fftshift(
            fft(preamble[n_short+np.int(n_long/5):n_short+np.int(n_long/5 + n_long*2/5)]))
        Ltf2_rx = fftshift(fft(preamble[n_short+np.int(n_long/5 + n_long*2/5):n_short+n_long]))
        Ltf_mid_rx = fftshift(
            fft(preamble[n_short + 2*L - np.int(L/2):n_short + 2*L+N - np.int(L/2)]))

        Ltf_avg_rx = (Ltf1_rx + Ltf2_rx)/2
        # Ltf_avg_rx = Ltf1_rx
        # Ltf_avg_rx = Ltf2_rx

        # Ltf_mid_rx = Ltf_avg_rx

        # AA = np.zeros((N, N)) + 0j
        # for m in range(N):
        #   for n in range(L+1):
        #       AA[m, n] = Ltf[m] * np.exp(-1j*2*np.pi*m*n/N)
        # A = AA[:, :L+1] * np.exp(1j*np.pi*np.arange(L+1)).reshape(1, -1)

        # ind_all = np.arange(-32, 32) + 32
        # ind_guard = np.concatenate((np.arange(-32, -26), np.arange(27, 32))) + 32
        # ind_null = np.array([0]) + 32
        # mask_data_pilots = np.ones(64)
        # mask_data_pilots[list(np.concatenate((ind_guard, ind_null)))] = 0
        # ind_data_pilots = ind_all[mask_data_pilots==1]

        # h_hat_small, residuals, rank, singular_values = np.linalg.lstsq(A[ind_data_pilots,:], Ltf_mid_rx[ind_data_pilots], rcond=None)

        # h_hat = np.zeros(N)+0j
        # h_hat[:L+1] = h_hat_small
        # # h_hat = np.roll(h_hat, -np.int(L/2))
        # H_hat = fftshift(fft(h_hat))

        ind_all = np.arange(-32, 32) + (N//2)

        H_hat = np.zeros((N)) + 1j*np.zeros((N))

        # ipdb.set_trace()

        H_hat[ind_all] = Ltf_avg_rx[ind_all]*Ltf
        # H_hat[ind_all] = Ltf/Ltf_avg_rx[ind_all]

        if verbose is True:
            freq = np.arange(-32, 32)

            # H_hat_coarse = Ltf_mid_rx*Ltf
            H_hat_coarse = H_hat[ind_all]
            h_hat_coarse = ifft(ifftshift(H_hat_coarse))

            plt.figure(figsize=[10, 3])
            plt.subplot(1, 2, 1)
            plt.stem(freq, np.abs(H_hat_coarse))
            plt.grid(True)
            plt.title('Magnitude')
            plt.xlabel('Frequency bin')
            plt.subplot(1, 2, 2)
            # plt.stem(freq, np.unwrap(np.angle(H_hat)))
            plt.stem(freq, np.angle(H_hat_coarse))
            plt.title('Phase')
            plt.xlabel('Frequency bin')
            plt.suptitle('Coarse estimation'+label)
            plt.grid(True)
            plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.93])

            plt.figure(figsize=[10, 3])
            plt.subplot(1, 2, 1)
            plt.stem(np.abs(h_hat_coarse))
            plt.title('Magnitude')
            plt.xlabel('Time (in samples)')
            plt.grid(True)
            plt.subplot(1, 2, 2)
            # plt.stem(np.unwrap(np.angle(h_hat)))
            plt.stem(np.angle(h_hat_coarse))
            plt.title('Phase')
            plt.xlabel('Time (in samples)')
            plt.grid(True)
            plt.suptitle('Coarse estimation'+label)
            plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.93])

            # plt.figure(figsize=[10, 3])
            # plt.subplot(1,2,1)
            # plt.stem(freq, np.abs(H_hat))
            # plt.grid(True)
            # plt.title('Magnitude')
            # plt.xlabel('Frequency bin')
            # plt.subplot(1,2,2)
            # # plt.stem(freq, np.unwrap(np.angle(H_hat)))
            # plt.stem(freq, np.angle(H_hat))
            # plt.title('Phase')
            # plt.xlabel('Frequency bin')
            # plt.suptitle('Frequency domain least squares estimation')
            # plt.grid(True)
            # plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.9])

            # plt.figure(figsize=[10, 3])
            # plt.subplot(1,2,1)
            # plt.stem(np.abs(h_hat))
            # plt.title('Magnitude')
            # plt.xlabel('Time (in samples)')
            # plt.grid(True)
            # plt.subplot(1,2,2)
            # # plt.stem(np.unwrap(np.angle(h_hat)))
            # plt.stem(np.angle(h_hat))
            # plt.title('Phase')
            # plt.xlabel('Time (in samples)')
            # plt.grid(True)
            # plt.suptitle('Frequency domain least squares estimation')
            # plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.9])

            plt.show()

        # ----------------------------------------------------
        # Preamble equalization
        # ----------------------------------------------------
        ind_guard = np.concatenate((np.arange(-32, -26), np.arange(27, 32))) + (N//2)
        ind_null = np.concatenate(
            (np.array([0]), np.arange(-(N//2), -32), np.arange(32, (N//2)))) + (N//2)
        ind_pilots = np.array([-21, -7, 7, 21]) + (N//2)

        mask_data = np.ones(N)
        mask_data_pilots = np.ones(N)
        mask_data[list(np.concatenate((ind_guard, ind_null, ind_pilots)))] = 0
        mask_data_pilots[list(np.concatenate((ind_guard, ind_null)))] = 0
        ind_all_all = np.arange(-(N//2), (N//2)) + N//2
        ind_data = ind_all_all[mask_data == 1]
        ind_data_pilots = ind_all_all[mask_data_pilots == 1]

        Stf_1_eq = fftshift(fft(preamble[n_short-2*N:n_short-N]))
        Stf_2_eq = fftshift(fft(preamble[n_short-N:n_short]))
        Ltf_1_eq = fftshift(fft(preamble[n_short+n_long-2*N:n_short+n_long-N]))
        Ltf_2_eq = fftshift(fft(preamble[n_short+n_long-N:n_short+n_long]))

        Stf_1_eq[ind_data_pilots] /= (H_hat[ind_data_pilots]+0.001)
        Stf_2_eq[ind_data_pilots] /= (H_hat[ind_data_pilots]+0.001)
        Ltf_1_eq[ind_data_pilots] /= (H_hat[ind_data_pilots]+0.001)
        Ltf_2_eq[ind_data_pilots] /= (H_hat[ind_data_pilots]+0.001)

        Stf_1_eq[ind_guard] = 0
        Stf_2_eq[ind_guard] = 0
        Ltf_1_eq[ind_guard] = 0
        Ltf_2_eq[ind_guard] = 0

        Stf_1_eq[ind_null] = 0
        Stf_2_eq[ind_null] = 0
        Ltf_1_eq[ind_null] = 0
        Ltf_2_eq[ind_null] = 0

        # # Sanity check
        # Ltf_1_eq = Ltf
        # Ltf_2_eq = Ltf
        # Stf_1_eq = Stf_64
        # Stf_2_eq = Stf_64

        if verbose is True:

            Stf_1_eq_down = Stf_1_eq[ind_all]
            Stf_2_eq_down = Stf_2_eq[ind_all]
            Ltf_1_eq_down = Ltf_1_eq[ind_all]
            Ltf_2_eq_down = Ltf_2_eq[ind_all]

            plt.figure(figsize=[13, 4.8])
            plt.subplot(1, 3, 1)
            plt.scatter(Stf_1_eq_down.real, Stf_1_eq_down.imag)
            plt.title('Equalized STF - 1')
            plt.subplot(1, 3, 2)
            plt.scatter(Stf_2_eq_down.real, Stf_2_eq_down.imag)
            plt.title('Equalized STF - 2')
            plt.subplot(1, 3, 3)
            plt.scatter(Stf_64.real, Stf_64.imag)
            plt.title('Actual STF')
            plt.suptitle('Signal constellations')
            plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.93])

            plt.figure(figsize=[13, 4.8])
            plt.subplot(1, 3, 1)
            plt.scatter(Ltf_1_eq_down.real, Ltf_1_eq_down.imag)
            plt.title('Equalized LTF - 1')
            plt.subplot(1, 3, 2)
            plt.scatter(Ltf_2_eq_down.real, Ltf_2_eq_down.imag)
            plt.title('Equalized LTF - 2')
            plt.subplot(1, 3, 3)
            plt.scatter(Ltf.real, Ltf.imag)
            plt.title('Actual LTF')
            plt.suptitle('Signal constellations')
            plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.93])
            plt.show()

        # ipdb.set_trace()

        stf_1_eq = ifft(ifftshift(Stf_1_eq))
        stf_2_eq = ifft(ifftshift(Stf_2_eq))
        ltf_1_eq = ifft(ifftshift(Ltf_1_eq))
        ltf_2_eq = ifft(ifftshift(Ltf_2_eq))

        # preamble_eq = np.concatenate((stf_1_eq[:-(N//2)], stf_1_eq, stf_2_eq, ltf_1_eq[:-(N//2)], ltf_1_eq, ltf_2_eq))
        preamble_eq = np.concatenate(
            (stf_1_eq[-(N//4):], stf_1_eq, stf_2_eq[-(N//4):], stf_2_eq, ltf_1_eq[-(N//2):], ltf_1_eq, ltf_2_eq))

        # import pdb
        # pdb.set_trace()

        # shift = freq_offset['shift_coarse']
        # df1 = freq_offset['carrier_coarse']
        # df2 = freq_offset['carrier_fine']

        # preamble_eq_offset = preamble_eq.copy()

        # Add in coarse carrier freq offset, taking the shift into account
        # if shift>=0:
        #   preamble_eq_offset[shift:] = preamble_eq[shift:] * np.exp(1j*np.arange(0,preamble_eq.size - shift)*df1).flatten()
        # else:
        #   preamble_eq_offset= preamble_eq * np.exp(1j*(np.arange(0, preamble_eq.size)+shift)*df1).flatten()

        # # Add in fine carrier freq offset
        # preamble_eq_offset *= np.exp(1j*np.arange(0, preamble_eq.size)*df2).flatten()

        # return preamble_eq, preamble_eq_offset

    elif fs == 20e6:

        if preamble.size != 320:
            raise Exception('Size of preamble is {}, but it should be 320.'.format(preamble.size))

        n_short = 160
        n_long = 160

        # ----------------------------------------------------
        # Frequency offset correction
        # ----------------------------------------------------
        # sig3 = preamble[np.int(n_short/2):n_short-16].conj().copy()
        # sig4 = preamble[np.int(n_short/2)+16:n_short].copy()
        # df1 = 1/16 * np.angle(sig3.dot(sig4.T))
        # preamble *= np.exp(-1j*np.arange(0, preamble.size)*df1).flatten()

        # sig5 = preamble[n_short+32:n_short+32+64].conj().copy()
        # sig6 = preamble[n_short+64+32:n_short+n_long].reshape(1,-1).copy()
        # df2 = 1/64 * np.angle(sig5.dot(sig6.T))
        # preamble *= np.exp(-1j*np.arange(0, preamble.size)*df2).flatten()

        # ------------------------------------------------------------------------
        # LTI channel estimation (with delay spread <= length of cyclic prefix)
        # ------------------------------------------------------------------------

        Stf_64 = np.sqrt(13/6)*np.array([0, 0, 0, 0, 0, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0,
                                         1+1j, 0, 0, 0, 0, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 0, 0, 0, 0])

        Ltf = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1,
                        1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        L = 16
        N = 64

        Ltf1_rx = fftshift(
            fft(preamble[n_short+np.int(n_long/5):n_short+np.int(n_long/5 + n_long*2/5)]))
        Ltf2_rx = fftshift(fft(preamble[n_short+np.int(n_long/5 + n_long*2/5):n_short+n_long]))
        Ltf_mid_rx = fftshift(
            fft(preamble[n_short + 2*L - np.int(L/2):n_short + 2*L+N - np.int(L/2)]))
        Ltf_avg_rx = (Ltf1_rx + Ltf2_rx)/2

        # Ltf_mid_rx = Ltf_avg_rx

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
        ind_data_pilots = ind_all[mask_data_pilots == 1]

        h_hat_small, residuals, rank, singular_values = np.linalg.lstsq(
            A[ind_data_pilots, :], Ltf_mid_rx[ind_data_pilots], rcond=None)

        h_hat = np.zeros(N)+0j
        h_hat[:L+1] = h_hat_small
        # h_hat = np.roll(h_hat, -np.int(L/2))
        H_hat = fftshift(fft(h_hat))

        H_hat = Ltf_avg_rx*Ltf

        if verbose is True:
            freq = np.arange(-32, 32)

            H_hat_coarse = Ltf_mid_rx*Ltf
            h_hat_coarse = ifft(ifftshift(H_hat_coarse))

            plt.figure(figsize=[10, 3])
            plt.subplot(1, 2, 1)
            plt.stem(freq, np.abs(H_hat_coarse))
            plt.grid(True)
            plt.title('Magnitude')
            plt.xlabel('Frequency bin')
            plt.subplot(1, 2, 2)
            # plt.stem(freq, np.unwrap(np.angle(H_hat)))
            plt.stem(freq, np.angle(H_hat_coarse))
            plt.title('Phase')
            plt.xlabel('Frequency bin')
            plt.suptitle('Coarse estimation')
            plt.grid(True)
            plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.93])

            plt.figure(figsize=[10, 3])
            plt.subplot(1, 2, 1)
            plt.stem(np.abs(h_hat_coarse))
            plt.title('Magnitude')
            plt.xlabel('Time (in samples)')
            plt.grid(True)
            plt.subplot(1, 2, 2)
            # plt.stem(np.unwrap(np.angle(h_hat)))
            plt.stem(np.angle(h_hat_coarse))
            plt.title('Phase')
            plt.xlabel('Time (in samples)')
            plt.grid(True)
            plt.suptitle('Coarse estimation')
            plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.93])

            plt.figure(figsize=[10, 3])
            plt.subplot(1, 2, 1)
            plt.stem(freq, np.abs(H_hat))
            plt.grid(True)
            plt.title('Magnitude')
            plt.xlabel('Frequency bin')
            plt.subplot(1, 2, 2)
            # plt.stem(freq, np.unwrap(np.angle(H_hat)))
            plt.stem(freq, np.angle(H_hat))
            plt.title('Phase')
            plt.xlabel('Frequency bin')
            plt.suptitle('Frequency domain least squares estimation')
            plt.grid(True)
            plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.9])

            plt.figure(figsize=[10, 3])
            plt.subplot(1, 2, 1)
            plt.stem(np.abs(h_hat))
            plt.title('Magnitude')
            plt.xlabel('Time (in samples)')
            plt.grid(True)
            plt.subplot(1, 2, 2)
            # plt.stem(np.unwrap(np.angle(h_hat)))
            plt.stem(np.angle(h_hat))
            plt.title('Phase')
            plt.xlabel('Time (in samples)')
            plt.grid(True)
            plt.suptitle('Frequency domain least squares estimation')
            plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.9])

            # plt.show()

        # ----------------------------------------------------
        # Preamble equalization
        # ----------------------------------------------------
        ind_all = np.arange(-32, 32) + 32
        ind_guard = np.concatenate((np.arange(-32, -26), np.arange(27, 32))) + 32
        ind_null = np.array([0]) + 32
        ind_pilots = np.array([-21, -7, 7, 21]) + 32
        mask_data = np.ones(64)
        mask_data_pilots = np.ones(64)
        mask_data[list(np.concatenate((ind_guard, ind_null, ind_pilots)))] = 0
        mask_data_pilots[list(np.concatenate((ind_guard, ind_null)))] = 0
        ind_data = ind_all[mask_data == 1]
        ind_data_pilots = ind_all[mask_data_pilots == 1]

        Stf_1_eq = fftshift(fft(preamble[n_short-2*N:n_short-N]))
        Stf_2_eq = fftshift(fft(preamble[n_short-N:n_short]))
        Ltf_1_eq = fftshift(fft(preamble[n_short+n_long-2*N:n_short+n_long-N]))
        Ltf_2_eq = fftshift(fft(preamble[n_short+n_long-N:n_short+n_long]))

        Stf_1_eq[ind_data_pilots] /= H_hat[ind_data_pilots]
        Stf_2_eq[ind_data_pilots] /= H_hat[ind_data_pilots]
        Ltf_1_eq[ind_data_pilots] /= H_hat[ind_data_pilots]
        Ltf_2_eq[ind_data_pilots] /= H_hat[ind_data_pilots]

        Stf_1_eq[ind_guard] = 0
        Stf_2_eq[ind_guard] = 0
        Ltf_1_eq[ind_guard] = 0
        Ltf_2_eq[ind_guard] = 0

        Stf_1_eq[ind_null] = 0
        Stf_2_eq[ind_null] = 0
        Ltf_1_eq[ind_null] = 0
        Ltf_2_eq[ind_null] = 0

        # # Sanity check
        # Ltf_1_eq = Ltf
        # Ltf_2_eq = Ltf
        # Stf_1_eq = Stf_64
        # Stf_2_eq = Stf_64

        if verbose is True:

            plt.figure(figsize=[13, 4.8])
            plt.subplot(1, 3, 1)
            plt.scatter(Stf_1_eq.real, Stf_1_eq.imag)
            plt.title('Equalized STF - 1')
            plt.subplot(1, 3, 2)
            plt.scatter(Stf_2_eq.real, Stf_2_eq.imag)
            plt.title('Equalized STF - 2')
            plt.subplot(1, 3, 3)
            plt.scatter(Stf_64.real, Stf_64.imag)
            plt.title('Actual STF')
            plt.suptitle('Signal constellations')
            plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.93])

            plt.figure(figsize=[13, 4.8])
            plt.subplot(1, 3, 1)
            plt.scatter(Ltf_1_eq.real, Ltf_1_eq.imag)
            plt.title('Equalized LTF - 1')
            plt.subplot(1, 3, 2)
            plt.scatter(Ltf_2_eq.real, Ltf_2_eq.imag)
            plt.title('Equalized LTF - 2')
            plt.subplot(1, 3, 3)
            plt.scatter(Ltf.real, Ltf.imag)
            plt.title('Actual LTF')
            plt.suptitle('Signal constellations')
            plt.tight_layout(rect=[0.01, 0.03, 0.98, 0.93])
            plt.show()

        stf_1_eq = ifft(ifftshift(Stf_1_eq))
        stf_2_eq = ifft(ifftshift(Stf_2_eq))
        ltf_1_eq = ifft(ifftshift(Ltf_1_eq))
        ltf_2_eq = ifft(ifftshift(Ltf_2_eq))

        preamble_eq = np.concatenate(
            (stf_1_eq[-32:], stf_1_eq, stf_2_eq, ltf_1_eq[-32:], ltf_1_eq, ltf_2_eq))

        # shift = freq_offset['shift_coarse']
        # df1 = freq_offset['carrier_coarse']
        # df2 = freq_offset['carrier_fine']

        # preamble_eq_offset = preamble_eq.copy()

        # Add in coarse carrier freq offset, taking the shift into account
        # if shift>=0:
        #   preamble_eq_offset[shift:] = preamble_eq[shift:] * np.exp(1j*np.arange(0,preamble_eq.size - shift)*df1).flatten()
        # else:
        #   preamble_eq_offset= preamble_eq * np.exp(1j*(np.arange(0, preamble_eq.size)+shift)*df1).flatten()

        # # Add in fine carrier freq offset
        # preamble_eq_offset *= np.exp(1j*np.arange(0, preamble_eq.size)*df2).flatten()

        # return preamble_eq, preamble_eq_offset

    return preamble_eq


def rms(x):
    # Root mean squared value
    return np.sqrt(np.mean(x * np.conjugate(x)))


def shift_frequency(vector, freq_shift, fs):
    # Shift frequency of time-series signal by specified amount
    #  vector: complex time-series signal
    #  freq_shift: frequency shift amount
    #  fs: sampling frequency of complex signal

    t = np.arange(0, np.size(vector)) / fs     # define time axis

    # Sqrt(2) factor ensures that the power of the frequency downconverted signal
    # is equal to the power of its passband counterpart
    modulation = np.exp(-1j * 2 * np.pi * freq_shift * t) / np.sqrt(2)     # frequency shift factor

    return vector * modulation     # baseband signal


def resample(vector, fs, dfs):
    # Resample signal from original sample rate to desired sample rate
    #   fs: original sampling frequency
    #   dfs: desired sampling frequency

    fs = int(round(fs))     # convert to integers
    dfs = int(round(dfs))
    cfs = lcm(fs, dfs)     # common sampling frequency

    if cfs > fs:
        # Upsample from start-Hz to common-Hz
        vector = resampy.resample(vector, fs, cfs, filter='kaiser_best')

    # Downsample from common-Hz to desired-Hz
    return resampy.resample(vector, cfs, dfs, filter='kaiser_best')


def lcm(a, b):
    # Least common multiple of a and b
    return a * int(b / fractions.gcd(a, b)) if a and b else 0


def get_sliding_window(x, window_size=10, stride=1, fs=200e6, fs_natural=20e6):
    shape_ = x.shape

    window_size_samples = np.int(window_size * (fs/fs_natural))
    stride_samples = np.int(stride * (fs/fs_natural))

    # sliding_window = [None] * ((shape_[1]-100+10)//10)

    for i in tqdm(np.arange(0, shape_[1] - window_size_samples + stride_samples, stride_samples)):
        if i == 0:
            y = x[:, i:i + window_size_samples, :].copy()
        else:
            y = np.concatenate((y, x[:, i:i + window_size_samples, :]), axis=0)

    return y


def read_wifi(files, base_data_directory, device_map, progress=True):
    '''
    Read wifi data frin data directory
    '''

    csv = files['csv_objects'].items()
    if progress is True:
        csv = tqdm(csv)

    data_dict = dict(signal={}, device_key={},  # Complex signal and device label [0, N-1] from device_map
                     sample_rate={}, capture_sample_rate={}, capture_frequency={}, capture_hw={},
                     center_frequency={}, freq_lower_edge={}, freq_upper_edge={},
                     reference_number={}, data_file={}, sample_start={}, sample_count={},
                     device_type={}, device_id={}, device_manufacturer={}
                     )

    signal_index = 0
    for file, signal_list in csv:
        # Example:
        # file = 'adsb_gfi_3_dataset/10_sigmf_files_dataset/A-23937.sigmf-data'
        # signal_list = ['A-23937-34', 'A-23937-54']

        # check to see if the first character in "file" is a slash:
        while file[0] == '/' or file[0] == '\\':
            file = file[1:]
        # if 'Windows' in platform():
        #   file = file.replace("/", "\\")

        data_file = os.path.join(base_data_directory, file)
        metadata_file = data_file.replace('sigmf-data', 'sigmf-meta')

        all_signals = json.load(open(metadata_file))
        capture = dict(capture_sample_rate=all_signals['global']['core:sample_rate'],
                       sample_rate=all_signals['global']['core:sample_rate'],
                       capture_hw=all_signals['global']['core:hw'],
                       capture_frequency=all_signals['capture'][0]['core:frequency'],
                       data_file=data_file)

        for signal_name in signal_list:
            # data_dict['reference_number'][signal_index] = signal_name

            for key, value in capture.items():
                data_dict[key][signal_index] = value

            capture_properties = all_signals['capture']
            signal_properties = get_json_signal(
                all_signals['annotations'], capture_properties[0], signal_name, type='wifi')

            for key, value in signal_properties.items():
                data_dict[key][signal_index] = value
            device_id = signal_properties['device_id']
            data_dict['device_key'][signal_index] = device_map[device_id]

            filename = data_dict['data_file'][signal_index]
            start_sample = data_dict['sample_start'][signal_index]
            sample_count = data_dict['sample_count'][signal_index]
            data, buffer_start, buffer_end = read_sample(
                filename, start_sample, sample_count, desired_buffer=0)
            data_dict['signal'][signal_index] = data

            data_dict['center_frequency'][signal_index] = data_dict['capture_frequency'][signal_index]

            # ipdb.set_trace()

            signal_index = signal_index + 1

    return data_dict


def parse_input_files(input_csv, devices_csv):
    '''
    Parser for wifi dataset
    '''
    device_list = []  # a list of the devices to be trained/tested with
    device_map = {}   # a reverse map from device name to index
    csv_objects = {}  # a dictionary with filenames for keys, lists of signals as values

    with open(devices_csv) as devices_csv_file:
        devices_reader = csv.reader(devices_csv_file, delimiter=',')
        for device in devices_reader:
            device_list.append(device[0])

    for i, device in enumerate(device_list):
        device_map[device] = i

    with open(input_csv) as input_csv_file:
        input_reader = csv.reader(input_csv_file, delimiter=',')
        for row in input_reader:
            csv_objects[row[0]] = row[1:]

    return {'device_list': device_list,
            'device_map': device_map,
            'csv_objects': csv_objects}


def get_json_signal(json_annotations, capture, signal_id, type=None):
    '''
    Get signal from json
    '''

    for signal in json_annotations:
        if signal != {} and signal['capture_details:signal_reference_number'] == signal_id:
            if 'rfml:label' in signal:
                signal_label = signal['rfml:label']
                if type is None:
                    type = signal_label[0]
            else:
                signal_label = tuple(None, None, None)
                if type is None:
                    type = "unknown"

            if type == "wifi":
                return {'freq_lower_edge': signal['core:freq_lower_edge'],
                        'freq_upper_edge': signal['core:freq_upper_edge'],
                        'sample_start': signal['core:sample_start'],
                        'sample_count': signal['core:sample_count'],
                        'device_type': signal_label[0],
                        'device_manufacturer': signal_label[1],
                        'device_id': signal_label[2]}
            elif type == "ADS-B":
                return{'snr': signal['capture_details:SNRdB'],
                       'reference_number': signal['capture_details:signal_reference_number'],
                       'freq_lower_edge': capture['core:freq_lower_edge'],
                       'freq_upper_edge': capture['core:freq_upper_edge'],
                       'sample_start': signal['core:sample_start'],
                       'sample_count': signal['core:sample_count'],
                       'device_type': signal_label[0],
                       'device_id': signal_label[1]}
            else:
                print('Unknown signal type', type)
                return None
    return None


def read_sample(filename, start_sample, sample_count, desired_buffer):
    ''' 
    Read samples
    '''

    buffer_start = min(desired_buffer, start_sample)
    buffer_end = desired_buffer
    sample_count += (buffer_start + buffer_end)

    with open(filename, "rb") as f:
        # Seek to startSample
        f.seek((start_sample - buffer_start) * 4)  # 4bytes per sample (2x16 bit ints)

        # Read in as ints
        raw = np.fromfile(f, dtype='int16', count=2*sample_count)

        samples_read = int(raw.size / 2)
        buffer_end -= (sample_count - samples_read)

        # Convert interleaved ints into two planes, real and imaginary
        array = raw.reshape([samples_read, 2])

        # convert the array to complex
        array = array[:, 0] + 1j*array[:, 1]

        return array, buffer_start, buffer_end
