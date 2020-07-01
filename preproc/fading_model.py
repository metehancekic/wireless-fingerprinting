"""
Functions to add artificial channel to data
"""

import numpy as np
import numpy.random as random
from scipy.signal import resample
from tqdm import tqdm
import math

from .preproc_wifi import rms


def add_freq_offset(x, fc, fs=20e6, df=20e-6, rand=False):
    '''
    Adds frequency offset to given signal
    Input:     x  : input signal
               fc : carrier frequency
               fs : sampling frequency
               df : oscillator precision tolerance
               rand : which method to use to crate CFOs
    Output:    rotated : CFO added signal
    Description:

        rotated[t] = x[t] * exp(-j*2*pi*df*(fc/fs)*t)

    '''
    complex_data = x[..., 0].copy() + 1j * x[..., 1].copy()  # (N, T)
    N = complex_data.shape[0]
    T = complex_data.shape[1]

    if rand is False:
        cfo = df * fc  # size N
        theta = (-2*np.pi*cfo/fs).reshape(-1, 1)  # (N, 1)
    elif rand == 'unif':
        rv = np.random.RandomState(seed=None)
        cfo = df * fc  # size N
        theta = rv.uniform(low=-2*np.pi*cfo/fs, high=2*np.pi*cfo/fs).reshape(-1, 1)  # (N, 1)
    elif rand == 'ber':
        rv = np.random.RandomState(seed=None)
        theta = np.zeros([N, 1])
        for i in range(N):
            cfo = df * fc[i]  # size 1
            theta[i] = rv.choice(a=np.array([-2*np.pi*cfo/fs, 2*np.pi*cfo/fs]))

    # theta += -2*np.pi*N*cfo/fs*2

    exp_offset = np.exp(1j*theta.dot(np.arange(T).reshape(1, -1)))  # (N, T)
    complex_data *= exp_offset

    rotated = np.concatenate((complex_data.real[..., None], complex_data.imag[..., None]), axis=-1)

    return rotated


def add_custom_fading_channel(frame,
                              snr=500,
                              sampling_rate=20e6,
                              seed=0,
                              beta=None,
                              delay_seed=None,
                              channel_type=1,
                              channel_method='FFT',
                              noise_method='reg'):

    epa_delay = np.array([0, 30, 70, 90, 110, 190, 410], dtype=np.float64)
    epa_power = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])

    eva_delay = np.array([0, 30, 150, 310, 370, 710, 1090, 1730, 2510], dtype=np.float64)
    eva_power = np.array([0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9])

    etu_delay = np.array([0, 50, 120, 200, 230, 500, 1600, 2300, 5000], dtype=np.float64)
    etu_power = np.array([-1.0, -1.0, -1.0, -0.0, -0.0, -0.0, -3.0, -5.0, -7.0])

    if channel_type == 1:
        delay_ns = epa_delay
        power_dB = epa_power
    elif channel_type == 2:
        delay_ns = eva_delay
        power_dB = eva_power
    elif channel_type == 3:
        delay_ns = etu_delay
        power_dB = etu_power

    if delay_seed is not False:
        rv_delay = random.RandomState(seed=delay_seed)
        delay_ns += rv_delay.uniform(low=-50, high=50, size=delay_ns.size)

    rv_channel = random.RandomState(seed=seed)

    A = rv_channel.normal(loc=0, scale=1, size=(len(power_dB), 2)).dot(np.array([1, 1j]))
    A *= np.sqrt((10**(power_dB/10))/2)

    if channel_method == 'FFT':
        N = len(frame)
        M = int(np.ceil(max(delay_ns)*sampling_rate*1e-9))  # 9 - length of channel in symbols

        # (M+N, 1) -> set of FFT frequencies corresponding to [0, 1, ... M+N-1]
        f = np.arange(M+N)[:, None]*sampling_rate/(M+N)
        tau = np.array(delay_ns)[:, None].transpose()*1e-9  # (1, 7) -> delay_ns converted to s
        H = np.exp(-1j*2*np.pi*(f.dot(tau)))  # (M+N, 7) -> FFT of A

        X = np.fft.fft(frame, n=N+M, axis=0)
        Y = (X[:, None]*H).dot(A)
        frame_faded = np.fft.ifft(Y)[:N]

    elif channel_method == 'randn':
        N = len(frame)
        M = int(np.ceil(max(delay_ns)*sampling_rate*1e-9))  # 9 - length of channel in symbols

        rv_rand = random.RandomState(seed=seed)
        A = rv_rand.normal(loc=0, scale=1, size=(M+N, 2)).dot(np.array([1, 1j]))

        # import ipdb; ipdb.set_trace()
        X = np.fft.fft(frame, n=N+M, axis=0)
        Y = np.multiply(X[:], A)
        frame_faded = np.fft.ifft(Y)[:N]

    elif channel_method == 'RC':
        Fs = 200e6
        Tsym = 1/(20e6)
        # N = 640
        N = 320

        # Cascade channel with raised cosine pulse @ 200MHz
        h_epa_rc = np.zeros(N)+0j
        for i in range(7):
            h_rc_i = raised_cosine(shift=delay_ns[i]*1e-9, N=N, beta=beta, Tsym=Tsym, Fs=Fs)
            h_epa_rc += h_rc_i * A[i]

        # Subsample to desired sampling rate and convolve with signal
        if Fs % sampling_rate == 0:
            h = h_epa_rc[::np.int(Fs/sampling_rate)]
        else:
            # Use FFT interpolation to resample
            h = resample(h_epa_rc, np.int(N/Fs/sampling_rate))
        frame_faded = np.convolve(frame, h, mode='full')[np.int(h.size/2):-np.int(h.size/2)+1]

    if snr < 500:
        rv_noise = random.RandomState(seed=None)
        E_b = (np.abs(frame_faded)**2).mean()
        N_0 = E_b/(10**(snr/10))

        if noise_method == 'reg':  # regular
            N_0 *= sampling_rate/20e6
            N = len(frame)
            n = rv_noise.normal(loc=0, scale=np.sqrt(N_0/2), size=(N, 2)).dot(np.array([1, 1j]))
            frame_faded += n

        elif noise_method == 'bl':  # bandlimited
            Fs = 200e6
            # Generate noise @ 200 Msps, and set SNR
            N_0 *= 10
            M = np.int(frame.size * Fs/sampling_rate)
            n = rv_noise.normal(loc=0, scale=np.sqrt(N_0/2), size=(M, 2)).dot(np.array([1, 1j]))

            # Cascade noise with root raised cosine pulse @ 200MHz
            h_rrc = root_raised_cosine(shift=0, N=M, beta=beta, Tsym=Tsym, Fs=Fs)
            h_rrc /= np.sqrt((np.abs(h_rrc)**2).sum())
            n = np.convolve(n, h_rrc, mode='full')[np.int(M/2):-np.int(M/2)+1]

            # Subsample noise and add to signal
            if Fs % sampling_rate == 0:
                frame_faded += n[::np.int(Fs/sampling_rate)]
            else:
                frame_faded += resample(n, np.int(frame.size))

    return frame_faded


def add_noise(data_dict, snr=20, progress=True):

    signal_indices = range(len(data_dict['data_file']))
    if progress is True:
        signal_indices = tqdm(signal_indices)

    for i in signal_indices:
        sampling_rate = data_dict['sample_rate'][i]
        signal = data_dict['signal'][i]
        # signal = signal[:np.int(math.ceil(sample_duration*sampling_rate))]

        rv_noise = random.RandomState(seed=None)
        E_b = (np.abs(signal)**2).mean()
        N_0 = E_b/(10**(snr/10))
        N = len(signal)
        n = rv_noise.normal(loc=0, scale=np.sqrt(N_0/2), size=(N, 2)).dot(np.array([1, 1j]))
        signal += n

        signal = normalize(signal)
        data_dict['signal'][i] = signal

    return data_dict


def add_fading_channel(data_dict, seed=0, snr=20, beta=0.5, num_ch=1, progress=True, sample_duration=20.):

    signal_indices = range(len(data_dict['data_file']))
    if progress is True:
        signal_indices = tqdm(signal_indices)

    if beta <= 1:
        if num_ch is None:
            for i in signal_indices:
                sampling_rate = data_dict['sample_rate'][i]
                signal = data_dict['signal'][i]
                signal = signal[:np.int(math.ceil(sample_duration*sampling_rate))]

                signal_faded = add_fading_channel_to_frame(
                    signal, snr, sampling_rate, seed=None, beta=beta)
                signal_faded = normalize(signal_faded)
                data_dict['signal'][i] = signal_faded
        elif num_ch == 1:
            for i in signal_indices:
                sampling_rate = data_dict['sample_rate'][i]
                signal = data_dict['signal'][i]
                signal = signal[:np.int(math.ceil(sample_duration*sampling_rate))]

                signal_faded = add_fading_channel_to_frame(
                    signal, snr, sampling_rate, seed=seed, beta=beta)
                signal_faded = normalize(signal_faded)
                data_dict['signal'][i] = signal_faded
        else:
            seed = (seed+10)*100
            for i in signal_indices:
                seed_i = seed + (i % num_ch)
                # print(seed_i)

                sampling_rate = data_dict['sample_rate'][i]
                signal = data_dict['signal'][i]
                signal = signal[:np.int(math.ceil(sample_duration*sampling_rate))]

                signal_faded = add_fading_channel_to_frame(
                    signal, snr, sampling_rate, seed=seed_i, beta=beta)
                signal_faded = normalize(signal_faded)
                data_dict['signal'][i] = signal_faded
    else:
        print('Using FFT method to add channel\n')
        if num_ch is None:
            for i in signal_indices:
                sampling_rate = data_dict['sample_rate'][i]
                signal = data_dict['signal'][i]
                signal = signal[:np.int(math.ceil(sample_duration*sampling_rate))]

                signal_faded = fft_add_fading_channel_to_frame(
                    signal, snr, sampling_rate, seed=None, beta=beta)
                signal_faded = normalize(signal_faded)
                data_dict['signal'][i] = signal_faded
        elif num_ch == 1:
            for i in signal_indices:
                sampling_rate = data_dict['sample_rate'][i]
                signal = data_dict['signal'][i]
                signal = signal[:np.int(math.ceil(sample_duration*sampling_rate))]

                signal_faded = fft_add_fading_channel_to_frame(
                    signal, snr, sampling_rate, seed=seed, beta=beta)
                signal_faded = normalize(signal_faded)
                data_dict['signal'][i] = signal_faded
        else:
            seed = (seed+10)*100
            for i in signal_indices:
                seed_i = seed + (i % num_ch)
                # print(seed_i)

                sampling_rate = data_dict['sample_rate'][i]
                signal = data_dict['signal'][i]
                signal = signal[:np.int(math.ceil(sample_duration*sampling_rate))]

                signal_faded = fft_add_fading_channel_to_frame(
                    signal, snr, sampling_rate, seed=seed_i, beta=beta)
                signal_faded = normalize(signal_faded)
                data_dict['signal'][i] = signal_faded

    return data_dict


def fft_add_fading_channel_to_frame(frame, snr, sampling_rate, seed=0, beta=None):
    if sampling_rate != 20e6:
        raise ValueError('Fs must be 20e6 to use FFT method')
    Fs = 20e6
    # Tsym = 1/(20e6)

    rv_channel = random.RandomState(seed=seed)
    # rv_noise = random.RandomState(seed=None)
    # epa_delay = np.array([0, 30, 70, 90, 110, 190, 410])
    # epa_power = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])
    # h_epa = rv_channel.normal(loc=0, scale=1, size=(7, 2)).dot(np.array([1, 1j]))
    # h_epa *= np.sqrt((10**(epa_power/10))/2)

    delay_ns = np.array([0, 30, 70, 90, 110, 190, 410])
    power_dB = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])
    # A0 = (np.random.randn(len(self.power_dB)) + 1j*np.random.randn(len(self.power_dB)))/np.sqrt(2)
    # A = (10**(np.array(np.array(power_dB)/20)))*A0

    A = rv_channel.normal(loc=0, scale=1, size=(len(power_dB), 2)).dot(np.array([1, 1j]))
    A *= np.sqrt((10**(power_dB/10))/2)

    N = len(frame)
    M = int(np.ceil(max(delay_ns)*Fs*1e-9))  # 9 - length of channel in symbols

    # (M+N, 1) -> set of FFT frequencies corresponding to [0, 1, ... M+N-1]
    f = np.arange(M+N)[:, None]*Fs/(M+N)
    tau = np.array(delay_ns)[:, None].transpose()*1e-9  # (1, 7) -> delay_ns converted to s
    H = np.exp(-1j*2*np.pi*(f.dot(tau)))  # (M+N, 7) -> FFT of A

    X = np.fft.fft(frame, n=N+M, axis=0)
    Y = (X[:, None]*H).dot(A)
    frame_faded = np.fft.ifft(Y)[:N]

    if snr < 500:
        rv_noise = random.RandomState(seed=None)
        E_b = (np.abs(frame_faded)**2).mean()
        N_0 = E_b/(10**(snr/10))
        n = rv_noise.normal(loc=0, scale=np.sqrt(N_0/2), size=(N, 2)).dot(np.array([1, 1j]))
        frame_faded += n

    return frame_faded


def fft_add_fading_channel_to_frame_200(frame, snr, sampling_rate, seed=0, beta=None):
    if sampling_rate != 200e6:
        raise ValueError('Fs must be 20e6 to use FFT method')
    Fs = 2e6
    # Tsym = 1/(20e6)

    rv_channel = random.RandomState(seed=seed)
    # rv_noise = random.RandomState(seed=None)
    # epa_delay = np.array([0, 30, 70, 90, 110, 190, 410])
    # epa_power = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])
    # h_epa = rv_channel.normal(loc=0, scale=1, size=(7, 2)).dot(np.array([1, 1j]))
    # h_epa *= np.sqrt((10**(epa_power/10))/2)

    delay_ns = np.array([0, 30, 70, 90, 110, 190, 410])
    power_dB = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])
    # A0 = (np.random.randn(len(self.power_dB)) + 1j*np.random.randn(len(self.power_dB)))/np.sqrt(2)
    # A = (10**(np.array(np.array(power_dB)/20)))*A0

    A = rv_channel.normal(loc=0, scale=1, size=(len(power_dB), 2)).dot(np.array([1, 1j]))
    A *= np.sqrt((10**(power_dB/10))/2)

    N = len(frame)
    M = int(np.ceil(max(delay_ns)*Fs*1e-9))  # 9 - length of channel in symbols

    # (M+N, 1) -> set of FFT frequencies corresponding to [0, 1, ... M+N-1]
    f = np.arange(M+N)[:, None]*Fs/(M+N)
    tau = np.array(delay_ns)[:, None].transpose()*1e-9  # (1, 7) -> delay_ns converted to s
    H = np.exp(-1j*2*np.pi*(f.dot(tau)))  # (M+N, 7) -> FFT of A

    X = np.fft.fft(frame, n=N+M, axis=0)
    Y = (X[:, None]*H).dot(A)
    frame_faded = np.fft.ifft(Y)[:N]

    if snr < 500:
        rv_noise = random.RandomState(seed=None)
        E_b = (np.abs(frame_faded)**2).mean()
        N_0 = E_b/(10**(snr/10))
        n = rv_noise.normal(loc=0, scale=np.sqrt(N_0/2), size=(N, 2)).dot(np.array([1, 1j]))
        frame_faded += n

    return frame_faded


def add_fading_channel_to_frame(frame, snr, sampling_rate, seed=0, beta=0.5):
    """
    Adds fading channel to frame.

    Inputs:
        frame           - Frame from WiFi-2 dataset
        snr             - Desired SNR in dB
        sampling_rate   - Desired sampling rate
        seed            - Seed for random number generator
    Output:
        frame_faded - Frame with channel and noise
    """
    rv_channel = random.RandomState(seed=seed)
    rv_noise = random.RandomState(seed=None)
    Fs = 200e6
    Tsym = 1/(20e6)
    # N = 640
    N = 320

    # Extended Pedestrian A channel @ 200 Msps
    epa_delay = np.array([0, 30, 70, 90, 110, 190, 410])
    epa_power = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])
    h_epa = rv_channel.normal(loc=0, scale=1, size=(7, 2)).dot(np.array([1, 1j]))
    h_epa *= np.sqrt((10**(epa_power/10))/2)

    # Cascade channel with raised cosine pulse
    h_epa_rc = np.zeros(N)+0j
    for i in range(7):
        h_rc_i = raised_cosine(shift=epa_delay[i]*1e-9, N=N, beta=beta, Tsym=Tsym, Fs=Fs)
        h_epa_rc += h_rc_i * h_epa[i]

    # Subsample to desired sampling rate and convolve with signal
    if Fs % sampling_rate == 0:
        h = h_epa_rc[::np.int(Fs/sampling_rate)]
    else:
        # Use FFT interpolation to resample
        h = resample(h_epa_rc, np.int(N/Fs/sampling_rate))
    frame_faded = np.convolve(frame, h, mode='full')[np.int(h.size/2):-np.int(h.size/2)+1]

    if snr < 500:
        # Generate noise @ 200 Msps, and set SNR
        E_b = (np.abs(frame_faded)**2).mean()
        N_0 = 10*E_b/(10**(snr/10))
        M = np.int(frame.size * Fs/sampling_rate)
        n = rv_noise.normal(loc=0, scale=np.sqrt(N_0/2), size=(M, 2)).dot(np.array([1, 1j]))

        # Cascade noise with root raised cosine pulse
        h_rrc = root_raised_cosine(shift=0, N=M, beta=beta, Tsym=Tsym, Fs=Fs)
        h_rrc /= np.sqrt((np.abs(h_rrc)**2).sum())
        n = np.convolve(n, h_rrc, mode='full')[np.int(M/2):-np.int(M/2)+1]

        # Subsample noise and add to signal
        if Fs % sampling_rate == 0:
            frame_faded += n[::np.int(Fs/sampling_rate)]
        else:
            frame_faded += resample(n, np.int(frame.size))

    return frame_faded


def raised_cosine(shift, N=1600, beta=0.5, Tsym=4e-6, Fs=200e6):
    """
    Raised cosine pulse
    """
    time_idx = np.arange(-N/2, N/2)/Fs
    h_rc = np.zeros(time_idx.shape)
    if beta != 0:
        for i in range(time_idx.size):
            t = time_idx[i] - shift
            if t == 0:
                h_rc[i] = 1
            elif (t == Tsym/(2*beta)) or (t == -Tsym/(2*beta)):
                # h_rc[i] = (np.pi/4)*np.sinc(1/(2*beta))
                h_rc[i] = (beta/2)*np.sin(np.pi/(2*beta))
            else:
                # h_rc[i] = np.sinc(t/Tsym) * np.cos(np.pi*beta*t/Tsym) /(1* (1-(2*beta*t/Tsym)**2))
                h_rc[i] = np.sin(np.pi*t/Tsym) * np.cos(np.pi*beta*t/Tsym) / \
                    ((np.pi*t/Tsym) * (1-(2*beta*t/Tsym)**2))
    else:
        h_rc = np.sinc((time_idx-shift)/Tsym)
    return h_rc


def root_raised_cosine(shift, N=1600, beta=0.5, Tsym=4e-6, Fs=200e6):
    """
    Root raised cosine pulse
    """
    time_idx = np.arange(-N/2, N/2)/Fs
    h_rrc = np.zeros(time_idx.shape)
    for i in range(time_idx.size):
        t = time_idx[i] - shift
        if t == 0.0:
            h_rrc[i] = 1.0 - beta + (4*beta/np.pi)
        elif beta != 0 and t == Tsym/(4*beta):
            h_rrc[i] = (beta/np.sqrt(2))*(((1+2/np.pi) * (np.sin(np.pi/(4*beta)))) +
                                          ((1-2/np.pi)*(np.cos(np.pi/(4*beta)))))
        elif beta != 0 and t == -Tsym/(4*beta):
            h_rrc[i] = (beta/np.sqrt(2))*(((1+2/np.pi) * (np.sin(np.pi/(4*beta)))) +
                                          ((1-2/np.pi)*(np.cos(np.pi/(4*beta)))))
        else:
            h_rrc[i] = (np.sin(np.pi*t*(1-beta)/Tsym) + 4*beta*(t/Tsym) *
                        np.cos(np.pi*t*(1+beta)/Tsym)) / (np.pi*t*(1-(4*beta*t/Tsym)**2)/Tsym)
    return h_rrc


def normalize(signal_faded):
    '''
    Normalize the signal
    Input  :  signal_faded : signal to be normalized
    Output :  signal_faded : signal that is normalized
    '''
    with np.errstate(all='raise'):
        try:
            signal_faded = signal_faded / rms(signal_faded)  # normalize signal
        except FloatingPointError:
            # print('data_file = '+str(data_dict['data_file'][i]) + ',\t reference_number = '+str(data_dict['reference_number'][i]))
            try:
                # print('Normalization error. RMS = {}, Max = {}, Min = {}, Data size = {}'.format(rms(signal), np.abs(signal).min(), np.abs(signal).max(), signal.shape))
                signal_faded += 1.0/np.sqrt(2*signal_faded.size) + 1.0 / \
                    np.sqrt(2*signal_faded.size)*1j
            except FloatingPointError:
                # print('i = {}, signal.shape = {}'.format(i, signal.shape))
                # print('start_index = {}, end_index = {}'.format(start_index, end_index))
                signal_size = signal_faded.size
                signal_faded = np.ones([signal_size]) * (1.0 + 1.0*1j)/np.sqrt(2*signal_size)
    return signal_faded
