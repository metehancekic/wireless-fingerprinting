from preproc.fading_model import normalize, add_custom_fading_channel, add_freq_offset
from preproc.preproc_wifi import basic_equalize_preamble, offset_compensate_preamble, get_residuals_preamble
import matplotlib.pyplot as plt
'''
All Simulation Codes needed for CFO and Channel Experiments

Need to add comment to the functions
'''

import numpy as np
from timeit import default_timer as timer
import argparse
from tqdm import trange, tqdm
import json
import os
import matplotlib as mpl



def real_to_complex(signals):
    return signals[:, :, 0] + 1j*signals[:, :, 1]


class WifiConfoundingFactors:
    def __init__(self, name, signals, devices, seeds, parameters):
        """
        signals:   wifi signals raw IQ form  [batch, num_samples, 2(I,Q)]
        devices:   devices of wifi signals   [batch, num_devices]
        seeds:     seeds to generate confounding factors
        """
        self.name = name
        self.signals = signals
        self.devices = devices
        self.seeds = seeds
        self.num_signals = self.signals.shape[0]
        self.num_devices = self.devices.shape[1]
        self.parameters = parameters

    def add_channel(self, progress_bar=True):

        if progress_bar:
            device_iters = tqdm(
                iterable=range(self.num_devices),
                desc=self.name + " Devices",
                unit="device",
                leave=False)
        else:
            device_iters = range(self.num_devices)

        for n in device_iters:
            signals_temp = self.signals.copy()
            ind_n = np.where(self.devices.argmax(axis=1) == n)[0]
            num_signals_per_channel = len(ind_n)//len(self.seeds)  # per class

            for idx_seed, main_seed in enumerate(self.seeds):
                seed_n = main_seed + n

                for i in ind_n[idx_seed * num_signals_per_channel: (idx_seed+1) * num_signals_per_channel]:
                    signal = real_to_complex(self.signals[i])
                    signal_faded = add_custom_fading_channel(signal, 500, self.parameters["sampling_rate"],
                                                             seed=seed_n,
                                                             beta=0,
                                                             delay_seed=False,
                                                             channel_type=self.parameters["channel_type"],
                                                             channel_method=self.parameters["channel_method"],
                                                             noise_method=self.parameters["noise_method"])
                    signal_faded = normalize(signal_faded)
                    signals_temp[i] = np.concatenate((signal_faded.real.reshape(
                        [-1, 1]), signal_faded.imag.reshape([-1, 1])), axis=1).reshape((1, -1, 2))

        self.signals = signals_temp.copy()

        # data_format = data_format + \
        # 	'[-phy-{}-m-{}-s-{}]-'.format(channel_type_phy_train, phy_method, np.max(seed_phy_train))

    def add_cfo(self, progress_bar=True):

        signals_temp = self.signals.copy()

        if progress_bar:
            device_iters = tqdm(
                iterable=range(self.num_devices),
                desc=self.name + " Devices",
                unit="device",
                leave=False)
        else:
            device_iters = range(self.num_devices)

        for n in device_iters:
            ind_n = np.where(self.devices.argmax(axis=1) == n)[0]

            num_signals_per_cfo = len(ind_n)//len(self.seeds)  # per class

            for idx_seed, main_seed in enumerate(self.seeds):
                seed_n = main_seed + n

                for i in ind_n[idx_seed*num_signals_per_cfo: (idx_seed+1)*num_signals_per_cfo]:
                    rv_n = np.random.RandomState(seed=seed_n)
                    signals_temp[i:i+1] = add_freq_offset(signals_temp[i:i+1], rand=False,
                                                          df=rv_n.uniform(
                        low=-self.parameters["df"], high=self.parameters["df"]),
                        fc=self.parameters["fc"][i:i+1],
                        fs=self.parameters["sampling_rate"])

        self.signals = signals_temp.copy()
        # data_format = data_format + \
        # 	'[_cfo_{}-s-{}]-'.format(np.int(df_phy_train*1000000), np.max(seed_phy_train_cfo))

    def compansate_cfo(self, progress_bar=True):

        complex_signals = real_to_complex(self.signals)

        if progress_bar:
            signals_iters = tqdm(
                iterable=range(self.num_signals),
                desc=self.name + " Signals",
                unit="signal",
                leave=False)
        else:
            signals_iters = range(self.num_signals)

        for i in signals_iters:
            complex_signals[i], _ = offset_compensate_preamble(
                complex_signals[i], fs=self.parameters["sampling_rate"], verbose=False, option=2)

        self.signals = np.concatenate(
            (complex_signals.real[..., None], complex_signals.imag[..., None]), axis=-1)

        # data_format = data_format + '[_comp]-'

    def equalize_channel(self, verbosity=False, progress_bar=True):

        complex_signals = real_to_complex(self.signals)

        if progress_bar:
            signals_iters = tqdm(
                iterable=range(self.num_signals),
                desc=self.name + " Signals",
                unit="signal",
                leave=False)
        else:
            signals_iters = range(self.num_signals)

        for i in signals_iters:
            complex_signals[i] = basic_equalize_preamble(complex_signals[i],
                                                         fs=self.parameters["sampling_rate"],
                                                         verbose=verbosity)
        self.signals = np.concatenate(
            (complex_signals.real[..., None], complex_signals.imag[..., None]), axis=2)

    def get_residual(self, verbosity=False, progress_bar=True):
        complex_signals = real_to_complex(self.signals)

        if progress_bar:
            signals_iters = tqdm(
                iterable=range(self.num_signals),
                desc=self.name + " Signals",
                unit="signal",
                leave=False)
        else:
            signals_iters = range(self.num_signals)

        for i in signals_iters:
            complex_signals[i] = get_residuals_preamble(complex_signals[i],
                                                        fs=self.parameters["sampling_rate"],
                                                        verbose=verbosity)
        self.signals = np.concatenate(
            (complex_signals.real[..., None], complex_signals.imag[..., None]), axis=2)


class WifiAugmenter:
    def __init__(self, name, signals, devices, seeds):
        """
        signals:   wifi signals raw IQ form  [batch, num_samples, 2(I,Q)]
        devices:   devices of wifi signals   [batch, num_devices]
        seeds:     seeds to generate confounding factors
        """
        self.name = name
        self.signals = signals
        self.devices = devices
        self.seeds = seeds
        self.num_signals = self.signals.shape[0]
        self.num_devices = self.devices.shape[1]

    def channel_augmentation_on_run(self, parameters):

        signals = real_to_complex(self.signals)
        for i, signal in enumerate(signals):
            signal_faded = add_custom_fading_channel(signal, parameters["snr"], parameters["sampling_rate"],
                                                     seed=None,
                                                     beta=0,
                                                     delay_seed=False,
                                                     channel_type=parameters["channel_type"],
                                                     channel_method=parameters["channel_method"],
                                                     noise_method=parameters["noise_method"])
            signals[i] = signal_faded
            self.signals[i] = np.concatenate((signal_faded.real.reshape(
                [-1, 1]), signal_faded.imag.reshape([-1, 1])), axis=1).reshape((1, -1, 2))

    def channel_augmentation(self, parameters, progress_bar=True):

    	num_augmentation = parameters["num_augmentation"]
    	snr = parameters["snr"]
    	sampling_rate = parameters["sampling_rate"]

        signals = real_to_complex(self.signals)
        for i, signal in enumerate(signals):
            signal_faded = add_custom_fading_channel(signal, parameters["snr"], parameters["sampling_rate"],
                                                     seed=None,
                                                     beta=0,
                                                     delay_seed=False,
                                                     channel_type=parameters["channel_type"],
                                                     channel_method=parameters["channel_method"],
                                                     noise_method=parameters["noise_method"])
            signals[i] = signal_faded
            self.signals[i] = np.concatenate((signal_faded.real.reshape(
                [-1, 1]), signal_faded.imag.reshape([-1, 1])), axis=1).reshape((1, -1, 2))

        orthogonal_seeds = {}
	    for i in range(401):
	        orthogonal_seeds[i] = self.seeds

	    if progress_bar:
            augmentations_iters = tqdm(
                iterable=range(num_augmentation),
                desc=self.name + " Augmentations",
                unit="augmentation",
                leave=False)
        else:
            augmentations_iters = range(num_augmentation)

        for k in augmentations_iters:
            signals_augmented = np.zeros(self.signals.shape)
            if progress_bar:
	            signals_iters = tqdm(
	                iterable=range(self.num_signals),
	                desc=self.name + " Signals",
	                unit="signal",
	                leave=False)
	        else:
	            signals_iters = range(self.num_signals)

            for i in signals_iters:

                complex_signals = real_to_complex(self.signals)
                if parameters["augmentation_type"]=="full_random":
                    signal_faded = add_custom_fading_channel(complex_signals, parameters["snr"], parameters["sampling_rate"],
                                                     seed=self.seeds +
                                                             (i + k * self.num_signals) % (self.num_signals*num_augmentation),
                                                     beta=0,
                                                     delay_seed=False,
                                                     channel_type=parameters["channel_type"],
                                                     channel_method=parameters["channel_method"],
                                                     noise_method=parameters["noise_method"])
                elif parameters["augmentation_type"]=="orthogonal":
                    signal_faded = add_custom_fading_channel(complex_signals, parameters["snr"], parameters["sampling_rate"],
                                                     seed=orthogonal_seeds[np.argmax(
                                                                 y_train[i])],
                                                     beta=0,
                                                     delay_seed=False,
                                                     channel_type=parameters["channel_type"],
                                                     channel_method=parameters["channel_method"],
                                                     noise_method=parameters["noise_method"])
                    orthogonal_seeds[np.argmax(y_train[i])] += 1
                else:
                	raise NotImplementedError
                signal_faded = normalize(signal_faded)
                signals_augmented[i] = np.concatenate((signal_faded.real.reshape(
                    [-1, 1]), signal_faded.imag.reshape([-1, 1])), axis=1).reshape((1, -1, 2))

            if parameters["keep_original"] is False:
                if k == 0:
                    signals_aug = signals_augmented.copy()
                    devices_aug = y_train.copy()
                else:
                    signals_aug = np.concatenate((signals_aug, signals_augmented), axis=0)
                    devices_aug = np.concatenate((devices_aug, self.devices), axis=0)
            else:
                signals_aug = np.concatenate((signals_aug, signals_augmented), axis=0)
                devices_aug = np.concatenate((devices_aug, self.devices), axis=0)

        dict_wifi['x_train'] = signals_aug.copy()
	    dict_wifi['y_train'] = devices_aug.copy()
	    dict_wifi['fc_train'] = np.tile(dict_wifi['fc_train'], num_aug_train)

        return 


def augment_with_channel(dict_wifi, aug_type, channel_method, num_aug_train, num_aug_test, keep_orig_train, keep_orig_test, num_ch_train, num_ch_test, channel_type_aug_train, channel_type_aug_test, delay_seed_aug_train, snr_train, noise_method, seed_aug, sampling_rate, data_format):

    x_train = dict_wifi['x_train'].copy()
    y_train = dict_wifi['y_train'].copy()

    x_test = dict_wifi['x_test'].copy()
    y_test = dict_wifi['y_test'].copy()

    num_train = dict_wifi['x_train'].shape[0]
    num_test = dict_wifi['x_test'].shape[0]
    num_classes = dict_wifi['y_train'].shape[1]

    # print('\n-------------------------------')

    print('\nChannel augmentation')
    print('\tAugmentation type: {}'.format(aug_type))
    print('\tNo of augmentations: Train: {}, Test: {}\n\tKeep originals: Train: {}, Test: {}'.format(
        num_aug_train, num_aug_test, keep_orig_train, keep_orig_test))
    print('\tNo. of channels per aug: Train: {}, Test: {}'.format(num_ch_train, num_ch_test))
    print('\tChannel type: Train: {}, Test: {}\n'.format(
        channel_type_aug_train, channel_type_aug_test))

    print("Seed: Train: {:}".format(seed_aug))

    x_train_aug = x_train.copy()
    y_train_aug = y_train.copy()

    channel_dict = {}
    for i in range(401):
        channel_dict[i] = seed_aug

    if num_ch_train < -1:
        raise ValueError('num_ch_train')
    elif num_ch_train != 0:
        for k in tqdm(range(num_aug_train)):
            signal_ch = np.zeros(x_train.shape)
            for i in tqdm(range(num_train)):
                signal = x_train[i][:, 0]+1j*x_train[i][:, 1]
                if num_ch_train == -1:
                    signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate,
                                                             seed=seed_aug +
                                                             (i + k*num_train) % (num_train*num_aug_train),
                                                             beta=0,
                                                             delay_seed=delay_seed_aug_train,
                                                             channel_type=channel_type_aug_train,
                                                             channel_method=channel_method,
                                                             noise_method=noise_method)
                elif aug_type == 1:
                    signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate,
                                                             seed=channel_dict[np.argmax(
                                                                 y_train[i])],
                                                             beta=0,
                                                             delay_seed=delay_seed_aug_train,
                                                             channel_type=channel_type_aug_train,
                                                             channel_method=channel_method,
                                                             noise_method=noise_method)
                    channel_dict[np.argmax(y_train[i])] += 1
                elif aug_type == 0:
                    signal_faded = add_custom_fading_channel(signal, snr_train, sampling_rate,
                                                             # seed = 0,
                                                             seed=seed_aug + k * num_ch_train + \
                                                             (i %
                                                              num_ch_train),
                                                             beta=0,
                                                             delay_seed=delay_seed_aug_train,
                                                             channel_type=channel_type_aug_train,
                                                             channel_method=channel_method,
                                                             noise_method=noise_method)

                signal_faded = normalize(signal_faded)
                signal_ch[i] = np.concatenate((signal_faded.real.reshape(
                    [-1, 1]), signal_faded.imag.reshape([-1, 1])), axis=1).reshape((1, -1, 2))

            if keep_orig_train is False:
                if k == 0:
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
    dict_wifi['fc_train'] = np.tile(dict_wifi['fc_train'], num_aug_train)

    del x_train, y_train, x_train_aug, y_train_aug

    data_format = data_format + '[aug-{}-art-{}-ty-{}-nch-{}-snr-{:.0f}]-'.format(
        num_aug_train, channel_type_aug_train, aug_type, num_ch_train, snr_train)

    return dict_wifi, data_format


# def augment_with_channel_test(dict_wifi, aug_type, channel_method, num_aug_train, num_aug_test, keep_orig_train, keep_orig_test, num_ch_train, num_ch_test, channel_type_aug_train, channel_type_aug_test, delay_seed_aug_test, snr_test, noise_method, seed_aug, sampling_rate, data_format):

#     x_test = dict_wifi['x_test'].copy()
#     y_test = dict_wifi['y_test'].copy()

#     num_train = dict_wifi['x_train'].shape[0]
#     num_test = dict_wifi['x_test'].shape[0]
#     num_classes = dict_wifi['y_train'].shape[1]

#     x_test_aug = x_test.copy()
#     y_test_aug = y_test.copy()

#     if num_ch_test < -1:
#         raise ValueError('num_ch_test')
#     elif num_ch_test != 0:
#         for k in tqdm(range(num_aug_test)):
#             signal_ch = np.zeros(x_test.shape)
#             for i in tqdm(range(num_test)):
#                 signal = dict_wifi['x_test'][i][:, 0]+1j*dict_wifi['x_test'][i][:, 1]
#                 if num_ch_test == -1:
#                     signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate,
#                                                              seed=seed_aug + num_train*num_aug_train + 1 +
#                                                              (i + k*num_test) % (num_test*num_aug_test),
#                                                              beta=0,
#                                                              delay_seed=delay_seed_aug_test,
#                                                              channel_type=channel_type_aug_test,
#                                                              channel_method=channel_method,
#                                                              noise_method=noise_method)
#                 else:
#                     signal_faded = add_custom_fading_channel(signal, snr_test, sampling_rate,
#                                                              # seed = 1,
#                                                              seed=seed_aug + num_train*num_aug_train + \
#                                                              1 + (i % num_ch_test) + \
#                                                              k * num_ch_test,
#                                                              beta=0,
#                                                              delay_seed=delay_seed_aug_test,
#                                                              channel_type=channel_type_aug_test,
#                                                              channel_method=channel_method,
#                                                              noise_method=noise_method)

#                 signal_faded = normalize(signal_faded)
#                 signal_ch[i] = np.concatenate((signal_faded.real.reshape(
#                     [-1, 1]), signal_faded.imag.reshape([-1, 1])), axis=1)
#                 # dict_wifi['x_test'][i] = signal_ch
#             if keep_orig_test is False:
#                 if k == 0:
#                     x_test_aug = signal_ch.copy()
#                     y_test_aug = y_test.copy()
#                 else:
#                     x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
#                     y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)
#             else:
#                 x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
#                 y_test_aug = np.concatenate((y_test_aug, y_test), axis=0)

#     dict_wifi['x_test'] = x_test_aug.copy()
#     dict_wifi['y_test'] = y_test_aug.copy()
#     dict_wifi['fc_test'] = np.tile(dict_wifi['fc_test'], num_aug_test)

#     del x_test_aug, y_test_aug

#     return dict_wifi, data_format


# def augment_with_cfo(dict_wifi, aug_type_cfo, df_aug_train, num_aug_train_cfo, keep_orig_train_cfo, rand_aug_train, sampling_rate, seed_aug_cfo, data_format):

#     print('\nCFO augmentation')
#     print('\tAugmentation type: {}'.format(aug_type_cfo))
#     print('\tNo of augmentations: Train: {}, \n\tKeep originals: Train: {}'.format(
#         num_aug_train_cfo, keep_orig_train_cfo))

#     x_train_aug = dict_wifi['x_train'].copy()
#     y_train_aug = dict_wifi['y_train'].copy()

#     fc_train_orig = dict_wifi['fc_train']
#     fc_test_orig = dict_wifi['fc_test']

#     num_train = dict_wifi['x_train'].shape[0]
#     num_test = dict_wifi['x_test'].shape[0]
#     num_classes = dict_wifi['y_train'].shape[1]

#     if aug_type_cfo == 0:
#         for k in tqdm(range(num_aug_train_cfo)):
#             signal_ch = dict_wifi['x_train'].copy()
#             # import ipdb; ipdb.set_trace()
#             signal_ch = add_freq_offset(signal_ch, rand=rand_aug_train,
#                                         df=df_aug_train,
#                                         fc=fc_train_orig,
#                                         fs=sampling_rate)
#             if keep_orig_train_cfo is False:
#                 if k == 0:
#                     x_train_aug = signal_ch.copy()
#                     y_train_aug = dict_wifi['y_train'].copy()
#                 else:
#                     x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
#                     y_train_aug = np.concatenate((y_train_aug, dict_wifi['y_train']), axis=0)
#             else:
#                 x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
#                 y_train_aug = np.concatenate((y_train_aug, dict_wifi['y_train']), axis=0)

#     elif aug_type_cfo == 1:
#         offset_dict = {}
#         for i in range(401):
#             offset_dict[i] = seed_aug_cfo
#         for k in tqdm(range(num_aug_train_cfo)):
#             signal_ch = dict_wifi['x_train'].copy()
#             for i in tqdm(range(num_train)):
#                 rv_n = np.random.RandomState(seed=offset_dict[np.argmax(dict_wifi['y_train'][i])])
#                 if rand_aug_train == 'unif':
#                     df_n = rv_n.uniform(low=-df_aug_train, high=df_aug_train)
#                 elif rand_aug_train == 'ber':
#                     df_n = rv_n.choice(a=np.array([-df_aug_train, df_aug_train]))
#                 signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand=False,
#                                                    df=df_n,
#                                                    fc=fc_train_orig[i:i+1],
#                                                    fs=sampling_rate)
#                 offset_dict[np.argmax(dict_wifi['y_train'][i])] += 1
#             if keep_orig_train_cfo is False:
#                 if k == 0:
#                     x_train_aug = signal_ch.copy()
#                     y_train_aug = dict_wifi['y_train'].copy()
#                 else:
#                     x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
#                     y_train_aug = np.concatenate((y_train_aug, dict_wifi['y_train']), axis=0)
#             else:
#                 x_train_aug = np.concatenate((x_train_aug, signal_ch), axis=0)
#                 y_train_aug = np.concatenate((y_train_aug, dict_wifi['y_train']), axis=0)

#     dict_wifi['x_train'] = x_train_aug.copy()
#     dict_wifi['y_train'] = y_train_aug.copy()

#     del x_train_aug, y_train_aug, fc_train_orig, fc_test_orig

#     data_format = data_format + '[augcfo-{}-df-{}-rand-{}-ty-{}-{}-t-]-'.format(
#         num_aug_train_cfo, df_aug_train*1e6, rand_aug_train, aug_type_cfo, keep_orig_train_cfo)

#     return dict_wifi, data_format


# def augment_with_cfo_test(dict_wifi, aug_type_cfo, df_aug_test, num_aug_test_cfo, keep_orig_test_cfo, rand_aug_test, sampling_rate):

#     print('\nCFO augmentation')
#     print('\tAugmentation type: {}'.format(aug_type_cfo))
#     print('\tNo of augmentations: Test: {}, \n\tKeep originals: Test: {}'.format(
#         num_aug_test_cfo, keep_orig_test_cfo))

#     print('\tCFO aug type: {}\n'.format(aug_type_cfo))

#     x_test_aug = dict_wifi['x_test'].copy()
#     y_test_aug = dict_wifi['y_test'].copy()

#     fc_test_orig = dict_wifi['fc_test']

#     # if aug_type_cfo == 0:
#     for k in tqdm(range(num_aug_test_cfo)):
#         signal_ch = dict_wifi['x_test'].copy()
#         # import ipdb; ipdb.set_trace()
#         signal_ch = add_freq_offset(signal_ch, rand=False,
#                                     df=np.random.uniform(-df_aug_test, df_aug_test),
#                                     fc=fc_test_orig,
#                                     fs=sampling_rate)
#         if keep_orig_test_cfo is False:
#             if k == 0:
#                 x_test_aug = signal_ch.copy()
#                 y_test_aug = dict_wifi['y_test'].copy()
#             else:
#                 x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
#                 y_test_aug = np.concatenate((y_test_aug, dict_wifi['y_test']), axis=0)
#         else:
#             x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
#             y_test_aug = np.concatenate((y_test_aug, dict_wifi['y_test']), axis=0)

#     # elif aug_type_cfo == 1:
#     #   offset_dict = {}
#     #   for i in range(401):
#     #       offset_dict[i] = seed_phy_test_cfo+seed_phy_test_cfo+num_classes+1
#     #   for k in tqdm(range(num_aug_test_cfo)):
#     #       signal_ch = dict_wifi['x_test'].copy()
#     #       for i in tqdm(range(num_test)):
#     #           rv_n = np.random.RandomState(seed=offset_dict[np.argmax(dict_wifi['y_test'][i])])
#     #           if rand_aug_test=='unif':
#     #               df_n = rv_n.uniform(low=-df_aug_test, high=df_aug_test)
#     #           elif rand_aug_test=='ber':
#     #               df_n = rv_n.choice(a=np.array([-df_aug_test, df_aug_test]))
#     #           signal_ch[i:i+1] = add_freq_offset(signal_ch[i:i+1], rand = False,
#     #                                                                df = df_n,
#     #                                                                fc = fc_test_orig[i:i+1],
#     #                                                                fs = fs)
#     #           offset_dict[np.argmax(dict_wifi['y_test'][i])] += 1
#     #       if keep_orig_test_cfo is False:
#     #           if k==0:
#     #               x_test_aug = signal_ch.copy()
#     #               y_test_aug = dict_wifi['y_test'].copy()
#     #           else:
#     #               x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
#     #               y_test_aug = np.concatenate((y_test_aug, dict_wifi['y_test']), axis=0)
#     #       else:
#     #           x_test_aug = np.concatenate((x_test_aug, signal_ch), axis=0)
#     #           y_test_aug = np.concatenate((y_test_aug, dict_wifi['y_test']), axis=0)

#     dict_wifi['x_test'] = x_test_aug.copy()
#     dict_wifi['y_test'] = y_test_aug.copy()

#     del x_test_aug, y_test_aug, fc_test_orig

#     return dict_wifi
