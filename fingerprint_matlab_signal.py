'''
Trains data for a WiFi experiment.

Data is read from npz files.
'''

import numpy as np
import numpy.random as random
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from scipy import io
import argparse
import ipdb
import os
from tqdm import trange

from keras import optimizers

# from .cxnn.train_network  import train
# from .cxnn.train_network _aug import train
from .cxnn.train_llr  import train_200 as train
from .preproc.fading_model  import normalize


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--mat_file', type=str, default='UniThirdOrderSatPreambles1_20.mat', help='mat file') 
parser.add_argument('-s', '--snr', type=int, default=20, help='snr in db') 
parser.add_argument('-ntr', '--n_train_per', type=int, default=200, help='no of training samples per device')
parser.add_argument('-nte', '--n_test_per', type=int, default=100, help='no of test samples per device')
parser.add_argument('-se', '--seed', type=int, default=0, help='seed')
parser.add_argument('-r', '--residual', action='store_true', help='whether to use residual preprocessing')
args = parser.parse_args()

exp_dir = '/home/rfml/wifi/mat_files'

data_format = 'f_{:}_snr_{:}_ntr_{:}_nte_{:}_s_{:}'.format(args.mat_file, args.snr, args.n_train_per, args.n_test_per, args.seed)

all_data = io.loadmat(os.path.join(exp_dir, args.mat_file))
data = all_data['PreamblePerDevice'].dot(np.array([1, 1j]))

n_train_per = args.n_train_per
n_test_per = args.n_test_per
n_classes = 19
n_steps = 3200
x_train = np.zeros((n_train_per*n_classes, n_steps, 2))
x_test = np.zeros((n_test_per*n_classes, n_steps, 2))
y_train = np.zeros((n_train_per*n_classes, n_classes))
y_test = np.zeros((n_test_per*n_classes, n_classes))

rv_noise = random.RandomState(seed=args.seed)
eye = np.eye(n_classes)

for i in trange(n_train_per):
    for j in range(n_classes):
        signal = data[j].copy()

        if args.snr < 500:

            E_b = (np.abs(signal)**2).mean()
            N_0 = E_b/(10**(args.snr/10))
            N = len(signal)
            n = rv_noise.normal(loc=0, scale=np.sqrt(N_0/2), size=(N, 2)).dot(np.array([1, 1j]))
            signal += n

        x_train[i*n_classes + j, :, 0] = signal.real
        x_train[i*n_classes + j, :, 1] = signal.imag
        y_train[i*n_classes + j, :] = eye[j]

for i in trange(n_test_per):
    for j in range(n_classes):
        signal = data[j].copy()

        if args.snr < 500:
            E_b = (np.abs(signal)**2).mean()
            N_0 = E_b/(10**(args.snr/10))
            N = len(signal)
            n = rv_noise.normal(loc=0, scale=np.sqrt(N_0/2), size=(N, 2)).dot(np.array([1, 1j]))
            signal += n

        x_test[i*n_classes + j, :, 0] = signal.real
        x_test[i*n_classes + j, :, 1] = signal.imag
        y_test[i*n_classes + j, :] = eye[j]

if args.residual is True:
    data_format = 'residual_' + data_format
    preamble = all_data['Preamble']

    # plt.figure()
    # plt.plot(x_train[0, :, 0])
    # plt.title('Orig')

    x_train = x_train - preamble
    x_test = x_test - preamble

    # plt.figure()
    # plt.plot(x_train[0, :, 0])
    # plt.title('After residual preproc')
    # plt.show()

for i in range(n_train_per*n_classes):
    signal = x_train[i, :, 0] + 1j* x_train[i, :, 1]
    signal = normalize(signal)
    x_train[i, :, 0] = signal.real
    x_train[i, :, 1] = signal.imag

for i in range(n_test_per*n_classes):
    signal = x_test[i, :, 0] + 1j* x_test[i, :, 1]
    signal = normalize(signal)
    x_test[i, :, 0] = signal.real
    x_test[i, :, 1] = signal.imag

dict_wifi = {}
dict_wifi['x_train'] = x_train.copy()
dict_wifi['x_test'] = x_test.copy()
dict_wifi['y_train'] = y_train.copy()
dict_wifi['y_test'] = y_test.copy()
dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]

# import ipdb; ipdb.set_trace()

# Checkpoint path
checkpoint = exp_dir + '/ckpt-' + data_format + '.h5'

# optimizer = optimizers.SGD(lr=0.001, momentum=0.5, decay=0.0, nesterov=True)
optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
epochs = 5000

print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
# train_output, model_name, summary, conf_matrix_test = train(dict_wifi, checkpoint_in=None, checkpoint_out=checkpoint)
train_output, model_name, summary = train(dict_wifi, checkpoint_in=None, 
                                                     checkpoint_out=checkpoint, 
                                                     batch_size=100,
                                                     epochs=epochs,
                                                     num_aug_test=0,
                                                     optimizer=optimizer,
                                                     val='test')
print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

# Write logs
with open(exp_dir + '/logs-' + data_format  + '.txt', 'a+') as f:
    f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')
    for keys, dicts in train_output.items():
        f.write(str(keys)+':\n')
        for key, value in dicts.items():
            f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
    f.write('\n'+str(summary))

