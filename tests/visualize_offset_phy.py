from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from IPython import get_ipython
get_ipython().magic('matplotlib')

import os
import numpy as np
from collections import OrderedDict as odict
from timeit import default_timer as timer
import ipdb
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import trange

import keras
from keras import backend as K
from keras import optimizers, regularizers, losses
from keras.layers import Dense, Input, Activation, Conv1D, Dropout, GlobalAveragePooling1D
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.engine.topology import Layer


from cxnn.complexnn import ComplexDense, ComplexConv1D, utils, Modrelu

import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble = r'\usepackage{amsmath}, \usepackage{sfmath}, \usepackage{amssymb}, \usepackage{bm}, \DeclareMathOperator*{\supp}{supp}, \DeclareMathOperator*{\proj}{\mathcal{P}_K}, \newcommand{\be}{{\bm e}}, \newcommand{\bx}{{\bm x}}, \DeclareMathOperator*{\support}{\mathcal{S}_K}')

np.set_printoptions(precision=3)

#-------------------------------------------------
# Model name
#-------------------------------------------------

# Regular model
# model_name = '-modrelu-100-100-new-'
model_name = '-100C200x100-ModReLU-100C10x1-ModReLU-Abs-100shared_D-100shared_D-Avg'

# Early abs
# model_name = '-100C200x100-ModReLU-Abs-100shared_C10x1-100shared_D-100shared_D-Avg'

# Short stride
# model_name = '-100C200x10-ModReLU-100C10x1-ModReLU-Abs-100shared_D-100shared_D-Avg'

# Short conv
# model_name = '-100C40x10-ModReLU-100C10x1-ModReLU-Abs-100shared_D-100shared_D-Avg'

# Short conv, but no modrelu
# model_name = '-100C40x10-ModReLU-100C10x1-Abs-100shared_D-100shared_D-Avg'

# Short conv and early abs
# model_name = '-100C40x10-ModReLU-Abs-100shared_C10x1-100shared_D-100shared_D-Avg'

# Short conv and early abs, but no modrelu
# model_name = '-100C40x10-Abs-100shared_C10x1-100shared_D-100shared_D-Avg'


exp_dir = os.environ['path_to_data']


preprocess_type = 1
sample_rate = 200
sample_duration = 16

#-------------------------------
# Physical offset params
#-------------------------------
df_phy_train = 40e-6
df_phy_test = 40e-6

# seed_phy_pairs = [(0, 20), (40, 60), (80, 100), (120, 140), (160, 180)]
seed_phy_pairs = [(0, 20)]

seed_phy_train, seed_phy_test = seed_phy_pairs[0]

#-------------------------------
# Augmentation offset params
#-------------------------------
df_aug_train = df_phy_train 
rand_aug_train = 'unif'
# rand_aug_train = 'ber'
# rand_aug_train = 'False'

df_aug_test = df_aug_train 
rand_aug_test = 'unif'
# rand_aug_test = 'ber'
# rand_aug_test = 'False'

# num_aug_train = 0
# num_aug_train = 5
num_aug_train = 20

num_aug_test = 0
# num_aug_test = 5
# num_aug_test = 20
# num_aug_test = 100

keep_orig_train = False
# keep_orig_train = True

keep_orig_test = False
# keep_orig_test = True
'''
aug_type:
    0 - usual offset aug
    1 - same offset for ith example in each class
'''
# aug_type = 0
aug_type = 1

data_format = '{:.0f}-pp-{:}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

start = timer()

npz_filename = exp_dir + '/sym-' + data_format + '.npz'
np_dict = np.load(npz_filename)
dict_wifi = {}
dict_wifi['x_train'] = np_dict['arr_0']
dict_wifi['y_train'] = np_dict['arr_1']
dict_wifi['x_test'] = np_dict['arr_2']
dict_wifi['y_test'] = np_dict['arr_3']
dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]

end = timer()
print('Load time: {:} sec'.format(end - start))


print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

x_train = dict_wifi['x_train']
y_train = dict_wifi['y_train']
x_test = dict_wifi['x_test']
y_test = dict_wifi['y_test']
num_classes = dict_wifi['num_classes']
num_train = x_train.shape[0]
num_test = x_test.shape[0]
num_features = x_train.shape[1]

batch_size = 100    
epochs = 100
weight_decay = 1e-3

data_format = 'offset-phy-{}-s-{}-aug-{}-df-{}-rand-{}-ty-{}-{}-t-'.format(df_phy_train*1e6, seed_phy_train, num_aug_train, df_aug_train*1e6, rand_aug_train, aug_type, keep_orig_train)
data_format += '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

# checkpoint_in = exp_dir + '/ckpt-' + data_format +'.h5'
# checkpoint_in += '-modrelu-100-100'+'-new-'
# checkpoint_in += '-new.h5'

checkpoint_in = exp_dir + '/ckpt-' + data_format +'.h5'
checkpoint_in +=  model_name
checkpoint_in += '-new.h5'

print('\n-------------------------------')
print("Loading model from checkpoint")
print('Model name: {}'.format(model_name[1:]))
batch_size = 100

print("========================================") 
print("== BUILDING MODEL... ==")

if checkpoint_in is None:
    raise ValueError('Cannot test without a checkpoint')

# checkpoint_in += '-new.h5'
densenet = load_model(checkpoint_in, 
                      custom_objects={'ComplexConv1D':complexnn.ComplexConv1D,
                                      'GetAbs': utils.GetAbs,
                                      'Modrelu': Modrelu})

probs = densenet.predict(x=x_test, batch_size=batch_size, verbose=0)
label_pred = probs.argmax(axis=1) 
label_act = y_test.argmax(axis=1) 
ind_correct = np.where(label_pred==label_act)[0] 
ind_wrong = np.where(label_pred!=label_act)[0] 
assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
test_acc = 100.*ind_correct.size / num_test

print("\n========================================") 
print('Test accuracy: {:.2f}%'.format(test_acc))

densenet.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         (None, 3200, 2)           0         
# _________________________________________________________________
# ComplexConv1 (ComplexConv1D) (None, 31, 200)           40200     
# _________________________________________________________________
# ModRelu1 (Modrelu)           (None, 31, 200)           100       
# _________________________________________________________________
# ComplexConv2 (ComplexConv1D) (None, 22, 200)           200200    
# _________________________________________________________________
# ModRelu2 (Modrelu)           (None, 22, 200)           100       
# _________________________________________________________________
# Abs (GetAbs)                 (None, 22, 100)           0         
# _________________________________________________________________
# Shared_Dense1 (Dense)        (None, 22, 100)           10100     
# _________________________________________________________________
# Shared_Dense2 (Dense)        (None, 22, 100)           10100     
# _________________________________________________________________
# global_average_pooling1d_1 ( (None, 100)               0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 19)                1919      
# =================================================================
# Total params: 262,719
# Trainable params: 262,719
# Non-trainable params: 0

# ipdb.set_trace()

newInput = Input(batch_shape=(1, 3200, 2))
newOutputs = densenet(newInput)
model = Model(newInput, newOutputs)
input_signal = model.inputs[0]

print(input_signal._keras_shape) # (1, 320, 2)
print(model.output._keras_shape) # (1, 15, 100) or (1, 11, 100)

# filter_index = 0  # can be any integer from 0 to 199, as there are 200 filters in that layer
filter_indices = np.arange(0, 4)

epochs = 500
# step = 1
step = 0.01
momentum = 0.1

decay = 0.01

# input_data_init = np.random.random((1, 3200)) + 1j*np.random.random((1, 3200))
input_data_init = np.random.uniform(low=-1.0, high=1.0, size=(1, 3200)) + 1j*np.random.uniform(low=-1.0, high=1.0, size=(1, 3200))
# input_data_init = np.ones((1, 320)) + 1j*np.ones((1,320))

input_data_init /= np.sqrt(np.mean(input_data_init * np.conjugate(input_data_init)))
input_data_init = np.concatenate((input_data_init.real[..., None], input_data_init.imag[..., None]), axis=2)

num_filters = filter_indices.size
plt.figure(figsize=(10, 6))

for index in range(num_filters):
    filter_index = filter_indices[index]
    print('\n\nFilter {}'.format(filter_index))
    time_length = model.output.shape[1]


    loss = K.mean(model.output[:, filter_index])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_signal)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_signal], [loss, grads])

    # we start from a gray image with some noise
    # input_data = np.ones((1, 320)) + 1j*np.ones((1,320))
    input_data = input_data_init
    velocity = 0

    # run gradient ascent for 100 steps
    for i in range(epochs):
        loss_value, grads_value = iterate([input_data])
        velocity = momentum*velocity + step*grads_value
        input_data = input_data + velocity

        # input_data += grads_value * step
        input_complex = input_data[0:1, :, 0] + 1j* input_data[0:1, :, 1]
        input_complex /= np.sqrt(np.mean(input_complex * np.conjugate(input_complex)))
        input_data = np.concatenate((input_complex.real[..., None], input_complex.imag[..., None]), axis=2)
        if i%20 ==0:
            print('Step {}, Loss {}'.format(i, loss_value))

    input_complex = input_data[0, :, 0] + 1j* input_data[0, :, 1]
    # input_complex /= np.sqrt(np.mean(input_complex * np.conjugate(input_complex)))

    # input_complex = input_complex[:100]
    # input_complex = input_complex[:120]
    symbols = np.arange(input_complex.size)*1.0

    plt.subplot(num_filters, 2, index*2 + 1)
    # plt.stem(np.abs(input_complex))
    plt.plot(symbols, np.abs(input_complex), '-')
    plt.title('Ground truth')
    # plt.ylabel('Magnitude')
    # plt.xlabel('Symbols')
    # plt.title('Filter {}, Magnitude'.format(filter_index+1))
    plt.title('Filter {}, Magnitude'.format(index+1))
    plt.grid(True)
    plt.subplot(num_filters, 2, index*2 + 2)
    # plt.plot(np.unwrap(np.angle(input_complex)), '-')
    # plt.stem(np.angle(input_complex))
    plt.plot(symbols, np.angle(input_complex), '-')
    # plt.xlabel('Symbols')
    # plt.ylabel('Phase')
    plt.grid(True)
    # plt.title('Filter {}, Phase'.format(filter_index+1))
    plt.title('Filter {}, Phase'.format(index+1))
plt.subplot(num_filters, 2, num_filters*2-1)
plt.xlabel('Time in symbols')
plt.subplot(num_filters, 2, num_filters*2)
plt.xlabel('Time in symbols')

plt.suptitle('Output layer')
plt.tight_layout(rect=[0.01, 0.01, 0.98, 0.99], h_pad=0.2)
# plt.savefig('layer_2_c', format='pdf', dpi=1000, bbox_inches='tight')

