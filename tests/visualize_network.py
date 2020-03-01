from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from IPython import get_ipython
# get_ipython().magic('matplotlib')

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

# import complexnn
from ..cxnn.complexnn import ComplexDense, ComplexConv1D, utils
# from .cxnn.train_network _rotated_after_conv import set_keras_backend, net_address_20, net_preamble_20, net_preamble_20_no_crelu, net_preamble_20_real, RotateComplex, output_of_lambda, relu_manual, net_preamble_20_rotated, RotateComplex, relu_manual, Modrelu, net_preamble_20_modrelu, net_preamble_20_rot16_real, net_preamble_50, net_preamble_100
from ..cxnn.models_adsb import Modrelu, set_keras_backend

import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble = r'\usepackage{amsmath}, \usepackage{sfmath}, \usepackage{amssymb}, \usepackage{bm}, \DeclareMathOperator*{\supp}{supp}, \DeclareMathOperator*{\proj}{\mathcal{P}_K}, \newcommand{\be}{{\bm e}}, \newcommand{\bx}{{\bm x}}, \DeclareMathOperator*{\support}{\mathcal{S}_K}')

set_keras_backend("theano")

np.set_printoptions(precision=3)

exp_dir = 'dataset-directory'


preprocess_type = 1
# preprocess_type = 2
# preprocess_type = 3

# sample_rate = 200
# sample_rate = 100
sample_rate = 20

sample_duration = 16
# sample_duration = 64
# Set this to 16 to avoid plane ID !!!

data_format = '{:.0f}-pp-{:}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

start = timer()

npz_filename = exp_dir + '/sym-' + data_format + '.npz'
np_dict = np.load(npz_filename)
dict_adsb = {}
dict_adsb['x_train'] = np_dict['arr_0']
dict_adsb['y_train'] = np_dict['arr_1']
dict_adsb['x_test'] = np_dict['arr_2']
dict_adsb['y_test'] = np_dict['arr_3']
dict_adsb['num_classes'] = dict_adsb['y_test'].shape[1]

end = timer()
print('Load time: {:} sec'.format(end - start))

# Checkpoint path
# checkpoint_in = exp_dir + '/ckpt-Last' + data_format + '-new.h5'
checkpoint_in = exp_dir + '/ckpt-' + data_format + '.h5-modrelu-100-100-new.h5'
# ckpt-Last16-pp-1-fs-20-new.h5

print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

x_train = dict_adsb['x_train']
y_train = dict_adsb['y_train']
x_test = dict_adsb['x_test']
y_test = dict_adsb['y_test']
num_classes = dict_adsb['num_classes']
num_train = x_train.shape[0]
num_test = x_test.shape[0]
num_features = x_train.shape[1]

batch_size = 100    
epochs = 100
weight_decay = 1e-3

print("========================================") 
print("== BUILDING MODEL... ==")

if checkpoint_in is None:
	raise ValueError('Cannot test without a checkpoint')

# checkpoint_in += '-new.h5'
densenet = load_model(checkpoint_in, 
					  custom_objects={'ComplexConv1D': ComplexConv1D,
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
# input_1 (InputLayer)         (100, 320, 2)             0         
# _________________________________________________________________
# ComplexConv1 (ComplexConv1D) (100, 15, 400)            16400     
# _________________________________________________________________
# ModRelu (Modrelu)            (100, 15, 400)            200       
# _________________________________________________________________
# ComplexConv2 (ComplexConv1D) (100, 11, 256)            256256    
# _________________________________________________________________
# ModRelu2 (Modrelu)           (100, 11, 256)            128       
# _________________________________________________________________
# Abs (GetAbs)                 (100, 11, 128)            0         
# _________________________________________________________________
# GlobalAvg (GlobalAveragePool (100, 128)                0         
# _________________________________________________________________
# Dense1 (Dense)               (100, 100)                12900     
# _________________________________________________________________
# Dense2 (Dense)               (100, 100)                10100     
# _________________________________________________________________
# Softmax (Lambda)             (100, 100)                0         
# =================================================================

layer_name = 'modrelu_1'
# layer_name = 'ModRelu1'
# layer_name = 'ModRelu2'
# layer_name = 'Dense2'

# build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_dict = dict([(layer.name, layer) for layer in densenet.layers])
layer_output = layer_dict[layer_name].output
layer_output = utils.GetAbs(name="Abs")(layer_output)
print(layer_output.shape)

model2 = Model(inputs=densenet.input, 
			   outputs=layer_output)
model2.summary()

features_test = model2.predict(x=x_test, batch_size=batch_size, verbose=0) 
features_train = model2.predict(x=x_train, batch_size=batch_size, verbose=0)
# shape (batch_size, 15, 100) or (batch_size, 11, 100)

tol = 1e-2
tol2 = 90
# sparsity = (features_test > tol).mean(axis=0).max(axis=0)*100
# sparsity = (sparsity > tol2).sum()
# print('Test sparsity = {} out of {}'.format(sparsity, features_test.shape[-1]))

# sparsity = (features_train > tol).mean(axis=0).max(axis=0)*100
sparsity = (features_train > tol).mean(axis=0).min(axis=0)*100
useful_filters = np.where(sparsity > tol2)[0]
sparsity = (sparsity > tol2).sum()
print('Train sparsity = {} out of {}'.format(sparsity, features_train.shape[-1]))

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         (100, 320, 2)             0         
# _________________________________________________________________
# ComplexConv1 (ComplexConv1D) (100, 15, 400)            16400     
# _________________________________________________________________
# ModRelu (Modrelu)            (100, 15, 400)            200       
# _________________________________________________________________
# Abs (GetAbs)                 (100, 15, 200)            0         
# =================================================================
# print(model2.layers[-1].output.get_shape) # (batch_size, 15, 200) 

newInput = Input(batch_shape=(1, 320, 2))
newOutputs = model2(newInput)
model = Model(newInput, newOutputs)
input_signal = model.inputs[0]

print(input_signal._keras_shape) # (1, 320, 2)
print(model.output._keras_shape) # (1, 15, 100) or (1, 11, 100)


# filter_index = 0  # can be any integer from 0 to 199, as there are 200 filters in that layer
# filter_indices = useful_filters[list(np.arange(0, 5))]
# filter_indices = useful_filters[list(np.arange(5, 10))]
# filter_indices = useful_filters[list(np.arange(10, 15))]
filter_indices = useful_filters[list(np.arange(10, 13))]
# filter_indices = np.arange(10, 15)

epochs = 500
# step = 1
step = 0.01
momentum = 0.5

decay = 0.01

input_data_init = np.random.random((1, 320)) + 1j*np.random.random((1, 320))
# input_data_init = np.random.uniform(low=-1.0, high=1.0, size=(1, 320)) + 1j*np.random.uniform(low=-1.0, high=1.0, size=(1, 320))
# input_data_init = np.ones((1, 320)) + 1j*np.ones((1,320))

input_data_init /= np.sqrt(np.mean(input_data_init * np.conjugate(input_data_init)))
input_data_init = np.concatenate((input_data_init.real[..., None], input_data_init.imag[..., None]), axis=2)

num_filters = filter_indices.size
plt.figure(figsize=(10, 6))

for index in range(num_filters):
	filter_index = filter_indices[index]
	print('\n\nFilter {}'.format(filter_index))
	time_length = model.output.shape[1]
	# loss = model.output[0, time_length//2, filter_index]
	loss = model.output[0, 0, filter_index]
	# loss = K.mean(model.output[:, :, filter_index])

	# loss = model.output[time_length//2, filter_index]
	# loss = K.mean(model.output[:, filter_index])

	# loss += decay * K.sum(K.square(input_signal[:, :, 0]) + K.square(input_signal[:, :, 1]))
	# loss += decay * K.sum(K.square(input_signal))

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

	input_complex = input_complex[:100]
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
# plt.suptitle('Layer 2')
plt.suptitle('Layer 1')
plt.tight_layout(rect=[0.01, 0.01, 0.98, 0.99], h_pad=0.2)

ipdb.set_trace()
# plt.savefig('layer_2_c', format='pdf', dpi=1000, bbox_inches='tight')

