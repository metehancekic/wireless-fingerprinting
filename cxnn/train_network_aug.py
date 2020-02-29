from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import numpy.random as random
from collections import OrderedDict as odict

import keras
from keras import backend as K
from keras.utils import plot_model
from keras import optimizers, regularizers, losses
from keras.layers import Dense, Input, Activation, Conv1D, Dropout, GlobalAveragePooling1D, Lambda, Average
from keras.models import Model, load_model
from keras.regularizers import l2

# import complexnn
from .complexnn import ComplexDense, ComplexConv1D, utils

from .models_adsb import net_address_20, net_preamble_20, net_preamble_20_no_crelu, net_preamble_20_real, RotateComplex, output_of_lambda, relu_manual, net_preamble_20_rotated, RotateComplex, relu_manual, Modrelu, net_preamble_20_modrelu, net_preamble_20_rot16_real, net_preamble_50, net_preamble_100, softmax_manual

from ..preproc.preproc_wifi import rms
from ..preproc.fading_model import normalize

from sklearn import metrics

def set_keras_backend(backend):
	if K.backend() != backend:
		os.environ['KERAS_BACKEND'] = backend
		reload(K)
		assert K.backend() == backendr

set_keras_backend("theano")

def network_20_2(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 100
	strides = 10
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	filters = 50
	k_size = 10
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################

	########################
	filters = 50
	k_size = 10
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', 
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation=None,
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal")(o)

	# x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
			  # kernel_regularizer=regularizers.l2(weight_decay))(o)
	return x , model_name

def network_20_wifi2(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 100
	strides = 10
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################

	########################
	filters = 200
	k_size = 10
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', 
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation=None,
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal")(o)

	# x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
			  # kernel_regularizer=regularizers.l2(weight_decay))(o)
	return x , model_name

def network_20_wifi2_new(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 100
	strides = 10
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################

	########################
	filters = 200
	k_size = 10
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', 
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal")(o)

	# x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
			  # kernel_regularizer=regularizers.l2(weight_decay))(o)
	return x , model_name

def network_20_wifi2_new_2(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 100
	strides = 10
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	filters = 200
	k_size = 10
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	
	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = ComplexDense(neuron_num,
			  activation='relu', 
			  kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal")(o)

	# x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
			  # kernel_regularizer=regularizers.l2(weight_decay))(o)
	return x , model_name

def network_20(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 20 MHz data (eg. after fractionally spaced equalization) with channel
	'''
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 10
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################

	########################
	filters = 200
	k_size = 10
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', 
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal")(o)

	# x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
			  # kernel_regularizer=regularizers.l2(weight_decay))(o)
	return x , model_name

def network_20_new(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 20 MHz data (eg. after fractionally spaced equalization) with channel
	'''
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 30
	strides = 10
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################

	########################
	filters = 200
	k_size = 5
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', 
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal")(o)

	# x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
			  # kernel_regularizer=regularizers.l2(weight_decay))(o)
	return x , model_name

def network_200(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 200 MHz data with channel
	'''
	
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 200
	strides = 100
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################

	########################
	filters = 200
	k_size = 5
	strides = 5
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', 
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation=None,
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num,
			 activation='softmax',
			 kernel_initializer="he_normal")(o)

	return x, model_name

def network_200_new(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 200 MHz data with channel
	'''
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 100
	strides = 10
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################

	########################
	filters = 200
	k_size = 10
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', 
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################


	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num,
			 activation='softmax',
			 kernel_initializer="he_normal")(o)

	return x, model_name

# def output_of_lambda(input_shape):
# 	return (input_shape[0], input_shape[2])

def network_20_modrelu_short(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	# filters = 100
	# k_size = 16
	# strides = 8

	# filters = 300

	filters = 100

	#############
	# 2 symbols
	#############
	# k_size = 2
	# strides = 1

	#############
	# 5 symbols
	#############
	k_size = 5
	strides = 2

	#############
	# 10 symbols
	#############
	# k_size = 10
	# strides = 5

	#############
	# 16 symbols
	#############
	# k_size = 16
	# strides = 8

	#############
	# 20 symbols
	#############
	# k_size = 20
	# strides = 10

	#############
	# 100 symbols
	#############
	# k_size = 100
	# strides = 10

	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	# o = Dropout(0.5)(o)

	########################
	o = Modrelu(name="ModRelu1")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	filters = 100
	k_size = 10
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = Modrelu(name="ModRelu2")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	o = utils.GetAbs(name="Abs")(o)
	model_name = model_name + "-Abs"
	########################

	########################
	o = GlobalAveragePooling1D(name="Avg")(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay), 
			  name="Dense1")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal", 
			  name="Dense2")(o)

	# x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
			  # kernel_regularizer=regularizers.l2(weight_decay))(o)
	return x , model_name

def network_20_modrelu_long(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 100
	strides = 10
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = Modrelu(name="ModRelu1")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	filters = 100
	k_size = 10
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = Modrelu(name="ModRelu2")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	o = utils.GetAbs(name="Abs")(o)
	model_name = model_name + "-Abs"
	########################

	########################
	o = GlobalAveragePooling1D(name="Avg")(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay), 
			  name="Dense1")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal", 
			  name="Dense2")(o)

	# x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
			  # kernel_regularizer=regularizers.l2(weight_decay))(o)
	return x , model_name

def network_200_modrelu_short(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 160
	strides = 80
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = Modrelu(name="ModRelu1")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	filters = 100
	k_size = 10
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = Modrelu(name="ModRelu2")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	o = utils.GetAbs(name="Abs")(o)
	model_name = model_name + "-Abs"
	########################

	########################
	o = GlobalAveragePooling1D(name="Avg")(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay), 
			  name="Dense1")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal", 
			  name="Dense2")(o)

	# x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
			  # kernel_regularizer=regularizers.l2(weight_decay))(o)
	return x , model_name

def network_200_modrelu_long(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 200
	strides = 100
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = Modrelu(name="ModRelu1")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	filters = 100
	k_size = 10
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = Modrelu(name="ModRelu2")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	o = utils.GetAbs(name="Abs")(o)
	model_name = model_name + "-Abs"
	########################

	########################
	o = GlobalAveragePooling1D(name="Avg")(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay), 
			  name="Dense1")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal", 
			  name="Dense2")(o)

	# x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
			  # kernel_regularizer=regularizers.l2(weight_decay))(o)
	return x , model_name

def network_20_modrelu_globecom(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 20
	strides = 10
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = Modrelu(name="ModRelu1")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	filters = 100
	k_size = 10
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = Modrelu(name="ModRelu2")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	o = utils.GetAbs(name="Abs")(o)
	model_name = model_name + "-Abs"
	########################

	########################
	o = GlobalAveragePooling1D(name="Avg")(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal",
			  kernel_regularizer=l2(weight_decay), name="Dense1")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	pre_softmax = Dense(classes_num,
						activation=None,
						kernel_initializer="he_normal", name="Dense2")(o)

	x = Lambda(softmax_manual, output_shape=output_of_lambda, name="Softmax")(pre_softmax)

	model_name = model_name + "_reim"
	return x, pre_softmax, model_name

def network_20_crelu_globecom(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 20
	strides = 10
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', 
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################


	########################
	filters = 100
	k_size = 10
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', 
					  name="ComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################


	########################
	o = utils.GetAbs(name="Abs")(o)
	model_name = model_name + "-Abs"
	########################

	########################
	o = GlobalAveragePooling1D(name="Avg")(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal",
			  kernel_regularizer=l2(weight_decay), name="Dense1")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	pre_softmax = Dense(classes_num,
						activation=None,
						kernel_initializer="he_normal", name="Dense2")(o)

	x = Lambda(softmax_manual, output_shape=output_of_lambda, name="Softmax")(pre_softmax)

	model_name = model_name + "_reim"
	return x, pre_softmax, model_name

def network_20_avg_old(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 200 MHz data with channel
	'''
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 10
	strides = 5
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu',
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################

	########################
	filters = 100
	k_size = 10
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu',
			   name="Conv2", 
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	neuron_num = 100
	shared_dense = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="Shared_Dense1")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	temporal_length = K.int_shape(o)[1]
	parallel_out = [None] * temporal_length

	for indexx in range(temporal_length):

		parallel_out[indexx] = Lambda(lambda o : o[:,indexx,:], output_shape=output_of_lambda)(o)

	for indexx in range(temporal_length):

		parallel_out[indexx] = shared_dense(parallel_out[indexx])


	########################
	neuron_num = 100
	shared_dense2 = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="Shared_Dense2")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################


	for indexx in range(temporal_length):

		parallel_out[indexx] = shared_dense2(parallel_out[indexx])

	o = Average()(parallel_out)

	x = Dense(classes_num,
			 activation='softmax',
			 kernel_initializer="he_normal")(o)

	return x, model_name

def network_20_avg_new(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 200 MHz data with channel
	'''
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 16
	strides = 8
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu',
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################

	########################
	filters = 100
	k_size = 10
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu',
			   name="Conv2", 
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	neuron_num = 100
	shared_dense = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="Shared_Dense1")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	o = shared_dense(o)


	########################
	neuron_num = 100
	shared_dense2 = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="Shared_Dense2")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	o = shared_dense2(o)

	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################
		

	x = Dense(classes_num,
			 activation='softmax',
			 kernel_initializer="he_normal")(o)

	return x, model_name

def network_200_avg_old(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 200 MHz data with channel
	'''
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 100
	strides = 50
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu',
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################

	########################
	filters = 100
	k_size = 10
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu',
			   name="Conv2", 
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	neuron_num = 100
	shared_dense = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="Shared_Dense1")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	temporal_length = K.int_shape(o)[1]
	parallel_out = [None] * temporal_length

	for indexx in range(temporal_length):

		parallel_out[indexx] = Lambda(lambda o : o[:,indexx,:], output_shape=output_of_lambda)(o)

	for indexx in range(temporal_length):

		parallel_out[indexx] = shared_dense(parallel_out[indexx])


	########################
	neuron_num = 100
	shared_dense2 = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="Shared_Dense2")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################


	for indexx in range(temporal_length):

		parallel_out[indexx] = shared_dense2(parallel_out[indexx])

	o = Average()(parallel_out)

	x = Dense(classes_num,
			 activation='softmax',
			 kernel_initializer="he_normal")(o)

	return x, model_name

def network_200_avg_new(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 200 MHz data with channel
	'''
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	# filters = 400

	filters = 200

	#############
	# 2 symbols
	#############
	# k_size = 20
	# strides = 10

	#############
	# 5 symbols
	#############
	# k_size = 50
	# strides = 25

	#############
	# 10 symbols
	#############
	# k_size = 100
	# strides = 50

	#############
	# 16 symbols
	#############
	k_size = 160
	strides = 80

	#############
	# 20 symbols
	#############
	# k_size = 200
	# strides = 100

	#############
	# 100 symbols
	#############
	# k_size = 1000
	# strides = 100

	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu',
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	# o = Dropout(0.5)(o)

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################

	########################
	filters = 200
	k_size = 10
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu',
			   name="Conv2", 
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	neuron_num = 100
	shared_dense = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="Shared_Dense1")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	o = shared_dense(o)


	########################
	neuron_num = 100
	shared_dense2 = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="Shared_Dense2")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	o = shared_dense2(o)

	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################
		

	x = Dense(classes_num,
			 activation='softmax',
			 kernel_initializer="he_normal")(o)

	return x, model_name

def network_20_real_short(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 100
	strides = 10
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = Modrelu(name="ModRelu1")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	filters = 100
	k_size = 10
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = Modrelu(name="ModRelu2")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	o = utils.GetAbs(name="Abs")(o)
	model_name = model_name + "-Abs"
	########################

	########################
	o = GlobalAveragePooling1D(name="Avg")(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay), 
			  name="Dense1")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal", 
			  name="Dense2")(o)

	# x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
			  # kernel_regularizer=regularizers.l2(weight_decay))(o)
	return x , model_name

def network_20_real_long(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay))
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 100
	strides = 10
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', 
			   name="Conv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################


	########################
	filters = 100
	k_size = 10
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', 
			   name="Conv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	# ########################
	# o = Modrelu(name="ModRelu2")(o)
	# model_name = model_name + "-ModReLU"
	# ########################

	# ########################
	# o = utils.GetAbs(name="Abs")(o)
	# model_name = model_name + "-Abs"
	# ########################

	########################
	o = GlobalAveragePooling1D(name="Avg")(o)
	model_name = model_name + "-Avg"
	########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay), 
			  name="Dense1")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal", 
			  name="Dense2")(o)

	# x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
			  # kernel_regularizer=regularizers.l2(weight_decay))(o)
	return x , model_name


def network_200_shared_modrelu(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 200 MHz data with channel
	'''
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	# filters = 400
	# filters = 200
	filters = 100

	#############
	# 2 symbols
	#############
	# k_size = 20
	# strides = 10

	#############
	# 5 symbols
	#############
	# k_size = 50
	# strides = 25

	#############
	# 10 symbols
	#############
	# k_size = 100
	# strides = 50

	#############
	# 16 symbols
	#############
	# k_size = 160
	# strides = 80

	#############
	# 20 symbols
	#############
	k_size = 200
	strides = 100

	#############
	# 100 symbols
	#############
	# k_size = 1000
	# strides = 100

	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None,
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	# o = Dropout(0.5)(o)

	########################
	o = Modrelu(name="ModRelu1")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	filters = 100
	k_size = 10
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = Modrelu(name="ModRelu2")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	o = utils.GetAbs(name="Abs")(o)
	model_name = model_name + "-Abs"
	########################

	########################
	neuron_num = 100
	shared_dense = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="Shared_Dense1")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	o = shared_dense(o)

	########################
	neuron_num = 100
	shared_dense2 = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="Shared_Dense2")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	o = shared_dense2(o)

	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################
		

	x = Dense(classes_num,
			 activation='softmax',
			 kernel_initializer="he_normal")(o)

	return x, model_name

def network_20_shared_modrelu(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 20 MHz data with channel
	'''
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	# filters = 400
	# filters = 200
	filters = 100

	#############
	# 2 symbols
	#############
	# k_size = 2
	# strides = 1

	#############
	# 5 symbols
	#############
	# k_size = 5
	# strides = 2

	#############
	# 10 symbols
	#############
	# k_size = 10
	# strides = 5

	#############
	# 16 symbols
	#############
	# k_size = 16
	# strides = 8

	#############
	# 20 symbols
	#############
	k_size = 20
	strides = 10

	#############
	# 100 symbols
	#############
	# k_size = 100
	# strides = 10

	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None,
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	# o = Dropout(0.5)(o)

	########################
	o = Modrelu(name="ModRelu1")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	filters = 100
	k_size = 10
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="ComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = Modrelu(name="ModRelu2")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	o = utils.GetAbs(name="Abs")(o)
	model_name = model_name + "-Abs"
	########################

	########################
	neuron_num = 100
	shared_dense = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="Shared_Dense1")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	o = shared_dense(o)

	########################
	neuron_num = 100
	shared_dense2 = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="Shared_Dense2")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	o = shared_dense2(o)

	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################
		

	x = Dense(classes_num,
			 activation='softmax',
			 kernel_initializer="he_normal")(o)

	return x, model_name

def net_reim(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay))
	model_name = "MODEL:"


	##################################
	filters = 100
	k_size = 20
	strides = 10
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', name="Conv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	##################################
	filters = 100
	k_size = 10
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', name="Conv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = GlobalAveragePooling1D(name="GlobalAvg")(o)
	model_name = model_name + "-Avg"
	########################

	# ########################
	# d_rate = 0.2
	# o = Dropout(d_rate)(o)
	# model_name = model_name + "-d({:.2f})".format(d_rate)
	# ########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal",
			  kernel_regularizer=l2(weight_decay), name="Dense1")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	pre_softmax = Dense(classes_num,
						activation=None,
						kernel_initializer="he_normal", name="Dense2")(o)

	x = Lambda(softmax_manual, output_shape=output_of_lambda, name="Softmax")(pre_softmax)

	model_name = model_name + "_reim"

	return x, pre_softmax, model_name


def net_reim_2x(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay))
	model_name = "MODEL:"


	##################################
	filters = 200
	k_size = 20
	strides = 10
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', name="Conv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	filters = 200
	k_size = 10
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', name="Conv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = GlobalAveragePooling1D(name="GlobalAvg")(o)
	model_name = model_name + "-Avg"
	########################


	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal",
			  kernel_regularizer=l2(weight_decay), name="Dense1")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	pre_softmax = Dense(classes_num,
						activation=None,
						kernel_initializer="he_normal", name="Dense2")(o)

	x = Lambda(softmax_manual, output_shape=output_of_lambda, name="Softmax")(pre_softmax)

	model_name = model_name + "_reim_2x"

	return x, pre_softmax, model_name


def net_reim_sqrt2x(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay))
	model_name = "MODEL:"


	##################################
	filters = 140
	k_size = 20
	strides = 10
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', name="Conv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	filters = 140
	k_size = 10
	strides = 1
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', name="Conv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = GlobalAveragePooling1D(name="GlobalAvg")(o)
	model_name = model_name + "-Avg"
	########################


	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal",
			  kernel_regularizer=l2(weight_decay), name="Dense1")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	pre_softmax = Dense(classes_num,
						activation=None,
						kernel_initializer="he_normal", name="Dense2")(o)

	x = Lambda(softmax_manual, output_shape=output_of_lambda, name="Softmax")(pre_softmax)

	model_name = model_name + "_reim_2x"

	return x, pre_softmax, model_name


def net_mag(data_input, classes_num=10, weight_decay=1e-4, num_features=320):

	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay))
	model_name = "MODEL:"


	# ########################
	# filters = 10
	# k_size = 40
	# strides = 20
	# o = Conv1D(input_shape=(num_features, 1),
	# 		   filters=filters, 
	# 		   kernel_size=[k_size], 
	# 		   strides=strides, 
	# 		   padding='valid', 
	# 		   activation='relu', name="Conv1", **convArgs)(data_input)
	# model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	# ########################

	# ########################
	# filters = 10
	# k_size = 5
	# strides = 1
	# o = Conv1D(filters=filters, 
	# 		   kernel_size=[k_size], 
	# 		   strides=strides, 
	# 		   padding='valid', 
	# 		   activation='relu', name="Conv2", **convArgs)(o)
	# model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	# ########################

	# ########################
	# o = GlobalAveragePooling1D(name="GlobalAvg")(o)
	# model_name = model_name + "-Avg"
	# ########################

	# ########################
	# d_rate = 0.2
	# o = Dropout(d_rate)(o)
	# model_name = model_name + "-d({:.2f})".format(d_rate)
	# ########################

	o = data_input

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal",
			  kernel_regularizer=l2(weight_decay), name="Dense_")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	# ########################
	# d_rate = 0.5
	# o = Dropout(d_rate)(o)
	# model_name = model_name + "-d({:.2f})".format(d_rate)
	# ########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal",
			  kernel_regularizer=l2(weight_decay), name="Dense__")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	# ########################
	# d_rate = 0.5
	# o = Dropout(d_rate)(o)
	# model_name = model_name + "-d({:.2f})".format(d_rate)
	# ########################

	########################
	neuron_num = 100
	o = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal",
			  kernel_regularizer=l2(weight_decay), name="Dense1")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	pre_softmax = Dense(classes_num,
						activation=None,
						kernel_initializer="he_normal", name="Dense2")(o)

	x = Lambda(softmax_manual, output_shape=output_of_lambda, name="Softmax")(pre_softmax)

	model_name = model_name + "_mag"

	return x, pre_softmax, model_name

def train(dict_data, checkpoint_in=None, checkpoint_out=None, architecture='modfrelu'):

	x_train = dict_data['x_train']
	y_train = dict_data['y_train']
	x_test = dict_data['x_test']
	y_test = dict_data['y_test']
	num_classes = dict_data['num_classes']
	num_train = x_train.shape[0]
	num_test = x_test.shape[0]
	num_features = x_train.shape[1]

	print('Training data size: {}'.format(x_train.shape))
	print('Test data size: {}'.format(x_test.shape))

	batch_size = 100	
	epochs = 200
	# epochs = 500
	weight_decay = 1e-3

	n_val = batch_size
	x_val = x_test[:n_val].copy()
	y_val = y_test[:n_val].copy()

	print("========================================") 
	print("MODEL HYPER-PARAMETERS") 
	print("BATCH SIZE: {:3d}".format(batch_size)) 
	print("WEIGHT DECAY: {:.4f}".format(weight_decay))
	print("EPOCHS: {:3d}".format(epochs))
	print("========================================") 
	print("== BUILDING MODEL... ==")

	if checkpoint_in is None:


		# Define input shape
		if architecture not in ['magnitude', 'phase', 're', 'im']:
			data_input = Input(batch_shape=(None, num_features, 2))
		else:
			data_input = Input(batch_shape=(None, num_features))

		# Load architecture
		if architecture=='reim':
			output, pre_softmax, model_name = net_reim(data_input, num_classes, weight_decay)
		elif architecture=='reim2x':
			output, pre_softmax, model_name = net_reim_2x(data_input, num_classes, weight_decay)
		elif architecture=='reimsqrt2x':
			output, pre_softmax, model_name = net_reim_sqrt2x(data_input, num_classes, weight_decay)
		elif architecture=='magnitude':
			x_train = np.abs(x_train[..., 0] + 1j*x_train[..., 1])
			x_test = np.abs(x_test[..., 0] + 1j*x_test[..., 1])
			x_val = np.abs(x_val[..., 0] + 1j*x_val[..., 1])

			minim = x_train.min()
			maxim = x_train.max()
			x_train = (x_train - minim) / (maxim - minim)
			x_test = (x_test - minim) / (maxim - minim)
			x_val = (x_val - minim) / (maxim - minim)

			# mean = x_train.mean()
			# std = x_train.std()
			# x_train = (x_train - mean) / std
			# x_test = (x_test - mean) / std
			# x_val = (x_val - mean) / std

			output, pre_softmax, model_name = net_mag(data_input, num_classes, weight_decay, num_features)
		elif architecture=='phase':
			x_train = np.angle(x_train[..., 0] + 1j*x_train[..., 1])
			x_test = np.angle(x_test[..., 0] + 1j*x_test[..., 1])
			x_val = np.angle(x_val[..., 0] + 1j*x_val[..., 1])

			minim = x_train.min()
			maxim = x_train.max()
			x_train = (x_train - minim) / (maxim - minim)
			x_test = (x_test - minim) / (maxim - minim)
			x_val = (x_val - minim) / (maxim - minim)

			# mean = x_train.mean()
			# std = x_train.std()
			# x_train = (x_train - mean) / std
			# x_test = (x_test - mean) / std
			# x_val = (x_val - mean) / std

			output, pre_softmax, model_name = net_mag(data_input, num_classes, weight_decay, num_features)
		elif architecture=='re':
			x_train = x_train[..., 0]
			x_test = x_test[..., 0]
			x_val = x_val[..., 0]

			minim = x_train.min()
			maxim = x_train.max()
			x_train = (x_train - minim) / (maxim - minim)
			x_test = (x_test - minim) / (maxim - minim)
			x_val = (x_val - minim) / (maxim - minim)

			# mean = x_train.mean()
			# std = x_train.std()
			# x_train = (x_train - mean) / std
			# x_test = (x_test - mean) / std
			# x_val = (x_val - mean) / std

			output, pre_softmax, model_name = net_mag(data_input, num_classes, weight_decay, num_features)
		elif architecture=='im':
			x_train = x_train[..., 1]
			x_test = x_test[..., 1]
			x_val = x_val[..., 1]

			minim = x_train.min()
			maxim = x_train.max()
			x_train = (x_train - minim) / (maxim - minim)
			x_test = (x_test - minim) / (maxim - minim)
			x_val = (x_val - minim) / (maxim - minim)

			# mean = x_train.mean()
			# std = x_train.std()
			# x_train = (x_train - mean) / std
			# x_test = (x_test - mean) / std
			# x_val = (x_val - mean) / std

			output, pre_softmax, model_name = net_mag(data_input, num_classes, weight_decay, num_features)
		elif architecture=='modrelu':
			output, pre_softmax, model_name = network_20_modrelu_globecom(data_input, num_classes, weight_decay)
		elif architecture=='crelu':
			output, pre_softmax, model_name = network_20_crelu_globecom(data_input, num_classes, weight_decay)
		# elif architecture=='fullsignal':
		# 	output, model_name = net_address_20(data_input, num_classes, weight_decay)
		# 	output, model_name = network_20_modrelu_long(data_input, num_classes, weight_decay)

		# data_input = Input(batch_shape=(batch_size, num_features, 2))

		# output, model_name = network_200_shared_modrelu(data_input, num_classes, weight_decay)
		# # output, model_name = network_20_shared_modrelu(data_input, num_classes, weight_decay)

		# # output, model_name = network_200_avg_new(data_input, num_classes, weight_decay)

		# # output, model_name = network_20_avg_new(data_input, num_classes, weight_decay)

		# # output, model_name = network_200_modrelu_short(data_input, num_classes, weight_decay)
		# # output, model_name = network_200_modrelu_long(data_input, num_classes, weight_decay)

		# # output, model_name = network_20_modrelu_short(data_input, num_classes, weight_decay)
		# # output, model_name = network_20_modrelu_long(data_input, num_classes, weight_decay)

		# checkpoint_out += '' +'-new-'

		# # output, model_name = network_20_real_short(data_input, num_classes, weight_decay)
		# # output, model_name = network_20_real_long(data_input, num_classes, weight_decay)
		
		# # output, model_name = network_20_modrelu_short(data_input, num_classes, weight_decay)
		# # output, model_name = network_20_2(data_input, num_classes, weight_decay)
		# # output, model_name = network_200(data_input, num_classes, weight_decay)
		# # output, model_name = network_200_new(data_input, num_classes, weight_decay)
		# # output, model_name = network_200_new(data_input, num_classes, weight_decay)
		
		densenet = Model(data_input, output)
	else:
		densenet = load_model(checkpoint_in, 
							  custom_objects={'ComplexConv1D':complexnn.ComplexConv1D,
							  				  'GetAbs': utils.GetAbs})

	# Print model architecture
	print(densenet.summary())
	# plot_model(densenet, to_file='model_architecture.png')

	# Set optimizer and loss function
	optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	densenet.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	print("== START TRAINING... ==")
	history = densenet.fit(x=x_train, 
						   y=y_train, 
						   epochs=epochs, 
						   batch_size=batch_size, 
						   validation_data=(x_val, y_val), 
						   callbacks=[])



	if checkpoint_out is not None:
		checkpoint_out = checkpoint_out+'-new.h5'
		if epochs!=200:
			checkpoint_out+='-{}-ep.h5'.format(epochs)
		densenet.save(checkpoint_out)

	probs = densenet.predict(x=x_test, batch_size=batch_size, verbose=0)
	label_pred = probs.argmax(axis=1) 
	label_act = y_test.argmax(axis=1) 
	ind_correct = np.where(label_pred==label_act)[0] 
	ind_wrong = np.where(label_pred!=label_act)[0] 
	assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
	test_acc = 100.*ind_correct.size / num_test

	# conf_matrix_test = metrics.confusion_matrix(label_act, label_pred)
	# conf_matrix_test = 100*conf_matrix_test/(num_test/num_classes)
	# print('{}'.format(conf_matrix_test))
	# plt.figure()
	# plt.imshow(100*conf_matrix_test/(num_test/num_classes), vmin=0, vmax=100)
	# plt.title('Test confusion matrix')
	# plt.colorbar()

	print("\n========================================") 
	print('Test accuracy: {:.2f}%'.format(test_acc))

	output_dict = odict(acc=odict(), comp=odict(), loss=odict())

	output_dict['acc']['test'] = test_acc
	# output_dict['acc']['val'] = 100.*history.history['val_acc'][-1]
	output_dict['acc']['train'] = 100.*history.history['acc'][-1]

	# output_dict['loss']['val'] = history.history['val_loss'][-1]
	output_dict['loss']['train'] = history.history['loss'][-1]

	stringlist = []
	densenet.summary(print_fn=lambda x: stringlist.append(x))
	summary = '\n' + \
			'Batch size: {:3d}\n'.format(batch_size) + \
			'Weight decay: {:.4f}\n'.format(weight_decay) + \
			'Epochs: {:3d}\n'.format(epochs) + \
			'Optimizer:' + str(densenet.optimizer) + '\n'
	summary += '\n'.join(stringlist)

	return output_dict, model_name, summary