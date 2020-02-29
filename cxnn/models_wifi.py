from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from collections import OrderedDict as odict
import math

import keras
from keras import backend as K
from keras.utils import plot_model
from keras import optimizers, regularizers, losses
from keras.layers import Dense, Input, Activation, Conv1D, Dropout, GlobalAveragePooling1D, merge, Maximum, Add, Lambda, Concatenate
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.engine.topology import Layer
import numpy as np

# import complexnn
from .complexnn import ComplexDense, ComplexConv1D, utils


def set_keras_backend(backend):
	if K.backend() != backend:
		os.environ['KERAS_BACKEND'] = backend
		reload(K)
		assert K.backend() == backendr

set_keras_backend("theano")

class Modrelu(Layer):

	def __init__(self, **kwargs):
		super(Modrelu, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self._b = self.add_weight(name='b', 
									shape=(input_shape[-1]//2,),
									initializer='zeros',
									trainable=True)
		super(Modrelu, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		
		real = utils.GetReal()(x)
		imag = utils.GetImag()(x)

		abs1 = K.relu( utils.GetAbs()(x) )
		abs2 = K.relu( utils.GetAbs()(x) - self._b )

		
		real = real * abs2 / (abs1 + 0.0000001)
		imag = imag * abs2 / (abs1 + 0.0000001)

		merged = Concatenate()([real, imag])

		return merged

	def compute_output_shape(self, input_shape):
		return input_shape


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

def output_of_lambda(input_shape):
	return (input_shape[0], input_shape[2])

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
	filters = 100
	k_size = 16
	strides = 8
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
	filters = 200

	k_size = 160
	strides = 80

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

def net_preamble_20_real(data_input1, data_input2, data_input3, data_input4, classes_num=10, weight_decay=1e-4):
	
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 160
	strides = 80
	shared_conv = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu',
					  name="ComplexConv1", **convArgs)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides) + "shared"
	########################


	o0 = shared_conv(data_input1)
	o1 = shared_conv(data_input2)
	o2 = shared_conv(data_input3)
	o3 = shared_conv(data_input4)

	# o =  merge([o0, o1, o2, o3], mode='max', concat_axis = 1)
	o = Maximum()([o0, o1, o2, o3])

	# cross1 = Lambda(max_out, output_shape = ....)([d1,d4])
	###################################



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
	########################

	########################
	# o = utils.GetAbs()(o)
	# model_name = model_name + "-Abs"
	########################

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


def preamble_block(data_input, classes_num=10, weight_decay=1e-4):
	
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 160
	strides = 80
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu',
					  name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides) + "shared"
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
	########################

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


	return o, model_name


def output_of_lambda(input_shape):
	return input_shape

output_shape=output_of_lambda


def preamble_network(x, classes_num=10, weight_decay=1e-4, batch_size=100, is_training=True):

	model_name = "MODEL: LLR"

	pre_softmax = Dense(classes_num,
			  activation=None,
			  kernel_initializer="he_normal", name="PreSoftmax")

	if is_training is True:
		logits, model_name = preamble_block(x, weight_decay)
		llr = pre_softmax(logits)		
		out_test = None
		out_train = Lambda(softmax_manual, output_shape=output_of_lambda, name="Softmax")(llr)

	else:
		logits = [None] * len(x)
		llr = [None] * len(x)
		for i in range(len(x)):
			logits[i], model_name = preamble_block(x[i], weight_decay)
			llr[i] = pre_softmax(logits[i])

		added_llr = Add(name='llr')(llr)

		out_test = Lambda(softmax_manual, output_shape=output_of_lambda, name="Softmax")(added_llr)

		out_train = Lambda(softmax_manual, output_shape=output_of_lambda, name="Softmax")(llr[0])


	return out_test, out_train, model_name

# class SlidingWindow200(Layer):

# 	def __init__(self, **kwargs):
# 		super(SlidingWindow200, self).__init__(**kwargs)

# 	def build(self, input_shape):
# 		# Create a trainable weight variable for this layer.
		
# 		super(SlidingWindow200, self).build(input_shape)  # Be sure to call this at the end

# 	def call(self, x):

# 		real = utils.GetReal()(x)
# 		imag = utils.GetImag()(x)


# 		for i in range(input_shape[1]-100+1):
# 			y_real = Lambda(lambda x: x[:,i:100+i,:], output_shape = output_of_lambda[0:1] + (100,) + output_of_lambda[1:2] )(real)
# 			y_imag = Lambda(lambda x: x[:,i:100+i,:], output_shape = output_of_lambda[0:1] + (100,) + output_of_lambda[1:2] )(imag)

	
		
# 		real = real * abs2 / (abs1 + 0.0000001)
# 		imag = imag * abs2 / (abs1 + 0.0000001)

# 		merged = Concatenate()([real, imag])

# 		return merged

# 	def compute_output_shape(self, input_shape):
# 		return input_shape


def network_200_shared_modrelu_short_conv_sliding_window(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 200 MHz data with channel
	'''
	shapee = K.int_shape(data_input)

	shape_ = K.int_shape(data_input)[1]

	real = utils.GetReal()(data_input)
	imag = utils.GetImag()(data_input)
	
	windows_real = [None] * (shape_-100+1) 
	windows_imag = [None] * (shape_-100+1)


	for i in range(shape_-100+1):
		windows_real[i] = Lambda(lambda x: x[:,i:100+i,:], output_shape = shapee[0:1] + (100,) + shapee[1:2] )(real)
		windows_imag[i] = Lambda(lambda x: x[:,i:100+i,:], output_shape = shapee[0:1] + (100,) + shapee[1:2] )(imag)

	merged = Concatenate()(windows_real + windows_imag)



	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:slide"

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
	# 4 symbols
	#############
	k_size = 40
	strides = 10

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
	# k_size = 200
	# strides = 100
	# strides = 10

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


def network_200_shared_modrelu_short_conv_sliding_window_sym_stride(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 200 MHz data with channel
	'''
	shapee = K.int_shape(data_input)

	shape_ = K.int_shape(data_input)[1]

	real = utils.GetReal()(data_input)
	imag = utils.GetImag()(data_input)
	
	windows_real = [None] * ((shape_-100+10)//10) 
	windows_imag = [None] * ((shape_-100+10)//10)


	for i in np.arange(0,shape_-100+10,10):
		windows_real[i//10] = Lambda(lambda x: x[:,i:100+i,:], output_shape = shapee[0:1] + (100,) + shapee[1:2] )(real)
		windows_imag[i//10] = Lambda(lambda x: x[:,i:100+i,:], output_shape = shapee[0:1] + (100,) + shapee[1:2] )(imag)

	merged = Concatenate()(windows_real + windows_imag)



	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:slide"

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
	# 4 symbols
	#############
	k_size = 40
	strides = 10

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
	# k_size = 200
	# strides = 100
	# strides = 10

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
def stft_network(data_input, classes_num=10, weight_decay = 1e-4):


	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 320
	strides = 160


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
	k_size = 3
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
			  name="LShared_Dense2")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	o = shared_dense2(o)

	########################
	neuron_num = 19
	shared_dense3 = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="LShared_Dense2")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	o = shared_dense3(o)

	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################
		

	# x = Dense(classes_num,
	# 		 activation='softmax',
	# 		 kernel_initializer="he_normal")(o)

	return o

def ltft_network(data_input, classes_num=10, weight_decay = 1e-4):


	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 320
	strides = 160


	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None,
					  name="LComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################


	########################
	o = Modrelu(name="LModRelu1")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	filters = 100
	k_size = 3
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, 
					  name="LComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = Modrelu(name="LModRelu2")(o)
	model_name = model_name + "-ModReLU"
	########################

	########################
	o = utils.GetAbs(name="LAbs")(o)
	model_name = model_name + "-Abs"
	########################

	########################
	neuron_num = 100
	shared_dense = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="LShared_Dense1")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	o = shared_dense(o)

	########################
	neuron_num = 100
	shared_dense2 = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="LShared_Dense2")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	o = shared_dense2(o)

	########################
	neuron_num = 19
	shared_dense3 = Dense(neuron_num,
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay),
			  name="LShared_Dense2")
	model_name = model_name + "-" + str(neuron_num) + "shared_D"
	########################

	o = shared_dense3(o)

	########################
	o = GlobalAveragePooling1D()(o)
	model_name = model_name + "-Avg"
	########################
		

	# x = Dense(classes_num,
	# 		 activation='softmax',
	# 		 kernel_initializer="he_normal")(o)

	return o


def network_200_shared_modrelu_parallelnetworks(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 200 MHz data with channel
	'''
	shapee = K.int_shape(data_input)

	# print(K.int_shape(data_input))

	stft = Lambda(lambda x: x[:,:shapee[1]//2,:], output_shape = (shapee[1]//2,) + (shapee[2],) )(data_input)
	ltft = Lambda(lambda x: x[:,shapee[1]//2:,:], output_shape = (shapee[1]//2,) + (shapee[2],))(data_input)

	# print(K.int_shape(ltft))
	# print(K.int_shape(stft))
	
	so = stft_network(stft, classes_num, weight_decay)
	lo = ltft_network(ltft, classes_num, weight_decay)
	o = Add(name = 'llr')([so, lo])

	x = Lambda(softmax_manual, output_shape=(19,), name="Softmax")(o)
	
	# x = Dense(classes_num,
	# 		 activation='softmax',
	# 		 kernel_initializer="he_normal")(o)

	model_name = "parallel_network"

	return x, model_name














