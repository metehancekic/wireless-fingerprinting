'''
models
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import numpy.random as random
from collections import OrderedDict as odict
from sklearn import metrics

import keras
from keras import backend as K
from keras.utils import plot_model
from keras import optimizers, regularizers, losses
from keras.layers import Dense, Input, Activation, Conv1D, Dropout, GlobalAveragePooling1D, Lambda, Average
from keras.models import Model, load_model
from keras.regularizers import l2

# # import complexnn
from .complexnn import ComplexDense, ComplexConv1D, utils

from .models_adsb import Modrelu

def set_keras_backend(backend):
	if K.backend() != backend:
		os.environ['KERAS_BACKEND'] = backend
		reload(K)
		assert K.backend() == backendr

set_keras_backend("theano")


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
			  kernel_regularizer=l2(weight_decay), 
			  name="Dense1")(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal", 
			  name="Dense2")(o)

	return x , model_name


def network_20_reim(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay))
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 20
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

	# o = Dropout(0.5)(o)

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal", 
			  name="Dense2")(o)

	model_name = model_name + "_reim"

	return x , model_name


def network_20_reim_2x(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay))
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 20
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
	filters = 200
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

	model_name = model_name + "_reim_2x"

	return x , model_name


def network_20_reim_sqrt2x(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay))
	model_name = "MODEL:"

	########################
	filters = 140
	k_size = 20
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
	filters = 140
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

	model_name = model_name + "_reim_sqrt2x"

	return x , model_name


def network_20_mag(data_input, classes_num=10, weight_decay=1e-4, num_features=320):

	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay))
	model_name = "MODEL:"


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

	x = Dense(classes_num,
			  activation='softmax',
			  kernel_initializer="he_normal", name="Dense2")(o)


	model_name = model_name + "_mag"

	return x, model_name



	

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

	return x , model_name

def network_200_modrelu_short_shared(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=True,
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
	o = GlobalAveragePooling1D(name="Avg")(o)
	model_name = model_name + "-Avg"
	########################

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal", 
			  name="Dense3")(o)

	return x , model_name


def network_200_reim(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay))
	model_name = "MODEL:"

	########################
	filters = 100
	k_size = 200
	strides = 100
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

	# o = Dropout(0.5)(o)

	x = Dense(classes_num, 
			  activation='softmax', 
			  kernel_initializer="he_normal", 
			  name="Dense2")(o)

	model_name = model_name + "_reim"

	return x , model_name


def network_200_reim_2x(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay))
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 200
	strides = 100
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', 
			   name="Conv1", **convArgs)(data_input)
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
			   activation='relu', 
			   name="Conv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

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

	model_name = model_name + "_reim_2x"

	return x , model_name


def network_200_reim_sqrt2x(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network that gets 99% acc on 20 MHz WiFi-2 data without channel
	'''
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay))
	model_name = "MODEL:"

	########################
	filters = 140
	k_size = 200
	strides = 100
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', 
			   name="Conv1", **convArgs)(data_input)
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
			   activation='relu', 
			   name="Conv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

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

	model_name = model_name + "_reim_sqrt2x"

	return x , model_name


def network_200_mag(data_input, classes_num=10, weight_decay=1e-4, num_features=320):

	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay))
	model_name = "MODEL:"


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

	x = Dense(classes_num,
			  activation='softmax',
			  kernel_initializer="he_normal", name="Dense2")(o)


	model_name = model_name + "_mag"

	return x, model_name



	