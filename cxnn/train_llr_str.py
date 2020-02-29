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

from .models_wifi import *
from .models_adsb import Modrelu

from ..preproc.preproc_wifi import rms
from ..preproc.fading_model import normalize

from sklearn import metrics


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


def network_200_shared_modrelu_early_abs(data_input, classes_num=10, weight_decay = 1e-4):
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
	o = utils.GetAbs(name="Abs")(o)
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
			   name="SharedConv1", 
			   use_bias=True,
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "shared_C" + str(k_size) + "x" + str(strides)
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


def network_200_shared_modrelu_small_stride(data_input, classes_num=10, weight_decay = 1e-4):
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
	# strides = 100
	strides = 10

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


def network_200_shared_modrelu_short_conv(data_input, classes_num=10, weight_decay = 1e-4):
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

def network_200_shared_modrelu_short_conv_sld(data_input, classes_num=10, weight_decay = 1e-4):
	'''
	Network for 200 MHz data with channel
	'''
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:sliding_window"

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




def network_200_shared_modrelu_short_conv_no_modrelu(data_input, classes_num=10, weight_decay = 1e-4):
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

	# ########################
	# o = Modrelu(name="ModRelu2")(o)
	# model_name = model_name + "-ModReLU"
	# ########################

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


def network_200_shared_modrelu_early_abs_short_conv(data_input, classes_num=10, weight_decay = 1e-4):
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
	o = utils.GetAbs(name="Abs")(o)
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
			   name="SharedConv1", 
			   use_bias=True,
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "shared_C" + str(k_size) + "x" + str(strides)
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




def network_200_shared_modrelu_early_abs_short_conv_no_modrelu(data_input, classes_num=10, weight_decay = 1e-4):
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
	o = utils.GetAbs(name="Abs")(o)
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
			   name="SharedConv1", 
			   use_bias=True,
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "shared_C" + str(k_size) + "x" + str(strides)
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



def train(dict_data, num_aug_test=1, checkpoint_in=None, checkpoint_out=None, batch_size=100, epochs=200):
	
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

	# batch_size = 100	
	# epochs = 200
	# epochs = 500
	weight_decay = 1e-3

	print("========================================") 
	print("MODEL HYPER-PARAMETERS") 
	print("BATCH SIZE: {:3d}".format(batch_size)) 
	print("WEIGHT DECAY: {:.4f}".format(weight_decay))
	print("EPOCHS: {:3d}".format(epochs))
	print("========================================") 
	print("== BUILDING MODEL... ==")

	if checkpoint_in is None:
		data_input = Input(batch_shape=(None, num_features, 2))

		# output, model_name = network_200_shared_modrelu_early_abs(data_input, num_classes, weight_decay)

		output, model_name = network_200_shared_modrelu_short_conv_sld(data_input, num_classes, weight_decay)
		# output, model_name = network_200_shared_modrelu_short_conv_sliding_window(data_input, num_classes, weight_decay)
		
		# output, model_name = network_200_shared_modrelu_short_conv_no_modrelu(data_input, num_classes, weight_decay)

		# output, model_name = network_200_shared_modrelu_small_stride(data_input, num_classes, weight_decay)

		# output, model_name = network_200_shared_modrelu_early_abs_short_conv(data_input, num_classes, weight_decay)
		# output, model_name = network_200_shared_modrelu_early_abs_short_conv_no_modrelu(data_input, num_classes, weight_decay)

		checkpoint_out += model_name[6:]
		
		print('----------------------------\n{}\n----------------------------'.format(model_name))

		# output, model_name = network_200_shared_modrelu(data_input, num_classes, weight_decay)
		# checkpoint_out += '-modrelu-100-100'+'-new-'

		densenet = Model(data_input, output)
	else:
		densenet = load_model(checkpoint_in, 
							  custom_objects={'ComplexConv1D':complexnn.ComplexConv1D,
							  				  'GetAbs': utils.GetAbs})

	# Print model architecture
	print(densenet.summary())
	plot_model(densenet, to_file='model_architecture.png')

	# Set optimizer and loss function
	optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	# optimizer = optimizers.SGD(lr=0.001, momentum=0.5, decay=0.0, nesterov=True)

	densenet.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	print("== START TRAINING... ==")
	history = densenet.fit(x=x_train, 
						   y=y_train, 
						   epochs=epochs, 
						   batch_size=batch_size, 
						   # validation_data=(x_test[:num_test//num_aug_test], y_test[:num_test//num_aug_test]), 
						   callbacks=[])



	if checkpoint_out is not None:
		checkpoint_out = checkpoint_out+'-new.h5'
		if epochs > 200:
			checkpoint_out+='-{}-ep.h5'.format(epochs)
		densenet.save(checkpoint_out)



	output_dict = odict(acc=odict(), comp=odict(), loss=odict())

	if num_aug_test!=0:
		logits = densenet.layers[-1].output

		model2 = Model(densenet.input, logits)

		logits_test = model2.predict(x=x_test,
									 batch_size=batch_size,
								 	 verbose=0)		
		logits_test_new = np.zeros((num_test//num_aug_test, num_classes))
		for i in range(num_aug_test):
			# list_x_test.append(x_test[i*num_test:(i+1)*num_test])

			logits_test_new += logits_test[i*num_test//num_aug_test:(i+1)*num_test//num_aug_test]

		num_test = num_test // num_aug_test


		label_pred_llr = logits_test_new.argmax(axis=1)


		y_test = y_test[:num_test]
		# label_pred = probs.argmax(axis=1) 
		label_act = y_test.argmax(axis=1) 
		ind_correct = np.where(label_pred_llr==label_act)[0] 
		ind_wrong = np.where(label_pred_llr!=label_act)[0] 
		assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
		test_acc_llr = 100.*ind_correct.size / num_test

		probs = densenet.predict(x=x_test[:num_test],
								 batch_size=batch_size,
								 verbose=0)
		label_pred = probs.argmax(axis=1)
		ind_correct = np.where(label_pred==label_act)[0] 
		ind_wrong = np.where(label_pred!=label_act)[0] 
		assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
		test_acc = 100.*ind_correct.size / num_test

		# conf_matrix_test = metrics.confusion_matrix(label_act, label_pred_llr)
		# conf_matrix_test = 100*conf_matrix_test/(num_test/num_classes)
		# print('{}'.format(conf_matrix_test))
		# plt.figure()
		# plt.imshow(100*conf_matrix_test/(num_test/num_classes), vmin=0, vmax=100)
		# plt.title('Test confusion matrix')
		# plt.colorbar()

		print("\n========================================") 
		print('Test accuracy (plain): {:.2f}%'.format(test_acc))
		print('Test accuracy with LLR: {:.2f}%'.format(test_acc_llr))
		output_dict['acc']['test'] = test_acc_llr

	else:
		probs = densenet.predict(x=x_test,
								 batch_size=batch_size,
								 verbose=0)
		label_pred = probs.argmax(axis=1)
		label_act = y_test.argmax(axis=1) 
		ind_correct = np.where(label_pred==label_act)[0] 
		ind_wrong = np.where(label_pred!=label_act)[0] 
		assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
		test_acc = 100.*ind_correct.size / num_test

		print("\n========================================") 
		print('Test accuracy (plain): {:.2f}%'.format(test_acc))
		output_dict['acc']['test'] = test_acc

	print('----------------------------\n{}\n----------------------------'.format(model_name))


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