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

# # import complexnn
from cxnn.complexnn import ComplexDense, ComplexConv1D, utils, Modrelu

def RotateComplex(x, theta):

	real = utils.GetReal()(x)
	imag = utils.GetImag()(x)

	rotated_real = math.cos(theta) * real - math.sin(theta) * imag
	rotated_imag = math.sin(theta) * real + math.cos(theta) * imag

	merged = Concatenate()([rotated_real, rotated_imag])

	return merged

def output_of_lambda(input_shape):
	return input_shape

output_shape=output_of_lambda


def RotateComplex(x, theta):

	real = utils.GetReal()(x)
	imag = utils.GetImag()(x)

	rotated_real = math.cos(theta) * real - math.sin(theta) * imag
	rotated_imag = math.sin(theta) * real + math.cos(theta) * imag

	merged = Concatenate()([rotated_real, rotated_imag])

	return merged


def net_address_20(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"
	
	########################
	filters = 200
	k_size = 100
	strides = 50
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
	filters = 128
	k_size = 10
	strides = 2
	o = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', 
			   kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	# ########################


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

	########################
	d_rate = 0.8
	o = Dropout(d_rate)(o)
	model_name = model_name + "-d({:.2f})".format(d_rate)
	########################

	x = Dense(classes_num,
			  activation='softmax',
			  kernel_initializer="he_normal")(o)

	return x, model_name

def net_preamble_20(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 40
	strides = 20
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	filters = 128
	k_size = 5
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################

	########################
	o = GlobalAveragePooling1D()(o)
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


	return x, pre_softmax, model_name

def net_preamble_20_modrelu(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=True,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"


	##################################
	########################
	filters = 200
	k_size = 40
	strides = 20
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################
	# print(K.int_shape(o))

	########################
	# o = Lambda(modrelu, arguments = {"b" : 5 }, output_shape=output_of_lambda)(o)
	o = Modrelu(name="ModRelu")(o)
	model_name = model_name + "-ModReLU"
	# print(K.int_shape(o))
	########################


	##################################
	########################
	filters = 128
	k_size = 5
	strides = 1

	# convArgs['use_bias']=True
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, name="ComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = Modrelu(name="ModRelu2")(o)
	model_name = model_name + "-ModReLU"
	########################


	########################
	o = utils.GetAbs(name="Abs")(o)
	model_name = model_name + "-Abs"
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

	model_name = model_name + "Last"

	return x, pre_softmax, model_name

def net_preamble_20_modrelu_100(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"


	##################################
	########################
	filters = 100
	k_size = 40
	strides = 20
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################
	# print(K.int_shape(o))

	########################
	# o = Lambda(modrelu, arguments = {"b" : 5 }, output_shape=output_of_lambda)(o)
	o = Modrelu(name="ModRelu")(o)
	model_name = model_name + "-ModReLU"
	# print(K.int_shape(o))
	########################


	##################################
	########################
	filters = 100
	k_size = 5
	strides = 1

	# convArgs['use_bias']=True
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, name="ComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = Modrelu(name="ModRelu2")(o)
	model_name = model_name + "-ModReLU"
	########################


	########################
	o = utils.GetAbs(name="Abs")(o)
	model_name = model_name + "-Abs"
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

	model_name = model_name + "Last"

	return x, pre_softmax, model_name

def net_preamble_20_no_crelu(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 40
	strides = 20
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	filters = 64
	k_size = 5
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	#########################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
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

def net_preamble_20_real(data_input1, data_input2, data_input3, data_input4, classes_num=10, weight_decay=1e-4):
	
	
	model_name = "MODEL:"


	##################################
	########################
	filters = 200
	k_size = 40
	strides = 20
	shared_conv = Conv1D(filters=filters, 
			   kernel_size=[k_size], 
			   strides=strides, 
			   padding='valid', 
			   activation='relu', 
			   kernel_regularizer=l2(weight_decay))
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
	filters = 128
	k_size = 5
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



def net_preamble_20_rotated(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"


	##################################
	########################
	filters = 200
	k_size = 40
	strides = 20
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################


	rotated0 = Lambda(RotateComplex, arguments = {"theta": 0} ,output_shape=output_of_lambda)(o)
	rotated1 = Lambda(RotateComplex, arguments = {"theta": math.pi/8} ,output_shape=output_of_lambda)(o)
	rotated2 = Lambda(RotateComplex, arguments = {"theta": math.pi/4} ,output_shape=output_of_lambda)(o)
	rotated3 = Lambda(RotateComplex, arguments = {"theta": 3*math.pi/8} ,output_shape=output_of_lambda)(o)

	rotated4 = Lambda(RotateComplex, arguments = {"theta": math.pi/2} ,output_shape=output_of_lambda)(o)
	rotated5 = Lambda(RotateComplex, arguments = {"theta": 5*math.pi/8} ,output_shape=output_of_lambda)(o)
	rotated6 = Lambda(RotateComplex, arguments = {"theta": 3*math.pi/4} ,output_shape=output_of_lambda)(o)
	rotated7 = Lambda(RotateComplex, arguments = {"theta": 7*math.pi/8} ,output_shape=output_of_lambda)(o)

	rotated8 = Lambda(RotateComplex, arguments = {"theta": math.pi} ,output_shape=output_of_lambda)(o)
	rotated9 = Lambda(RotateComplex, arguments = {"theta": 9.*math.pi/8} ,output_shape=output_of_lambda)(o)
	rotated10 = Lambda(RotateComplex, arguments = {"theta": 5.*math.pi/4} ,output_shape=output_of_lambda)(o)
	rotated11 = Lambda(RotateComplex, arguments = {"theta": 11*math.pi/8} ,output_shape=output_of_lambda)(o)

	rotated12 = Lambda(RotateComplex, arguments = {"theta": 3*math.pi/2} ,output_shape=output_of_lambda)(o)
	rotated13 = Lambda(RotateComplex, arguments = {"theta": 13*math.pi/8} ,output_shape=output_of_lambda)(o)
	rotated14 = Lambda(RotateComplex, arguments = {"theta": 7*math.pi/4} ,output_shape=output_of_lambda)(o)
	rotated15 = Lambda(RotateComplex, arguments = {"theta": 15*math.pi/8} ,output_shape=output_of_lambda)(o)

	o = Maximum()([rotated0,rotated1,rotated2,rotated3,rotated4,rotated5,rotated6,rotated7,rotated8,rotated9,rotated10,rotated11,rotated12,rotated13,rotated14,rotated15])

	# print(K.int_shape(o))

	o = Lambda(relu_manual,output_shape=output_of_lambda)(o)

	# print(K.int_shape(o))

	###################################

	########################
	o = utils.GetAbs()(o)
	model_name = model_name + "-Abs"
	########################


	########################
	filters = 128
	k_size = 5
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

	x = Dense(classes_num,
			  activation='softmax',
			  kernel_initializer="he_normal")(o)

	return x, model_name



def net_preamble_20_modrelu_best(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"


	##################################
	########################
	filters = 200
	k_size = 100
	strides = 10
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################
	# print(K.int_shape(o))

	########################
	# o = Lambda(modrelu, arguments = {"b" : 5 }, output_shape=output_of_lambda)(o)
	o = Modrelu(name="ModRelu")(o)
	model_name = model_name + "-ModReLU"
	# print(K.int_shape(o))
	########################


	##################################
	########################
	filters = 128
	k_size = 5
	strides = 1

	# convArgs['use_bias']=True
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, name="ComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	########################
	o = Modrelu(name="ModRelu2")(o)
	model_name = model_name + "-ModReLU"
	########################


	########################
	o = utils.GetAbs(name="Abs")(o)
	model_name = model_name + "-Abs"
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


	return x, pre_softmax, model_name

def net_preamble_20_tall(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"


	##################################
	########################
	filters = 128
	k_size = 16
	strides = 8
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, name="ComplexConv1", **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################
	# print(K.int_shape(o))

	# o = Lambda(modrelu, arguments = {"b" : 5 }, output_shape=output_of_lambda)(o)
	o = Modrelu(name="ModRelu1")(o)
	model_name = model_name + "-ModReLU"
	# print(K.int_shape(o))


	##################################
	########################
	filters = 128
	k_size = 3
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, name="ComplexConv2", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################

	o = Modrelu(name="ModRelu2")(o)
	model_name = model_name + "-ModReLU"

	########################
	filters = 128
	k_size = 3
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, name="ComplexConv3", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################


	o = Modrelu(name="ModRelu3")(o)
	model_name = model_name + "-ModReLU"

	########################
	filters = 128
	k_size = 3
	strides = 1
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', name="ComplexConv4", **convArgs)(o)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################


	########################
	o = utils.GetAbs(name="Abs")(o)
	model_name = model_name + "-Abs"
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


	return x, pre_softmax, model_name

def net_preamble_20_rot16_real(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"


	##################################
	########################
	filters = 200
	k_size = 40
	strides = 20
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation=None, **convArgs)(data_input)
	model_name = model_name + "-" + str(filters) + "C" + str(k_size) + "x" + str(strides)
	########################


	rotated0 = Lambda(RotateComplex, arguments = {"theta": 0} ,output_shape=output_of_lambda)(o)
	rotated1 = Lambda(RotateComplex, arguments = {"theta": math.pi/8} ,output_shape=output_of_lambda)(o)
	rotated2 = Lambda(RotateComplex, arguments = {"theta": math.pi/4} ,output_shape=output_of_lambda)(o)
	rotated3 = Lambda(RotateComplex, arguments = {"theta": 3*math.pi/8} ,output_shape=output_of_lambda)(o)

	rotated4 = Lambda(RotateComplex, arguments = {"theta": math.pi/2} ,output_shape=output_of_lambda)(o)
	rotated5 = Lambda(RotateComplex, arguments = {"theta": 5*math.pi/8} ,output_shape=output_of_lambda)(o)
	rotated6 = Lambda(RotateComplex, arguments = {"theta": 3*math.pi/4} ,output_shape=output_of_lambda)(o)
	rotated7 = Lambda(RotateComplex, arguments = {"theta": 7*math.pi/8} ,output_shape=output_of_lambda)(o)

	rotated8 = Lambda(RotateComplex, arguments = {"theta": math.pi} ,output_shape=output_of_lambda)(o)
	rotated9 = Lambda(RotateComplex, arguments = {"theta": 9.*math.pi/8} ,output_shape=output_of_lambda)(o)
	rotated10 = Lambda(RotateComplex, arguments = {"theta": 5.*math.pi/4} ,output_shape=output_of_lambda)(o)
	rotated11 = Lambda(RotateComplex, arguments = {"theta": 11*math.pi/8} ,output_shape=output_of_lambda)(o)

	rotated12 = Lambda(RotateComplex, arguments = {"theta": 3*math.pi/2} ,output_shape=output_of_lambda)(o)
	rotated13 = Lambda(RotateComplex, arguments = {"theta": 13*math.pi/8} ,output_shape=output_of_lambda)(o)
	rotated14 = Lambda(RotateComplex, arguments = {"theta": 7*math.pi/4} ,output_shape=output_of_lambda)(o)
	rotated15 = Lambda(RotateComplex, arguments = {"theta": 15*math.pi/8} ,output_shape=output_of_lambda)(o)


	rotated0 = utils.GetReal()(rotated0)
	rotated1 = utils.GetReal()(rotated1)
	rotated2 = utils.GetReal()(rotated2)
	rotated3 = utils.GetReal()(rotated3)
	rotated4 = utils.GetReal()(rotated4)
	rotated5 = utils.GetReal()(rotated5)
	rotated6 = utils.GetReal()(rotated6)
	rotated7 = utils.GetReal()(rotated7)
	rotated8 = utils.GetReal()(rotated8)
	rotated9 = utils.GetReal()(rotated9)
	rotated10 = utils.GetReal()(rotated10)
	rotated11 = utils.GetReal()(rotated11)
	rotated12 = utils.GetReal()(rotated12)
	rotated13 = utils.GetReal()(rotated13)
	rotated14 = utils.GetReal()(rotated14)
	rotated15 = utils.GetReal()(rotated15)

	o = Maximum()([rotated0,rotated1,rotated2,rotated3,rotated4,rotated5,rotated6,rotated7,rotated8,rotated9,rotated10,rotated11,rotated12,rotated13,rotated14,rotated15])

	# print(K.int_shape(o))

	o = Lambda(relu_manual,output_shape=output_of_lambda)(o)

	# print(K.int_shape(o))

	model_name = model_name + "-rot16--0,pi/8,...,15*pi/8--real-max"
	###################################

	# ########################
	# o = utils.GetAbs()(o)
	# model_name = model_name + "-Abs"
	# ########################


	########################
	filters = 128
	k_size = 5
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

	x = Dense(classes_num,
			  activation='softmax',
			  kernel_initializer="he_normal")(o)

	return x, model_name

def net_preamble_50(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"

	########################
	filters = 200
	k_size = 100
	strides = 50
	o = ComplexConv1D(filters=filters, 
					  kernel_size=[k_size], 
					  strides=strides, 
					  padding='valid', 
					  activation='relu', **convArgs)(o)
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
			  activation=None,
			  kernel_initializer="he_normal",
			  kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	# ########################
	# d_rate = 0.5
	# o = Dropout(d_rate)(o)
	# model_name = model_name + "-d({:.2f})".format(d_rate)
	# ########################

	x = Dense(classes_num,
			  activation='softmax',
			  kernel_initializer="he_normal")(o)

	return x , model_name

def net_preamble_100(data_input, classes_num=10, weight_decay=1e-4):
	
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
					  activation='relu', **convArgs)(o)
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

	########################
	d_rate = 0.5
	o = Dropout(d_rate)(o)
	model_name = model_name + "-d({:.2f})".format(d_rate)
	########################

	x = Dense(classes_num,
			  activation='softmax',
			  kernel_initializer="he_normal")(o)

	return x , model_name

