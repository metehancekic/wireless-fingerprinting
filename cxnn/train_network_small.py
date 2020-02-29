from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from collections import OrderedDict as odict

import keras
from keras import backend as K
from keras import optimizers, regularizers, losses
from keras.layers import Dense, Input, Activation, Conv1D, Dropout, GlobalAveragePooling1D, Concatenate, Lambda, merge, Maximum, Add
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.engine.topology import Layer

from sklearn import metrics

# import complexnn
from .complexnn import ComplexDense, ComplexConv1D, utils
import ipdb

import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rc('text', usetex=True)
# mpl.rc('text.latex', preamble = r'\usepackage{amsmath}, \usepackage{sfmath}, \usepackage{amssymb}, \usepackage{bm}, \DeclareMathOperator*{\supp}{supp}, \DeclareMathOperator*{\proj}{\mathcal{P}_K}, \newcommand{\be}{{\bm e}}, \newcommand{\bx}{{\bm x}}, \DeclareMathOperator*{\support}{\mathcal{S}_K}')

def set_keras_backend(backend):
	if K.backend() != backend:
		os.environ['KERAS_BACKEND'] = backend
		reload(K)
		assert K.backend() == backendr

set_keras_backend("theano")
np.set_printoptions(precision=2)

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

def output_of_lambda(input_shape):
	return input_shape

class Modrelu(Layer):

	def __init__(self, **kwargs):
		super(Modrelu, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self._b = self.add_weight(name='b', 
									shape=(input_shape[-1]//2,),
									initializer='uniform',
									trainable=True)
		super(Modrelu, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		
		real = utils.GetReal()(x)
		imag = utils.GetImag()(x)

		abs1 = K.relu( utils.GetAbs()(x) )
		abs2 = K.relu( utils.GetAbs()(x) - self._b )

		
		real = real * abs2 / (abs1+0.0000001)
		imag = imag * abs2 / (abs1+0.0000001)

		merged = Concatenate()([real, imag])

		return merged

	def compute_output_shape(self, input_shape):
		return input_shape

def net_preamble_20_modrelu(data_input, classes_num=10, weight_decay=1e-4):
	
	convArgs = dict(use_bias=False,
					kernel_regularizer=l2(weight_decay),
					spectral_parametrization=False,
					kernel_initializer='complex_independent')
	model_name = "MODEL:"


	##################################
	########################
	filters = 200
	k_size = 20
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
	filters = 200
	k_size = 10
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

	########################
	d_rate = 0.8
	o = Dropout(d_rate)(o)
	model_name = model_name + "-d({:.2f})".format(d_rate)
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


	return x,  model_name



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

def network_20_wifi2_new_short_conv(data_input, classes_num=10, weight_decay = 1e-4):
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
	k_size = 10
	strides = 5
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
	strides = 2
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

def network_20(data_input, classes_num=10, weight_decay = 1e-4):
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
			  activation='relu',
			  kernel_initializer="he_normal", 
			  kernel_regularizer=l2(weight_decay))(o)
	model_name = model_name + "-" + str(neuron_num) + "D"
	########################

	x = Dense(classes_num,
			 activation='softmax',
			 kernel_initializer="he_normal")(o)

	return x, model_name

def train(dict_data, checkpoint_in=None, checkpoint_out=None):

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
	epochs = 100
	weight_decay = 1e-3

	print("========================================") 
	print("MODEL HYPER-PARAMETERS") 
	print("BATCH SIZE: {:3d}".format(batch_size)) 
	print("WEIGHT DECAY: {:.4f}".format(weight_decay))
	print("EPOCHS: {:3d}".format(epochs))
	print("========================================") 
	print("== BUILDING MODEL... ==")

	if checkpoint_in is None:
		data_input = Input(batch_shape=(batch_size, num_features, 2))
		# output, model_name = network_20_new(data_input, num_classes, weight_decay)
		# output, model_name = network_20_wifi2_new_short_conv(data_input, num_classes, weight_decay)
		output, model_name = net_preamble_20_modrelu(data_input, num_classes, weight_decay)
		# output, model_name = network_200_new(data_input, num_classes, weight_decay)
		densenet = Model(data_input, output)
	else:
		densenet = load_model(checkpoint_in, 
							  custom_objects={'ComplexConv1D':complexnn.ComplexConv1D,
							  				  'GetAbs': utils.GetAbs})

	# Print model architecture
	print(densenet.summary())

	# Set optimizer and loss function
	optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	densenet.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	print("== START TRAINING... ==")
	history = densenet.fit(x=x_train, 
						   y=y_train, 
						   epochs=epochs, 
						   batch_size=batch_size, 
						   validation_data=(x_test, y_test), 
						   callbacks=[])

	if checkpoint_out is not None:
		checkpoint_out = checkpoint_out+'-new.h5'
		densenet.save(checkpoint_out)

	probs = densenet.predict(x=x_test, batch_size=batch_size, verbose=0)
	label_pred = probs.argmax(axis=1) 
	label_act = y_test.argmax(axis=1) 
	ind_correct = np.where(label_pred==label_act)[0] 
	ind_wrong = np.where(label_pred!=label_act)[0] 
	assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
	test_acc = 100.*ind_correct.size / num_test

	acc_class = np.zeros([num_classes])
	for class_idx in range(num_classes):
		idx_inclass = np.where(label_act==class_idx)[0]
		ind_correct = np.where(label_pred[idx_inclass]==label_act[idx_inclass])[0] 
		acc_class[class_idx] = 100*ind_correct.size / idx_inclass.size

	# from IPython import embed; embed()
	# ipdb.set_trace()

	conf_matrix_test = metrics.confusion_matrix(label_act, label_pred)
	# print('{}'.format(conf_matrix_test))
	# plt.figure()
	# plt.imshow(100*conf_matrix_test/(num_test/num_classes), vmin=0, vmax=100)
	# plt.title('Test confusion matrix')
	# plt.colorbar()


	probs_train = densenet.predict(x=x_train, batch_size=batch_size, verbose=0)
	label_pred_train = probs_train.argmax(axis=1) 
	label_act_train = y_train.argmax(axis=1) 

	conf_matrix_train = metrics.confusion_matrix(label_act_train, label_pred_train)
	# print('{}'.format(conf_matrix_train/(num_train/num_classes)))
	# plt.figure()
	# plt.imshow(100*conf_matrix_train/(num_train/num_classes), vmin=0, vmax=100)
	# plt.title('Train confusion matrix')
	# plt.colorbar()

	print("\n========================================") 
	print('Test accuracy: {:.2f}%'.format(test_acc))
	print('Accuracy per class: {}%'.format(acc_class))

	output_dict = odict(acc=odict(), comp=odict(), loss=odict())

	output_dict['acc']['test'] = test_acc
	output_dict['acc']['val'] = 100.*history.history['val_acc'][-1]
	output_dict['acc']['train'] = 100.*history.history['acc'][-1]

	output_dict['loss']['val'] = history.history['val_loss'][-1]
	output_dict['loss']['train'] = history.history['loss'][-1]

	stringlist = []
	densenet.summary(print_fn=lambda x: stringlist.append(x))
	summary = '\n' + \
			'Batch size: {:3d}\n'.format(batch_size) + \
			'Weight decay: {:.4f}\n'.format(weight_decay) + \
			'Epochs: {:3d}\n'.format(epochs) + \
			'Optimizer:' + str(densenet.optimizer) + '\n'
	summary += '\n'.join(stringlist)

	# plt.show()

	return output_dict, model_name, summary