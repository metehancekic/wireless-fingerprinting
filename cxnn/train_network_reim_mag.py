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

from cxnn.complexnn import ComplexDense, ComplexConv1D, utils, Modrelu




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


def train(dict_data, checkpoint_in=None, checkpoint_out=None, architecture='modrelu', fs=20):

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
    epochs = 500
    weight_decay = 1e-4

    n_val = batch_size
    x_val = x_test[:n_val].copy()
    y_val = y_test[:n_val].copy()
    print('\n----------------------------')
    print('Setting validation = First {} samples of test'.format(n_val))
    print('----------------------------\n') 

    print("========================================") 
    print("MODEL HYPER-PARAMETERS") 
    print("BATCH SIZE: {:3d}".format(batch_size)) 
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))
    print("========================================") 
    print("== BUILDING MODEL... ==")


    # Define input shape
    if architecture not in ['magnitude', 'phase', 're', 'im']:
        data_input = Input(batch_shape=(None, num_features, 2))
    else:
        data_input = Input(batch_shape=(None, num_features))

    # Load architecture

    if fs==20:
        if architecture=='reim':
            output, model_name = network_20_reim(data_input, num_classes, weight_decay)
        elif architecture=='reim2x':
            output, model_name = network_20_reim_2x(data_input, num_classes, weight_decay)
        elif architecture=='reimsqrt2x':
            output, model_name = network_20_reim_sqrt2x(data_input, num_classes, weight_decay)
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

            output, model_name = network_20_mag(data_input, num_classes, weight_decay, num_features)
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

            output, model_name = network_20_mag(data_input, num_classes, weight_decay, num_features)
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

            output, model_name = network_20_mag(data_input, num_classes, weight_decay, num_features)
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

            output, model_name = network_20_mag(data_input, num_classes, weight_decay, num_features)
        elif architecture=='modrelu':
            output, model_name = network_20_modrelu_short(data_input, num_classes, weight_decay)
        elif architecture=='crelu':
            raise NotImplementedError

    elif fs==200:
        if architecture=='reim':
            output, model_name = network_200_reim(data_input, num_classes, weight_decay)
        elif architecture=='reim2x':
            output, model_name = network_200_reim_2x(data_input, num_classes, weight_decay)
        elif architecture=='reimsqrt2x':
            output, model_name = network_200_reim_sqrt2x(data_input, num_classes, weight_decay)
        elif architecture=='modrelu':
            output, model_name = network_200_modrelu_short(data_input, num_classes, weight_decay)
        else:
            raise NotImplementedError
        model_name += '_fs_200.h5'

    if checkpoint_in is None:
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
    # optimizer = optimizers.SGD(lr=0.001, momentum=0.5, decay=0.0, nesterov=True)

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

    return output_dict, model_name, summary