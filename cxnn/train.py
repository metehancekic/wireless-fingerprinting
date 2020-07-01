
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
from cxnn.models import network_20_modrelu_short, network_20_reim, network_20_reim_2x, network_20_reim_sqrt2x, network_20_mag, network_200_modrelu_short, network_200_reim, network_200_reim_2x, network_200_reim_sqrt2x, network_200_mag, network_200_modrelu_short_shared


def train_20(dict_data, num_aug_test=1, checkpoint_in=None, checkpoint_out=None, architecture='modrelu', epochs=200, n_val=True):

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
    weight_decay = 1e-4

    if n_val is True:
        n_val = num_test
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
    if architecture == 'reim':
        output, model_name = network_20_reim(data_input, num_classes, weight_decay)
    elif architecture == 'reim2x':
        output, model_name = network_20_reim_2x(data_input, num_classes, weight_decay)
    elif architecture == 'reimsqrt2x':
        output, model_name = network_20_reim_sqrt2x(data_input, num_classes, weight_decay)
    elif architecture == 'magnitude':
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
    elif architecture == 'phase':
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
    elif architecture == 're':
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
    elif architecture == 'im':
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
    elif architecture == 'modrelu':
        output, model_name = network_20_modrelu_short(data_input, num_classes, weight_decay)
    elif architecture == 'crelu':
        raise NotImplementedError

    if checkpoint_in is None:
        densenet = Model(data_input, output)
    else:
        densenet = load_model(checkpoint_in,
                              custom_objects={'ComplexConv1D': complexnn.ComplexConv1D,
                                              'GetAbs': utils.GetAbs})

    # Print model architecture
    print(densenet.summary())
    # plot_model(densenet, to_file='model_architecture.png')

    # Set optimizer and loss function

    optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # learning_rate = 0.000001
    # decay_rate = learning_rate / epochs
    # optimizer = optimizers.SGD(lr=learning_rate, momentum=0.5, decay=decay_rate, nesterov=False)

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

    output_dict = odict(acc=odict(), comp=odict(), loss=odict())

    if num_aug_test != 0:
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
        ind_correct = np.where(label_pred_llr == label_act)[0]
        ind_wrong = np.where(label_pred_llr != label_act)[0]
        assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
        test_acc_llr = 100.*ind_correct.size / num_test

        probs = densenet.predict(x=x_test[:num_test],
                                 batch_size=batch_size,
                                 verbose=0)
        label_pred = probs.argmax(axis=1)
        ind_correct = np.where(label_pred == label_act)[0]
        ind_wrong = np.where(label_pred != label_act)[0]
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
        ind_correct = np.where(label_pred == label_act)[0]
        ind_wrong = np.where(label_pred != label_act)[0]
        assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
        test_acc = 100.*ind_correct.size / num_test

        print("\n========================================")
        print('Test accuracy (plain): {:.2f}%'.format(test_acc))
        output_dict['acc']['test'] = test_acc

    # conf_matrix_test = metrics.confusion_matrix(label_act, label_pred)
    # conf_matrix_test = 100*conf_matrix_test/(num_test/num_classes)
    # print('{}'.format(conf_matrix_test))
    # plt.figure()
    # plt.imshow(100*conf_matrix_test/(num_test/num_classes), vmin=0, vmax=100)
    # plt.title('Test confusion matrix')
    # plt.colorbar()

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


def train_200(dict_data, num_aug_test=1, checkpoint_in=None, checkpoint_out=None, architecture='modrelu', epochs=200, n_val=True):

    x_train = dict_data['x_train']
    y_train = dict_data['y_train']

    x_val = dict_data['x_validation']
    y_val = dict_data['y_validation']

    x_test = dict_data['x_test']
    y_test = dict_data['y_test']

    num_classes = dict_data['num_classes']
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_features = x_train.shape[1]

    print('Training data size: {}'.format(x_train.shape))
    print('Test data size: {}'.format(x_test.shape))

    batch_size = 100
    weight_decay = 1e-4

    # if num_aug_test == 0:
    #   n_val = num_test
    # else:
    #   n_val = num_test // num_aug_test
    if n_val is True:
        n_val = num_test
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
    if architecture == 'reim':
        output, model_name = network_200_reim(data_input, num_classes, weight_decay)
    elif architecture == 'reim2x':
        output, model_name = network_200_reim_2x(data_input, num_classes, weight_decay)
    elif architecture == 'reimsqrt2x':
        output, model_name = network_200_reim_sqrt2x(data_input, num_classes, weight_decay)
    elif architecture == 'magnitude':
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

        output, model_name = network_200_mag(data_input, num_classes, weight_decay, num_features)
    elif architecture == 'phase':
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

        output, model_name = network_200_mag(data_input, num_classes, weight_decay, num_features)
    elif architecture == 're':
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

        output, model_name = network_200_mag(data_input, num_classes, weight_decay, num_features)
    elif architecture == 'im':
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

        output, model_name = network_200_mag(data_input, num_classes, weight_decay, num_features)
    elif architecture == 'modrelu':
        # output, model_name = network_200_modrelu_short(data_input, num_classes, weight_decay)
        output, model_name = network_200_modrelu_short_shared(data_input, num_classes, weight_decay)
    elif architecture == 'crelu':
        raise NotImplementedError

    if checkpoint_in is None:
        densenet = Model(data_input, output)
    else:
        densenet = load_model(checkpoint_in,
                              custom_objects={'ComplexConv1D': ComplexConv1D,
                                              'GetAbs': utils.GetAbs})

    # Print model architecture
    print(densenet.summary())
    # plot_model(densenet, to_file='model_architecture.png')

    # optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # learning_rate = 0.01
    # decay_rate = learning_rate / epochs
    # optimizer = optimizers.SGD(lr=learning_rate, momentum=0.5, decay=decay_rate, nesterov=False)

    densenet.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    if checkpoint_out is not None:
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_out+'.h5', verbose=0, save_best_only=True, monitor='loss', mode='auto', period=10)
        checkpoint_to_test = checkpoint_out+'.h5'

    print("== START TRAINING... ==")
    history = densenet.fit(x=x_train,
                           y=y_train,
                           epochs=epochs,
                           batch_size=batch_size,
                           validation_data=(x_val, y_val),
                           callbacks=[checkpointer])

    check_last_epoch = True
    if check_last_epoch:
        probs = densenet.predict(x=x_train,
                                 batch_size=batch_size,
                                 verbose=0)
        label_pred = probs.argmax(axis=1)
        label_act = y_train.argmax(axis=1)
        ind_correct = np.where(label_pred == label_act)[0]
        ind_wrong = np.where(label_pred != label_act)[0]
        assert (num_train == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
        train_acc = 100.*ind_correct.size / num_train
        print("\n========================================")
        print('Train accuracy (Last Epoch): {:.2f}%'.format(train_acc))

        probs = densenet.predict(x=x_test,
                                 batch_size=batch_size,
                                 verbose=0)
        label_pred = probs.argmax(axis=1)
        label_act = y_test.argmax(axis=1)
        ind_correct = np.where(label_pred == label_act)[0]
        ind_wrong = np.where(label_pred != label_act)[0]
        assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
        test_acc = 100.*ind_correct.size / num_test

        print("\n========================================")
        print('Test accuracy (Last_epoch): {:.2f}%'.format(test_acc))

    del densenet
    # K.clear_session()

    densenet = load_model(checkpoint_to_test,
                          custom_objects={'ComplexConv1D': ComplexConv1D,
                                          'GetAbs': utils.GetAbs,
                                          'Modrelu': Modrelu})

    # if checkpoint_out is not None:
    #   checkpoint_out = checkpoint_out+'-new.h5'
    #   densenet.save(checkpoint_out)

    output_dict = odict(acc=odict(), comp=odict(), loss=odict())

    if num_aug_test != 0:
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
        ind_correct = np.where(label_pred_llr == label_act)[0]
        ind_wrong = np.where(label_pred_llr != label_act)[0]
        assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
        test_acc_llr = 100.*ind_correct.size / num_test

        probs = densenet.predict(x=x_test[:num_test],
                                 batch_size=batch_size,
                                 verbose=0)
        label_pred = probs.argmax(axis=1)
        ind_correct = np.where(label_pred == label_act)[0]
        ind_wrong = np.where(label_pred != label_act)[0]
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
        output_dict['acc']['test'] = test_acc
        output_dict['acc']['test_llr'] = test_acc_llr

    else:
        probs = densenet.predict(x=x_test,
                                 batch_size=batch_size,
                                 verbose=0)
        label_pred = probs.argmax(axis=1)
        label_act = y_test.argmax(axis=1)
        ind_correct = np.where(label_pred == label_act)[0]
        ind_wrong = np.where(label_pred != label_act)[0]
        assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
        test_acc = 100.*ind_correct.size / num_test

        print("\n========================================")
        print('Test accuracy (plain): {:.2f}%'.format(test_acc))
        output_dict['acc']['test'] = test_acc

    probs = densenet.predict(x=x_train,
                             batch_size=batch_size,
                             verbose=0)
    label_pred = probs.argmax(axis=1)
    label_act = y_train.argmax(axis=1)
    ind_correct = np.where(label_pred == label_act)[0]
    ind_wrong = np.where(label_pred != label_act)[0]
    assert (num_train == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
    train_acc = 100.*ind_correct.size / num_train

    print("\n========================================")
    print('Train accuracy (plain): {:.2f}%'.format(train_acc))
    output_dict['acc']['train'] = train_acc

    # conf_matrix_test = metrics.confusion_matrix(label_act, label_pred)
    # conf_matrix_test = 100*conf_matrix_test/(num_test/num_classes)
    # print('{}'.format(conf_matrix_test))
    # plt.figure()
    # plt.imshow(100*conf_matrix_test/(num_test/num_classes), vmin=0, vmax=100)
    # plt.title('Test confusion matrix')
    # plt.colorbar()

    # output_dict['acc']['val'] = 100.*history.history['val_acc'][-1]
    # output_dict['acc']['train'] = 100.*history.history['acc'][-1]

    # output_dict['loss']['val'] = history.history['val_loss'][-1]
    # output_dict['loss']['train'] = history.history['loss'][-1]

    stringlist = []
    densenet.summary(print_fn=lambda x: stringlist.append(x))
    summary = '\n' + \
        'Batch size: {:3d}\n'.format(batch_size) + \
        'Weight decay: {:.4f}\n'.format(weight_decay) + \
        'Epochs: {:3d}\n'.format(epochs) + \
        'Optimizer:' + str(densenet.optimizer) + '\n'
    summary += '\n'.join(stringlist)

    return output_dict, model_name, summary
