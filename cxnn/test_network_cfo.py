from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from collections import OrderedDict as odict
from timeit import default_timer as timer
import ipdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy.linalg as LA
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
import math

# import complexnn
from .complexnn import ComplexDense, ComplexConv1D, utils
from .cxnn.train_network  import set_keras_backend

from .models_adsb import Modrelu

# mpl.rc('text', usetex=True)
# mpl.rc('text.latex', preamble = r'\usepackage{amsmath}, \usepackage{sfmath}, \usepackage{amssymb}, \usepackage{bm}, \DeclareMathOperator*{\supp}{supp}, \DeclareMathOperator*{\proj}{\mathcal{P}_K}, \newcommand{\be}{{\bm e}}, \newcommand{\bx}{{\bm x}}, \DeclareMathOperator*{\support}{\mathcal{S}_K}')

from sklearn import metrics

set_keras_backend("theano")
np.set_printoptions(precision=2)



def rotate_data(x, theta):

	real = x[...,:1]
	imag = x[...,1:]

	rotated_real = math.cos(theta) * real - math.sin(theta) * imag
	rotated_imag = math.sin(theta) * real + math.cos(theta) * imag

	rotated = np.concatenate((rotated_real, rotated_imag), axis= -1)

	return rotated

def rotate_randomly_data(x):
	real = x[...,:1]
	imag = x[...,1:]
	rotated_real = real.copy().astype(np.float32)
	rotated_imag = imag.copy().astype(np.float32)
	theta = np.random.uniform(0, 2*math.pi, x.shape[0]) 
	# print(theta)
	for i in range(x.shape[0]): 
		rotated_real[i] = math.cos(theta[i]) * real[i] - math.sin(theta[i]) * imag[i]
		rotated_imag[i] = math.sin(theta[i]) * real[i] + math.cos(theta[i]) * imag[i]
	rotated = np.concatenate((rotated_real, rotated_imag), axis= -1)
	return rotated

def add_freq_offset_rand(x):
	real = x[...,:1]
	imag = x[...,1:]
	rotated_real = real.copy().astype(np.float32)
	rotated_imag = imag.copy().astype(np.float32)
	cfo = 20e-6 * 2.4e9 # 20 ppm of 2.4 GHz
	fs = 20e6
	rv = np.random.RandomState(seed=0)
	theta = rv.uniform(-2*math.pi*cfo/fs, 2*math.pi*cfo/fs, x.shape[0]) 
	# theta = np.random.uniform(-2*math.pi/2000, 2*math.pi/2000, x.shape[0]) 
	# print(theta)
	N = real.shape[1]

	for i in tqdm(range(x.shape[0])):
		for j in range(x.shape[1]): 
			rotated_real[i,j] = math.cos(theta[i]*j) * real[i,j] - math.sin(theta[i]*j) * imag[i,j]
			rotated_imag[i,j] = math.sin(theta[i]*j) * real[i,j] + math.cos(theta[i]*j) * imag[i,j]
	rotated = np.concatenate((rotated_real, rotated_imag), axis= -1)
	return rotated

def add_freq_offset_fixed(x, carrier_f):
	
	complex_data = x[...,0].copy() + 1j* x[...,1].copy() # (N, T)
	N = complex_data.shape[0]
	T = complex_data.shape[1]

	cfo = -10e-6 * carrier_f # 20 ppm of 2.4 GHz
	fs = 20e6
	rv = np.random.RandomState(seed=0)

	# theta = rv.uniform(-2*math.pi*cfo/fs, 2*math.pi*cfo/fs) 
	theta = (-2*math.pi*cfo/fs).reshape(-1, 1)

	# theta += -2*np.pi*N*cfo/fs*2

	exp_offset = np.exp(1j*theta.dot(np.arange(T).reshape(1,-1)))  # (1, T)
	# print(exp_offset.shape)
	# exp_offset = np.repeat(exp_offset, repeats=[complex_data.shape[0], 1, 1])
	# print(exp_offset.shape)
	
	complex_data *= exp_offset

	# rotated_real = complex_data.real
	# rotated_imag = complex_data.imag

	rotated = np.concatenate((complex_data.real[..., None], complex_data.imag[..., None]), axis= -1)
	return rotated
def mu_cov_calc(logits, y):
	num_classes = y.shape[1]
	labels = np.argmax(y, axis=1)
	mu = np.zeros((num_classes, num_classes))
	rho = np.zeros((num_classes, num_classes, num_classes))
	cov = np.zeros((num_classes, num_classes, num_classes))

	for class_idx in range(num_classes):
		idx_inclass = np.where(labels==class_idx)[0]
		mu[class_idx, :] = logits[idx_inclass].mean(axis=0)

	for class_idx in range(num_classes):
		idx_inclass = np.where(labels==class_idx)[0]
		X = logits[idx_inclass, :] # shape [num_test, num_classes]

		X_bar = X - X.mean(axis=0)
		X_norm = StandardScaler().fit_transform(X)
		cov[class_idx] = np.matmul(X_bar.T, X_bar) / idx_inclass.size
		rho[class_idx] = np.matmul(X_norm.T, X_norm) / idx_inclass.size

	return mu, cov, rho

def Z_calc(logits, mu, cov):
	num_classes = mu.shape[0]
	num_images = logits.shape[0]
	Z = np.zeros([num_images, num_classes])

	for i in trange(num_classes):
		m = mu[i].reshape([-1, 1])
		C = cov[i]

		C_inv = np.linalg.inv(C)

		# Cm, residuals, rank, s = np.linalg.lstsq(C, m, rcond=-1)
		# Cm = np.linalg.solve(C, m)
		Cm = C_inv.dot(m)
		t2 = 0.5* m.T.dot(Cm)

		sgn, t4 = np.linalg.slogdet(C)
		t4 *= 0.5


		for j in trange(num_images):
			y = logits[j].reshape([-1, 1])

			# Cy, residuals, rank, s = np.linalg.lstsq(C, y, rcond=-1)
			# Cy = np.linalg.solve(C, y)
			Cy = C_inv.dot(y)

			Z[j, i] = m.T.dot(Cy) - t2 - 0.5*y.T.dot(Cy) - t4
	return Z

def Z_calc_smallest_k(logits, mu, cov, K):
	num_classes = mu.shape[0]
	num_images = logits.shape[0]
	Z = np.zeros([num_images, num_classes])
	evals_cov = np.zeros([num_classes, num_classes])
	snr_cov = np.zeros([num_classes, num_classes])

	for i in trange(num_classes):
		m = mu[i].reshape([-1, 1])
		evals, evecs = LA.eigh(cov[i])

		evals_cov[i] = evals

		snr_cov[i] = (m.T.dot(evecs)) **2 / evals
		# print(evals[:5])
		# print(evecs.shape)
		# print(evals.shape)
		# print((cov[i]==evecs.dot(np.diag(evals).dot(evecs.T))).all())

		# print('k_i={}'.format(K))
		# print('k={}'.format(K))
		k = K

		evals_k = evals.copy()
		evecs_k = evecs.copy()
		evals_k[k:] = 0
		# evecs_k[:, :-k] = 0
		evecs_k = evecs.copy()
		C = evecs_k.dot(np.diag(evals_k).dot(evecs_k.T))

		# C = evecs[:, :-k].dot(np.diag(evals[:-k]).dot(evecs[:, :-k].T))
		print(k)
		print(C.shape)

		# from IPython import embed; embed() 
		# ipdb.set_trace()
		
		C_inv = np.linalg.inv(C)

		# Cm, residuals, rank, s = np.linalg.lstsq(C, m, rcond=-1)
		# Cm = np.linalg.solve(C, m)
		Cm = C_inv.dot(m)
		t2 = 0.5* m.T.dot(Cm)

		sgn, t4 = np.linalg.slogdet(C)
		t4 *= 0.5

		for j in trange(num_images):
			y = logits[j].reshape([-1, 1])

			# Cy, residuals, rank, s = np.linalg.lstsq(C, y, rcond=-1)
			# Cy = np.linalg.solve(C, y)
			Cy = C_inv.dot(y)

			Z[j, i] = m.T.dot(Cy) - t2 - 0.5*y.T.dot(Cy) - t4
	return Z, evals_cov, snr_cov

def test(dict_data, checkpoint_in=None):

	x_train = dict_data['x_train']
	y_train = dict_data['y_train']
	x_test = dict_data['x_test']
	y_test = dict_data['y_test']
	fc_train = dict_data['fc_train']
	fc_test = dict_data['fc_test']
	num_classes = dict_data['num_classes']
	num_train = x_train.shape[0]
	num_test = x_test.shape[0]
	num_features = x_train.shape[1]

	print(fc_train)

	print('Fc:')
	print('Train: {}'.format(np.unique(fc_train)))
	print('Test: {}'.format(np.unique(fc_test)))

	batch_size = 100	
	epochs = 100
	weight_decay = 1e-3

	# print("========================================") 
	# print("MODEL HYPER-PARAMETERS") 
	# print("BATCH SIZE: {:3d}".format(batch_size)) 
	# print("WEIGHT DECAY: {:.4f}".format(weight_decay))
	# print("EPOCHS: {:3d}".format(epochs))
	print("========================================") 
	print("== BUILDING MODEL... ==")

	if checkpoint_in is None:
		raise ValueError('Cannot test without a checkpoint')
		# data_input = Input(batch_shape=(batch_size, num_features, 2))
		# output, model_name = network_20_2(data_input, num_classes, weight_decay)
		# densenet = Model(data_input, output)

	checkpoint_in = checkpoint_in + '-new.h5'
	densenet = load_model(checkpoint_in, 
						  custom_objects={'ComplexConv1D':complexnn.ComplexConv1D,
						  				  'GetAbs': utils.GetAbs,
						  				  'Modrelu': Modrelu})

	densenet.summary()
	# for layer in densenet.layers:
	# 	print(layer.name)
	# densenet = ...  # create the original model

	x_test_freq_off_fixed = add_freq_offset_fixed(x_test, fc_test)

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

	print("\n========================================") 
	print('Test accuracy: {:.2f}%'.format(test_acc))


	probs = densenet.predict(x=x_test_freq_off_fixed, batch_size=batch_size, verbose=0)
	label_pred = probs.argmax(axis=1) 
	label_act = y_test.argmax(axis=1) 
	ind_correct = np.where(label_pred==label_act)[0] 
	ind_wrong = np.where(label_pred!=label_act)[0] 
	assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
	test_acc = 100.*ind_correct.size / num_test

	print("\n========================================") 
	print('Test accuracy with fixed freq offset: {:.2f}%'.format(test_acc))

	conf_matrix_test = metrics.confusion_matrix(label_act, label_pred)
	conf_matrix_test = 100*conf_matrix_test/(num_test/num_classes)
	print('{}'.format(conf_matrix_test))

	output_dict = odict(acc=odict(), comp=odict(), loss=odict())

	output_dict['acc']['test'] = test_acc
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

	return output_dict, summary

if __name__ == "__main__":

	# sample_rate = 200
	sample_rate = 20

	exp_dirs = []
	exp_dirs += ['/home/rfml/wifi/experiments/exp19']
	# exp_dirs += ['/home/rfml/wifi/experiments/exp19_S1']
	# exp_dirs += ['/home/rfml/wifi/experiments/exp19_S2']
	# exp_dirs += ['/home/rfml/wifi/experiments/exp100_S1']
	# exp_dirs += ['/home/rfml/wifi/experiments/exp100_S2']
	# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2']
	# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2']
	# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2']
	# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2']

	# exp_dirs = []
	# # exp_dirs += ['/home/rfml/code_dec/wifi_code/experiments/exp19']
	# # exp_dirs += ['/home/rfml/code_dec/wifi_code/experiments/Test3/converted_3Av2']
	# # exp_dirs += ['/home/rfml/code_dec/wifi_code/experiments/Test3/converted_3Bv2']
	# # exp_dirs += ['/home/rfml/code_dec/wifi_code/experiments/Test3/converted_3Cv2']
	# exp_dirs += ['/home/rfml/code_dec/wifi_code/experiments/Test3/converted_3Dv2']


	preprocess_type = 1
	# preprocess_type = 2
	# preprocess_type = 3
	# preprocess_type = 4

	beta = 0.5
	seed = 0

	sample_duration = 16

	# channel = True
	channel = False

	# diff_days = [False]
	diff_days = [True]
	# diff_days = [True, False]

	snrs = [20]
	# snrs = [10]
	# snrs = [0, 10, 15]
	# snrs = [5]

	num_ch = 10

	for exp_dir in exp_dirs:
		for diff_day in diff_days:
			for snr in snrs:
			
				if channel is False:
					data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)
				else:
					data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}-dd-{:}-snr-{:.0f}-b-{:.0f}-s-{:}'.format(sample_duration, preprocess_type, sample_rate, int(diff_day), snr, 100*beta, seed)

				start = timer()

				# data_format = '0-{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

				npz_filename = exp_dir + '/sym-' + data_format + '.npz'

				np_dict = np.load(npz_filename)
				dict_wifi = {}
				dict_wifi['x_train'] = np_dict['arr_0']
				dict_wifi['y_train'] = np_dict['arr_1']
				dict_wifi['x_test'] = np_dict['arr_2']
				dict_wifi['y_test'] = np_dict['arr_3']
				dict_wifi['fc_train'] = np_dict['arr_4']
				dict_wifi['fc_test'] = np_dict['arr_5']
				dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]

				# if (preprocess_type==3) and (exp_dir[-4:]=='3Av2'):
				# 	dict_wifi['x_train'] = dict_wifi['x_train'][..., 0]
				# 	dict_wifi['x_test'] = dict_wifi['x_test'][..., 0]

				end = timer()
				print('Load time: {:} sec'.format(end - start))

				# Checkpoint path
				checkpoint = exp_dir + '/ckpt-' + data_format + '.h5'

				print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
				train_output, summary = test(dict_wifi, checkpoint_in=checkpoint)
				print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')

				# # Write logs
				# with open(exp_dir + '/logs-' + data_format  + '.txt', 'a+') as f:
				# 	f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')
				# 	for keys, dicts in train_output.items():
				# 		f.write(str(keys)+':\n')
				# 		for key, value in dicts.items():
				# 			f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
				# 	f.write('\n'+str(summary))
