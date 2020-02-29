from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from collections import OrderedDict as odict
from timeit import default_timer as timer
import ipdb
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

# import complexnn
from .complexnn import ComplexDense, ComplexConv1D, utils

from .cxnn.train_network  import set_keras_backend

import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rc('text', usetex=True)
# mpl.rc('text.latex', preamble = r'\usepackage{amsmath}, \usepackage{sfmath}, \usepackage{amssymb}, \usepackage{bm}, \DeclareMathOperator*{\supp}{supp}, \DeclareMathOperator*{\proj}{\mathcal{P}_K}, \newcommand{\be}{{\bm e}}, \newcommand{\bx}{{\bm x}}, \DeclareMathOperator*{\support}{\mathcal{S}_K}')

set_keras_backend("theano")
np.set_printoptions(precision=2)

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
	num_classes = dict_data['num_classes']
	num_train = x_train.shape[0]
	num_test = x_test.shape[0]
	num_features = x_train.shape[1]

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
						  				  'GetAbs': utils.GetAbs})

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

	print(densenet.summary())
	# for layer in densenet.layers:
	# 	print(layer.name)
	# densenet = ...  # create the original model

	######################################
	# Mean and cov_train
	######################################

	layer_name = densenet.layers[-1].name
	print(layer_name)
	model_2 = Model(inputs=densenet.input,
                    outputs=densenet.get_layer(layer_name).input)
	weight, bias = densenet.get_layer(layer_name).get_weights()

	logits_test = model_2.predict(x=x_test, batch_size=batch_size, verbose=0)
	logits_test = logits_test.dot(weight) + bias

	logits_train = model_2.predict(x=x_train, batch_size=batch_size, verbose=0)
	logits_train = logits_train.dot(weight) + bias

	layer_name = densenet.layers[-2].name
	print(layer_name)
	# layer_name = 'dense_1'
	# layer_name = 'dense_2'
	# layer_name = 'dense_3'
	# layer_name = 'dense_5'
	# layer_name = 'dense_35'
	intermediate_layer_model = Model(inputs=densenet.input,
	                                 outputs=densenet.get_layer(layer_name).output)
	features_test = intermediate_layer_model.predict(x_test, batch_size=batch_size)
	features_train = intermediate_layer_model.predict(x_train, batch_size=batch_size)

	layer_name = densenet.layers[1].name
	weight, = densenet.get_layer(layer_name).get_weights()

	w = weight[:, 0, :200] + 1j*weight[:, 0, 200:]
	print(w.shape)

	X = w.copy() # shape [num_test, num_classes]
	ind = np.where(np.abs(X).max(axis=0) > 1e-10)[0]
	X = X[:, ind]
	X -= X.mean(axis=0)
	X /= X.std(axis=0)
	X = np.nan_to_num(X)
	rho_w = np.matmul(X.T.conj(), X) / X.shape[0]

	plt.figure(figsize=[5, 5])
	plt.imshow(rho_w.real, vmin=-1.0, vmax=1.0)   
	plt.colorbar() 
	plt.tight_layout(rect=[0, 0.03, 1, 0.9])
	plt.subplots_adjust(wspace=0.3, hspace=0.4)
	plt.title('Covariance of filters')
	# plt.savefig('filt_cov.pdf', format='pdf', dpi=1000, bbox_inches='tight')

	from IPython import embed; embed()
	ipdb.set_trace()


	# for i in range(5):
	# 	print(np.count_nonzero(logits_test[i, :]))

	# for i in range(5):
	# 	print(np.count_nonzero(features_test[i, :]))

	mu_train, cov_train, rho_train = mu_cov_calc(logits_train, y_train)
	mu_test, cov_test, rho_test = mu_cov_calc(logits_test, y_test)

	cov_eye = np.zeros([num_classes, num_classes, num_classes])
	for n in range(num_classes):
		cov_eye[n] = np.identity(num_classes)

	cov_train_diag = np.zeros([num_classes, num_classes, num_classes])
	for n in range(num_classes):
		cov_train_diag[n] = np.diag(np.diag(cov_train[n]))

	prec_rho_train = np.zeros([num_classes, num_classes, num_classes])
	for i in range(num_classes):
		prec_rho_train[i] = np.linalg.inv(rho_train[i])
		# if i<5:
		# 	print(100 - 100*np.count_nonzero(prec_rho_train[i])/num_classes/num_classes)
		# 	print(np.linalg.det(rho_train[i]))
		# 	print(np.linalg.cond(rho_train[i]))

	Z = Z_calc(logits_test, mu_train, cov_train_diag)

	label_z = Z.argmax(axis=1)
	ind_correct_z = np.where(label_z==label_act)[0] 
	test_acc_z = 100.*ind_correct_z.size / num_test
	print('\nTest accuracy: {:.2f}% \n Z acc: {:.2f}%'.format(test_acc, test_acc_z))

	# acc_z_i = np.zeros([num_classes])
	# for class_idx in range(num_classes):
	# 	idx_inclass = np.where(label_act==class_idx)[0]
	# 	ind_correct = np.where(label_z[idx_inclass]==label_act[idx_inclass])[0] 
	# 	acc_z_i[class_idx] = 100*ind_correct.size / idx_inclass.size

	from IPython import embed; embed()
	ipdb.set_trace()

	label_test = label_act.copy()

	for j in range(2):
		# w_proj = vh_test_class[j][:np.int(n_comp_test_class[j])].dot(weight)
		# w_proj = vh_test_class[j][:10].dot(weight)
		idx_inclass = np.where(label_test==j)[0]
		Z_j = Z[idx_inclass]

		plt.figure(figsize=(15,2))
		for i in range(5):
			plt.subplot(1, 5, i+1)
			plt.hist(Z_j[:, i], density=True, bins=15)
			# plt.hist(w_proj[:, i], density=True)
			# plt.xlim([-1, 1])
			# plt.ylim([0, 2.5])
			# plt.xlim([-3, 3])
			# plt.ylim([0, 1.2])
			# plt.ylim([0, 0.8])
			# plt.xticks(np.arange(1, k+1, 2))
			# plt.xlabel(r'$\left|\support\left(\bx\right) \cap \support\left(\bx+\be\right)\right|$, where K = {:.0f}'.format(k), fontsize=11)
			plt.title('Class {}'.format(i), fontsize=12)
			plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 
			# plt.ylim([0, 5000])
		plt.suptitle('Full decision rule for images of class {})'.format(j))	

	k = 100
	Z_top_k, evals_cov, snr_cov = Z_calc_smallest_k(logits_test, mu_train, cov_train_diag, k)


	for j in range(2):
		# w_proj = vh_test_class[j][:np.int(n_comp_test_class[j])].dot(weight)
		# w_proj = vh_test_class[j][:10].dot(weight)
		idx_inclass = np.where(label_test==j)[0]
		Z_j = Z_top_k[idx_inclass, :]

		plt.figure(figsize=(15,2))
		for i in range(5):
			plt.subplot(1, 5, i+1)
			plt.hist(Z_j[:, i], density=True, bins=15)
			# plt.hist(w_proj[:, i], density=True)
			# plt.xlim([-1, 1])
			# plt.ylim([0, 2.5])
			# plt.xlim([-3, 3])
			# plt.ylim([0, 1.2])
			# plt.ylim([0, 0.8])
			# plt.xticks(np.arange(1, k+1, 2))
			# plt.xlabel(r'$\left|\support\left(\bx\right) \cap \support\left(\bx+\be\right)\right|$, where K = {:.0f}'.format(k), fontsize=11)
			plt.title('Class {}'.format(i), fontsize=12)
			plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 
			# plt.ylim([0, 5000])

		plt.suptitle('Smallest-{} decision rule for images of class {})'.format(k, j))	

	# for j in range(2):

	# 	plt.figure(figsize=(15,2))
	# 	for i in range(5):
	# 		plt.subplot(1, 5, i+1)
	# 		plt.hist(evals_cov[i], density=True, bins=15)
	# 		# plt.hist(w_proj[:, i], density=True)
	# 		# plt.xlim([-1, 1])
	# 		# plt.ylim([0, 2.5])
	# 		# plt.xlim([-3, 3])
	# 		# plt.ylim([0, 1.2])
	# 		# plt.ylim([0, 0.8])
	# 		# plt.xticks(np.arange(1, k+1, 2))
	# 		# plt.xlabel(r'$\left|\support\left(\bx\right) \cap \support\left(\bx+\be\right)\right|$, where K = {:.0f}'.format(k), fontsize=11)
	# 		plt.title('Class {}'.format(i), fontsize=12)
	# 		plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 

	# 	plt.suptitle('Eigenvalues for images of class {})'.format(j))	


	plt.figure(figsize=(15,2))
	for i in np.arange(35, 45):
		plt.subplot(1, 10, i+1-35)
		plt.hist(snr_cov[i], density=True, bins=15)
		# plt.hist(w_proj[:, i], density=True)
		# plt.xlim([-1, 1])
		# plt.ylim([0, 2.5])
		# plt.xlim([-3, 3])
		# plt.ylim([0, 1.2])
		# plt.ylim([0, 0.8])
		# plt.xticks(np.arange(1, k+1, 2))
		# plt.xlabel(r'$\left|\support\left(\bx\right) \cap \support\left(\bx+\be\right)\right|$, where K = {:.0f}'.format(k), fontsize=11)
		plt.title('Class {}'.format(i), fontsize=12)
		plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 

	plt.suptitle('SNR')	

	plt.figure(figsize=(15,2))
	for i in np.arange(35, 45):
		plt.subplot(1, 10, i+1-35)
		plt.hist(1./evals_cov[i], density=True, bins=15)
		# plt.hist(w_proj[:, i], density=True)
		# plt.xlim([-1, 1])
		# plt.ylim([0, 2.5])
		# plt.xlim([-3, 3])
		# plt.ylim([0, 1.2])
		# plt.ylim([0, 0.8])
		# plt.xticks(np.arange(1, k+1, 2))
		# plt.xlabel(r'$\left|\support\left(\bx\right) \cap \support\left(\bx+\be\right)\right|$, where K = {:.0f}'.format(k), fontsize=11)
		plt.title('Class {}'.format(i), fontsize=12)
		plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 

		plt.suptitle('1/eigenvalues')	

	z_bias = np.array([100*(label_z==i).mean() for i in np.arange(100)])
	plt.figure()
	plt.plot(z_bias)
	plt.xlabel('Classes')
	plt.title('No of times each class is predicted by decision rule')

	plt.figure()
	plt.plot(Z.mean(axis=0))
	plt.xlabel('Classes')
	plt.title('Mean value of decision statistic for each class')


	num_classes_plot = 19
	plt.figure(figsize=[15, 6])
	num_rows = 4
	num_cols = 5
	for i in range(num_classes_plot):
		plt.subplot(num_rows, num_cols, i+1)
		plt.bar(np.arange(num_classes), height = mu_train[i]- mu_train[i].min())   
		plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 
		plt.title('Class {}'.format(i))
	plt.suptitle('Mean')

	plt.figure(figsize=[15, 10])
	for i in range(num_classes_plot):
		plt.subplot(num_rows, num_cols, i+1)
		plt.imshow(rho_train[i], vmin=-1.0, vmax=1.0)   
		plt.colorbar() 
		plt.tight_layout(rect=[0, 0.03, 1, 0.9])
		plt.subplots_adjust(wspace=0.3, hspace=0.4)
		plt.title('Class {} (acc = {:.0f})'.format(i, acc_class[i]))
	plt.suptitle('Covariance coefficient')


	plt.figure(figsize=[15, 10])
	for i in range(num_classes_plot):
		plt.subplot(num_rows, num_cols, i+1)
		plt.imshow(prec_rho_train[i])   
		plt.colorbar() 
		plt.tight_layout(rect=[0, 0.03, 1, 0.9])
		plt.subplots_adjust(wspace=0.3, hspace=0.4)
		plt.title('Class {} (acc = {:.0f})'.format(i, acc_class[i]))
	plt.suptitle('Precision')

	# for j in range(5):
	# 	# w_proj = vh_test_class[j][:np.int(n_comp_test_class[j])].dot(weight)
	# 	# w_proj = vh_test_class[j][:10].dot(weight)
	# 	idx_inclass = np.where(label_test==j)[0]
	# 	Z_j = Z[idx_inclass]

	# 	plt.figure(figsize=(15,2))
	# 	for i in range(10):
	# 		plt.subplot(1, 10, i+1)
	# 		plt.hist(Z_j[:, i], density=True, bins=15)
	# 		# plt.hist(w_proj[:, i], density=True)
	# 		# plt.xlim([-1, 1])
	# 		# plt.ylim([0, 2.5])
	# 		# plt.xlim([-3, 3])
	# 		# plt.ylim([0, 1.2])
	# 		# plt.ylim([0, 0.8])
	# 		# plt.xticks(np.arange(1, k+1, 2))
	# 		# plt.xlabel(r'$\left|\support\left(\bx\right) \cap \support\left(\bx+\be\right)\right|$, where K = {:.0f}'.format(k), fontsize=11)
	# 		plt.title('Class {}'.format(i), fontsize=12)
	# 		# plt.ylim([0, 5000])
	# 	plt.suptitle('Decision rule for images of class {} (acc = {:.0f})'.format(j, acc_z_i[j]))
	# 	plt.tight_layout(rect=[0, 0.03, 1, 0.9])
	# 	plt.subplots_adjust(wspace=0.3)

	# Z_test = Z_calc(logits_test, mu_test, cov_test)

	# label_z_test = Z_test.argmax(axis=1)
	# ind_correct_z_test = np.where(label_z_test==label_act)[0] 
	# test_acc_z_test = 100.*ind_correct_z_test.size / num_test
	# print('\nTest accuracy: {:.2f}% \n Z_test acc: {:.2f}%'.format(test_acc, test_acc_z_test))

	# from IPython import embed; embed()
	# ipdb.set_trace()

	# num_classes_plot = 19
	# plt.figure(figsize=[15, 6])
	# num_rows = 4
	# num_cols = 5
	# for i in range(num_classes_plot):
	# 	plt.subplot(num_rows, num_cols, i+1)
	# 	plt.bar(np.arange(num_classes), height = mu_test[i]- mu_test[i].min())   
	# 	plt.tight_layout() 
	# 	plt.title('Class {}'.format(i))
	# plt.suptitle('Mean, test data')

	# plt.figure(figsize=[15, 10])
	# for i in range(num_classes_plot):
	# 	plt.subplot(num_rows, num_cols, i+1)
	# 	plt.imshow(rho_test[i], vmin=-1.0, vmax=1.0)   
	# 	plt.colorbar() 
	# 	plt.tight_layout()
	# 	plt.subplots_adjust(wspace=0.3, hspace=0.4)
	# 	plt.title('Class {} (acc = {:.0f})'.format(i, acc_class[i]))
	# plt.suptitle('Covariance coefficient, test data')

	# plt.figure(figsize=[15, 10])
	# for i in range(num_classes_plot):
	# 	plt.subplot(num_rows, num_cols, i+1)
	# 	plt.imshow(np.linalg.inv(rho_test[i]))   
	# 	plt.colorbar() 
	# 	plt.tight_layout()
	# 	plt.subplots_adjust(wspace=0.3, hspace=0.4)
	# 	plt.title('Class {} (acc = {:.0f})'.format(i, acc_class[i]))
	# plt.suptitle('Precision, test data')


	# prec_train = np.zeros([num_classes, num_classes, num_classes])
	# for i in range(num_classes):
	# 	prec_train[i] = np.linalg.inv(cov_train[i])
	# 	if i<5:
	# 		print(100 - 100*np.count_nonzero(prec_train[i])/num_classes/num_classes)

	# plt.figure(figsize=[15, 10])
	# for i in range(num_classes_plot):
	# 	plt.subplot(num_rows, num_cols, i+1)
	# 	plt.imshow(prec_train[i])   
	# 	plt.colorbar() 
	# 	plt.tight_layout()
	# 	plt.subplots_adjust(wspace=0.3, hspace=0.4)
	# 	plt.title('Class {} (acc = {:.0f})'.format(i, acc_class[i]))
	# # plt.suptitle('Train data, inv')

	# y_bias = np.array([100*(label_pred==i).mean() for i in np.arange(50)])
	# plt.figure()
	# plt.plot(y_bias)
	# plt.xlabel('Classes')
	# plt.title('No of times each class is predicted by softmax')

	# plt.figure()
	# plt.plot(logits_test.mean(axis=0))
	# plt.xlabel('Classes')
	# plt.title('Mean value of logits_test for each class')


	'''
	######################################
	# PCA
	######################################
	layer_name = 'dense_1'
	# layer_name = 'dense_2'
	# layer_name = 'dense_3'
	# layer_name = 'dense_5'
	# layer_name = 'dense_35'
	intermediate_layer_model = Model(inputs=densenet.input,
	                                 outputs=densenet.get_layer(layer_name).output)
	features_test = intermediate_layer_model.predict(x_test, batch_size=batch_size)
	features_train = intermediate_layer_model.predict(x_train, batch_size=batch_size)

	features_test_scaled = StandardScaler().fit_transform(features_test)
	features_train_scaled = StandardScaler().fit_transform(features_train)

	sing = np.linalg.svd(features_test_scaled, full_matrices=True, compute_uv=False)
	# print('rank = {}'.format(sing.shape))
	# energy = (sing**2).cumsum()
	# energy *= 100/energy[-1]
	# ind = np.where(energy>99)[0][0]
	energy = (sing**2)
	energy *= 100/energy.sum()
	ind = np.where(energy.cumsum() > 99)[0][0]
	print('SVD: No of components = {}'.format(ind))

	n_comp_test_class = np.zeros([num_classes])
	energy_test_class = np.zeros([num_classes, 5])
	rank_test_class = np.zeros([num_classes])
	u_test_class = []
	vh_test_class = []

	for i in range(num_classes):
		ind_class = np.where(y_test.argmax(axis=1)==i)[0]
		features_test_class = features_test[ind_class, :]
		features_test_class_scaled = StandardScaler().fit_transform(features_test_class)

		u, sing, vh = np.linalg.svd(features_test_class_scaled, full_matrices=False, compute_uv=True)
		u_test_class += [u]
		vh_test_class += [vh]
		# print('rank = {}'.format(sing.shape))
		energy = (sing**2)
		energy *= 100/energy.sum()
		ind = np.where(energy.cumsum() > 99)[0][0]
		# print('SVD: No of components = {}'.format(ind))
		# print('Class {}: {}, {}'.format(i, ind, (sing**2)[:5]))
		n_comp_test_class[i] = ind
		energy_test_class[i] = energy[:5]
		rank_test_class[i] = sing.size

	print('\nNo of components per class: {} < {} +/- {} < {}'.format(n_comp_test_class.min(), n_comp_test_class.mean(), n_comp_test_class.std(), n_comp_test_class.max()))
	print('\nEnergy per class: \n   {} < \n   {} +/- {} \n < {}'.format(energy_test_class.min(axis=0), energy_test_class.mean(axis=0), energy_test_class.std(axis=0), energy_test_class.max(axis=0)))
	print('\nRank per class: {} < {} +/- {} < {}'.format(rank_test_class.min(), rank_test_class.mean(), rank_test_class.std(), rank_test_class.max()))


	weight, bias = densenet.get_layer('dense_2').get_weights()
	# ipdb.set_trace()
	print(weight.shape)

	# w_proj = vh_test_class[0].dot(weight)

	# for j in range(10):
	# 	# w_proj = vh_test_class[j][:np.int(n_comp_test_class[j])].dot(weight)
	# 	# w_proj = vh_test_class[j][:10].dot(weight)
	# 	w_proj = vh_test_class[j].dot(weight)

	# 	plt.figure(figsize=(15,2))
	# 	for i in range(10):
	# 		plt.subplot(1, 10, i+1)
	# 		plt.hist(w_proj[:, i], density=True, bins=15)
	# 		# plt.hist(w_proj[:, i], density=True)
	# 		# plt.xlim([-1, 1])
	# 		# plt.ylim([0, 2.5])
	# 		# plt.xlim([-3, 3])
	# 		# plt.ylim([0, 1.2])
	# 		# plt.ylim([0, 0.8])
	# 		# plt.xticks(np.arange(1, k+1, 2))
	# 		# plt.xlabel(r'$\left|\support\left(\bx\right) \cap \support\left(\bx+\be\right)\right|$, where K = {:.0f}'.format(k), fontsize=11)
	# 		plt.title('Class {}'.format(i+1), fontsize=12)
	# 		# plt.ylim([0, 5000])
	# 	plt.suptitle('Projections of weights for class {}'.format(j+1))
	# 	plt.tight_layout(rect=[0, 0.03, 1, 0.9])
	# 	plt.subplots_adjust(wspace=0.3)

	# plt.show()

	layer_name = 'global_average_pooling1d_1'
	glob_model = Model(inputs=densenet.input,
                       outputs=densenet.get_layer(layer_name).input)
	glob_test = glob_model.predict(x_test, batch_size=batch_size)
	glob_train = glob_model.predict(x_train, batch_size=batch_size)

	for t in range(glob_test.shape[1]):
		features_test = glob_test[:, t, :]
		features_train = glob_train[:, t, :]
		features_test_scaled = StandardScaler().fit_transform(features_test)
		features_train_scaled = StandardScaler().fit_transform(features_train)

		print('----------------------------')
		print('Time step {}'.format(t))
		print('----------------------------')

		sing = np.linalg.svd(features_test_scaled, full_matrices=True, compute_uv=False)
		# print('rank = {}'.format(sing.shape))
		# energy = (sing**2).cumsum()
		# energy *= 100/energy[-1]
		# ind = np.where(energy>99)[0][0]
		energy = (sing**2)
		energy *= 100/energy.sum()
		ind = np.where(energy.cumsum() > 99)[0][0]
		print('SVD: No of components = {}'.format(ind))

		n_comp_test_class = np.zeros([num_classes])
		energy_test_class = np.zeros([num_classes, 5])
		rank_test_class = np.zeros([num_classes])
		u_test_class = []
		vh_test_class = []

		for i in range(num_classes):
			ind_class = np.where(y_test.argmax(axis=1)==i)[0]
			features_test_class = features_test[ind_class, :]
			features_test_class_scaled = StandardScaler().fit_transform(features_test_class)

			u, sing, vh = np.linalg.svd(features_test_class_scaled, full_matrices=False, compute_uv=True)
			u_test_class += [u]
			vh_test_class += [vh]
			# print('rank = {}'.format(sing.shape))
			energy = (sing**2)
			energy *= 100/energy.sum()
			ind = np.where(energy.cumsum() > 99)[0][0]
			# print('SVD: No of components = {}'.format(ind))
			# print('Class {}: {}, {}'.format(i, ind, (sing**2)[:5]))
			n_comp_test_class[i] = ind
			energy_test_class[i] = energy[:5]
			rank_test_class[i] = sing.size

		print('\nNo of components per class: {} < {} +/- {} < {}'.format(n_comp_test_class.min(), n_comp_test_class.mean(), n_comp_test_class.std(), n_comp_test_class.max()))
		print('\nEnergy per class: \n   {} < \n   {} +/- {} \n < {}'.format(energy_test_class.min(axis=0), energy_test_class.mean(axis=0), energy_test_class.std(axis=0), energy_test_class.max(axis=0)))
		print('\nRank per class: {} < {} +/- {} < {}'.format(rank_test_class.min(), rank_test_class.mean(), rank_test_class.std(), rank_test_class.max()))

	

	from IPython import embed; embed()
	ipdb.set_trace()

	# print('Total no. of components in test data: {}'.format(features_test_scaled.shape[1]))
	# print('Total no. of components in train data: {}\n'.format(features_train_scaled.shape[1]))

	# # pca = PCA(n_components=0.99, svd_solver='full')
	# pca = PCA(n_components=0.99, svd_solver='full')
	# features_test_reduced = pca.fit_transform(features_test_scaled)
	# print('No of components in test data: {}'.format(pca.n_components_))
	# # print('% of variance explained: {}'.format(100*pca.explained_variance_ratio_.cumsum()))

	# n_comp_test_class = np.zeros([num_classes])
	# for i in range(num_classes):
	# 	ind_class = np.where(y_test.argmax(axis=1)==i)[0]
	# 	features_test_class = features_test[ind_class, :]
	# 	features_test_class_scaled = StandardScaler().fit_transform(features_test_class)

	# 	# features_test_class_scaled = features_test_scaled[ind_class, :]
	# 	# x_train_class = x_train[ind_class, :]
	# 	pca = PCA(n_components=0.99, svd_solver='full')
	# 	# ipdb.set_trace()
	# 	pca.fit(features_test_class_scaled)
	# 	# print('Class {}: {}'.format(i, pca.n_components_))
	# 	n_comp_test_class[i] = pca.n_components_
	# 	# print('Class {}: {}, {}'.format(i, pca.n_components_, 100*pca.explained_variance_ratio_.cumsum()))
	# print('Average per-class dimension: {} +/- {}'.format(n_comp_test_class.mean(), n_comp_test_class.std()))

	# pca = PCA(n_components=0.99, svd_solver='full')
	# pca.fit(features_train_scaled)
	# print('\nNo of components in train data: {}'.format(pca.n_components_))
	# # print('% of variance explained: {}'.format(100*pca.explained_variance_ratio_.cumsum()))

	# n_comp_train_class = np.zeros([num_classes])
	# for i in range(num_classes):
	# 	ind_class = np.where(y_train.argmax(axis=1)==i)[0]
	# 	# features_train_class = features_train[ind_class, :]
	# 	# features_train_class_scaled = StandardScaler().fit_transform(features_train_class)
		
	# 	features_train_class_scaled = features_train_scaled[ind_class, :]
	# 	pca = PCA(n_components=0.99, svd_solver='full')
	# 	pca.fit(features_train_class_scaled)
	# 	# print('Class {}: {}'.format(i, pca.n_components_))
	# 	n_comp_train_class[i] = pca.n_components_
	# 	# print('Class {}: {}, {}'.format(i, pca.n_components_, 100*pca.explained_variance_ratio_.cumsum()))
	# print('Average per-class dimension: {} +/- {}'.format(n_comp_train_class.mean(), n_comp_train_class.std()))
	
	# sing = np.linalg.svd(features_test_scaled, full_matrices=True, compute_uv=False)
	# print('rank = {}'.format(sing.shape))
	# energy = (sing**2).cumsum()
	# energy *= 100/energy[-1]
	# ind = np.where(energy>99)[0][0]
	# print('No of components = {}'.format(ind))

	# cov_train = features_test_scaled.T @ features_test_scaled
	# evals , evecs = LA.eigh(cov_train)
	# idx = np.argsort(evals)[::-1]
	# evecs = evecs[:,idx]
	# evals = evals[idx]
	# energy_eig = evals.cumsum()
	# energy_eig *= 100/energy_eig[-1]
	# ind = np.where(energy_eig>99)[0][0]
	# print('No of components = {}'.format(ind))
	'''

	# Print model architecture
	# print(densenet.summary())

	# # Set optimizer and loss function
	# optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	# densenet.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	# print("== START TRAINING... ==")
	# history = densenet.fit(x=x_train, 
	# 					   y=y_train, 
	# 					   epochs=epochs, 
	# 					   batch_size=batch_size, 
	# 					   validation_data=(x_test, y_test), 
	# 					   callbacks=[])

	# if checkpoint_out is not None:
	# 	densenet.save(checkpoint_out)



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
	# exp_dirs += ['/home/rfml/wifi/experiments/exp19']
	# exp_dirs += ['/home/rfml/wifi/experiments/exp19_S1']
	# exp_dirs += ['/home/rfml/wifi/experiments/exp19_S2']
	# exp_dirs += ['/home/rfml/wifi/experiments/exp100_S1']
	# exp_dirs += ['/home/rfml/wifi/experiments/exp100_S2']
	# exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2']
	exp_dirs += ['/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2']
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
