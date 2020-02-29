from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import numpy.random as random
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
mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble = r'\usepackage{amsmath}, \usepackage{sfmath}, \usepackage{amssymb}, \usepackage{bm}, \DeclareMathOperator*{\supp}{supp}, \DeclareMathOperator*{\proj}{\mathcal{P}_K}, \newcommand{\be}{{\bm e}}, \newcommand{\bx}{{\bm x}}, \DeclareMathOperator*{\support}{\mathcal{S}_K}')

set_keras_backend("theano")
np.set_printoptions(precision=2)

def mu_cov_calc(logits, y):
	# num_classes = classes_train.size
	num_classes = y.shape[1]
	labels = np.argmax(y, axis=1)

	num_neurons = logits.shape[1]

	mu = np.zeros((num_classes, num_neurons))
	rho = np.zeros((num_classes, num_neurons, num_neurons))
	cov = np.zeros((num_classes, num_neurons, num_neurons))

	for class_idx in trange(num_classes):
		idx_inclass = np.where(labels==class_idx)[0]
		mu[class_idx, :] = logits[idx_inclass].mean(axis=0)

	for class_idx in trange(num_classes):
		idx_inclass = np.where(labels==class_idx)[0]
		if idx_inclass.size != 0:
			X = logits[idx_inclass, :] # shape [num_test, num_classes]
			X_bar = X - X.mean(axis=0)
			X_norm = StandardScaler().fit_transform(X)
			cov[class_idx] = np.matmul(X_bar.T, X_bar) / idx_inclass.size
			rho[class_idx] = np.matmul(X_norm.T, X_norm) / idx_inclass.size

	return mu, cov, rho

def Z_calc(logits, mu, cov):
	num_classes = mu.shape[0]
	num_images = logits.shape[0]
	Z_test = np.zeros([num_images, num_classes])

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

			Z_test[j, i] = m.T.dot(Cy) - t2 - 0.5*y.T.dot(Cy) - t4
	return -Z_test

def Z_calc_2(logits, mu, cov):
	num_classes = mu.shape[0]
	num_images = logits.shape[0]
	Z_test = np.zeros([num_images, num_classes])

	for i in trange(num_classes):
		m = mu[i].reshape([-1, 1])
		C = cov[i]

		C_inv = np.linalg.inv(C)

		# Cm, residuals, rank, s = np.linalg.lstsq(C, m, rcond=-1)
		# Cm = np.linalg.solve(C, m)
		Cm = C_inv.dot(m)
		t2 = 0.5* m.T.dot(Cm)

		# sgn, t4 = np.linalg.slogdet(C)
		# t4 *= 0.5


		for j in trange(num_images):
			y = logits[j].reshape([-1, 1])

			# Cy, residuals, rank, s = np.linalg.lstsq(C, y, rcond=-1)
			# Cy = np.linalg.solve(C, y)
			Cy = C_inv.dot(y)

			Z_test[j, i] = m.T.dot(Cy) - t2 - 0.5*y.T.dot(Cy)
	return -Z_test

def Z_calc_smallest_k(logits, mu, cov, K):
	num_classes = mu.shape[0]
	num_images = logits.shape[0]
	Z_test = np.zeros([num_images, num_classes])
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

			Z_test[j, i] = m.T.dot(Cy) - t2 - 0.5*y.T.dot(Cy) - t4
	return -Z_test, evals_cov, snr_cov

def test(dict_data, checkpoint_in=None):

	x_train = dict_data['x_train']
	y_train = dict_data['y_train']
	x_test = dict_data['x_test']
	y_test = dict_data['y_test']
	x_novel = dict_data['x_novel']
	num_classes_test = dict_data['num_classes']
	num_train = x_train.shape[0]
	num_test = x_test.shape[0]
	num_features = x_train.shape[1]

	batch_size = 100	
	epochs = 100
	weight_decay = 1e-3

	print("========================================") 
	print("== BUILDING MODEL... ==")

	if checkpoint_in is None:
		raise ValueError('Cannot test without a checkpoint')

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

	# classes_train = np.setdiff1d(np.arange(num_classes_test), novel_classes)
	# num_classes = classes_train.size
	num_classes = num_classes_test

	acc_class = np.zeros([num_classes_test])
	for class_idx in range(num_classes_test):
		idx_inclass = np.where(label_act==class_idx)[0]
		ind_correct = np.where(label_pred[idx_inclass]==label_act[idx_inclass])[0] 
		acc_class[class_idx] = 100*ind_correct.size / idx_inclass.size

	print(densenet.summary())

	print("\n========================================") 
	print('Test accuracy: {:.2f}%'.format(test_acc))
	print('Accuracy per class: {}%'.format(acc_class))

	# print(densenet.layers)
	# for layer in densenet.layers:
	# 	print(layer.name)
	# densenet = ...  # create the original model

	######################################
	# Mean and cov_train
	######################################

	# layer_name = densenet.layers[-3].name
	# layer_name = densenet.layers[-2].name
	layer_name = densenet.layers[-1].name
	print(layer_name)
	# layer_name = 'dense_2'
	# layer_name = 'dense_4'
	# layer_name = 'dense_6'
	# layer_name = 'dense_10'
	# layer_name = 'dense_14'
	model_2 = Model(inputs=densenet.input,
                    outputs=densenet.get_layer(layer_name).input)
	weight, bias = densenet.get_layer(layer_name).get_weights()

	logits_test = model_2.predict(x=x_test, batch_size=batch_size, verbose=0)
	logits_test = logits_test.dot(weight) + bias

	logits_train = model_2.predict(x=x_train, batch_size=batch_size, verbose=0)
	logits_train = logits_train.dot(weight) + bias

	logits_novel = model_2.predict(x=x_novel, batch_size=batch_size, verbose=0)
	logits_novel = logits_novel.dot(weight) + bias

	# logits_test = logits_test[:, list(classes_train)]
	# logits_train = logits_train[:, list(classes_train)]
	num_neurons = logits_test.shape[1]

	print('Sparsity of logits_test:')
	for i in range(5):
		print(np.count_nonzero(logits_test[i, :]))

	mu_train, cov_train, rho_train = mu_cov_calc(logits_train, y_train)
	# mu_test, cov_test, rho_test = mu_cov_calc(logits_test, y_test)

	cov_eye = np.zeros([num_classes, num_neurons, num_neurons])
	for n in range(num_classes):
		cov_eye[n] = np.identity(num_neurons)

	cov_train_diag = np.zeros([num_classes, num_neurons, num_neurons])
	for n in range(num_classes):
		cov_train_diag[n] = np.diag(np.diag(cov_train[n]))

	# prec_rho_train = np.zeros([num_classes, num_classes, num_classes])
	# for i in range(num_classes):
	# 	prec_rho_train[i] = np.linalg.inv(rho_train[i])
		# if i<5:
		# 	print(100 - 100*np.count_nonzero(prec_rho_train[i])/num_classes/num_classes)
		# 	print(np.linalg.det(rho_train[i]))
		# 	print(np.linalg.cond(rho_train[i]))


	thresh = np.zeros([num_classes])

	# Z_train_full = Z_calc(logits_train, mu_train, cov_train)
	Z_test_full = Z_calc(logits_test, mu_train, cov_train)
	# Z_novel_full = Z_calc(logits_novel, mu_train, cov_train)

	Z_train = Z_calc_2(logits_train, mu_train, cov_train)
	Z_test = Z_calc_2(logits_test, mu_train, cov_train)
	Z_novel = Z_calc_2(logits_novel, mu_train, cov_train)

	# Z_train_full = Z_calc(logits_train, mu_train, cov_train_diag)
	# Z_test_full = Z_calc(logits_test, mu_train, cov_train_diag)
	# # Z_novel_full = Z_calc(logits_novel, mu_train, cov_train_diag)

	# Z_train = Z_calc_2(logits_train, mu_train, cov_train_diag)
	# Z_test = Z_calc_2(logits_test, mu_train, cov_train_diag)
	# Z_novel = Z_calc_2(logits_novel, mu_train, cov_train_diag)

	# thresh = Z_train.max(axis=0)
	# acc_det_novel = 100*(Z_novel > thresh).all(axis=1).mean()
	# print('Novel acc 1 = {:.2f}'.format(acc_det_novel))

	prob = 0.2
	for n in range(num_classes):
		ind_n = np.where(y_train.argmax(axis=1)==n)[0]
		K = np.ceil(prob*ind_n.size).astype(np.int)
		Z_n = Z_train[ind_n, n]
		ind_top_k = np.argpartition(Z_n, -K)[-K:]
		ind_top_k_sorted = ind_top_k[np.argsort(Z_n[ind_top_k])]
		thresh[n] = Z_n[ind_top_k_sorted[0]]
		# thresh[n] = Z_train[ind_n, n].max()

	# acc_det_test = 100*(Z_test <= thresh).any(axis=1).mean()

	acc_det_test = 100 - 100*((Z_test > thresh).all(axis=1)).mean()
	acc_det_novel = 100*(Z_novel > thresh).all(axis=1).mean()

	ind_det_test = np.where((Z_test <= thresh).any(axis=1))[0]
	label_z = Z_test_full[ind_det_test].argmin(axis=1)
	ind_correct_z = np.where(label_z==label_act[ind_det_test])[0] 
	test_acc_z = 100.*ind_correct_z.size / num_test
	print('\nTest accuracy: {:.2f}% \n Z_test acc: {:.2f}%'.format(test_acc, test_acc_z))
	print('Detection of known classes: {:.2f}'.format(acc_det_test))
	print('Detection of novel class: {:.2f}\n'.format(acc_det_novel))

	rv = random.RandomState(seed=1)
	ind_novel = np.arange(Z_novel.shape[0])

	rv.shuffle(ind_novel)

	fuse_size = 10

	# class_id = 6
	for class_id in range(18):
		ind_n = np.where(y_test.argmax(axis=1)==n)[0]
		ind_class_test = np.arange(Z_test[ind_n].shape[0])
		rv.shuffle(ind_class_test)
		Z_test_class_split = np.split(Z_test[ind_n[ind_class_test]], Z_test[ind_n].shape[0]//fuse_size)
		Z_test_fused = np.vstack([a.mean(axis=0) for a in Z_test_class_split])
		# Z_test_fused = np.vstack([np.median(a, axis=0) for a in Z_test_class_split])
		acc_det_test = 100 - 100*((Z_test_fused > thresh).all(axis=1)).mean()
		print('Detection of class {}: {:.2f}'.format(class_id, acc_det_test))

	for fuse_size in [2, 5, 10, 20, 25, 50]:
		print('\nFuse size = {}'.format(fuse_size))
		Z_novel_split = np.split(Z_novel[ind_novel], Z_novel.shape[0]//fuse_size)
		Z_novel_fused = np.vstack([a.mean(axis=0) for a in Z_novel_split])
		# Z_novel_fused = np.vstack([np.median(a, axis=0) for a in Z_novel_split])
		acc_det_novel = 100*(Z_novel_fused > thresh).all(axis=1).mean()
		print('Detection of novel class: {:.2f}'.format(acc_det_novel))


	prob = 0.2
	probs = np.arange(0.01, 0.3, 0.01)
	num = probs.size
	det_in = np.zeros([num])
	det_out = np.zeros([num])
	acc_in = np.zeros([num])
	for i in range(num):
		prob = probs[i]
		for n in range(num_classes):
			ind_n = np.where(y_train.argmax(axis=1)==n)[0]
			K = np.ceil(prob*ind_n.size).astype(np.int)
			Z_n = Z_train[ind_n, n]
			ind_top_k = np.argpartition(Z_n, -K)[-K:]
			ind_top_k_sorted = ind_top_k[np.argsort(Z_n[ind_top_k])]
			thresh[n] = Z_n[ind_top_k_sorted[0]]
			# thresh[n] = Z_train[ind_n, n].max()

		# acc_det_test = 100*(Z_test <= thresh).any(axis=1).mean()

		acc_det_test = 100 - 100*((Z_test > thresh).all(axis=1)).mean()
		acc_det_novel = 100*(Z_novel > thresh).all(axis=1).mean()

		ind_det_test = np.where((Z_test <= thresh).any(axis=1))[0]
		label_z = Z_test_full[ind_det_test].argmin(axis=1)
		ind_correct_z = np.where(label_z==label_act[ind_det_test])[0] 
		test_acc_z = 100.*ind_correct_z.size / num_test
		print('\nTest accuracy: {:.2f}% \n Z_test acc: {:.2f}%'.format(test_acc, test_acc_z))
		print('Detection of known classes: {:.2f}'.format(acc_det_test))
		print('Detection of novel class: {:.2f}\n'.format(acc_det_novel))

		det_in[i] = acc_det_test
		det_out[i] = acc_det_novel
		acc_in[i] = test_acc_z

	plt.figure(figsize=[6,3])
	plt.plot(probs, det_out, label='Outlier detection accuracy')
	plt.plot(probs, det_in, label='Inlier detection accuracy')
	plt.ylabel('Detection accuracy', fontsize=15)
	plt.xlabel(r'$p_m$', fontsize=17)
	plt.grid(True)
	plt.xticks(fontsize=13)
	plt.yticks(fontsize=13)
	plt.ylim([0, 100])
	plt.xlim([0, 0.3])
	plt.legend(bbox_transform=plt.gcf().transFigure, framealpha=1,fontsize=15)
	plt.savefig('det_single.pdf', format='pdf', dpi=1000, bbox_inches='tight')

	# ind_novel_pred = Z_novel.argmin(axis=1)
	# acc_det_novel = 0
	# for i in range(Z_novel.shape[0]):
	# 	acc_det_novel += (Z_novel[i, ind_novel_pred[i]] > thresh[ind_novel_pred[i]])
	# acc_det_novel = 100*acc_det_novel/Z_novel.shape[0]
	# print('Novel acc 3 = {:.2f}'.format(acc_det_novel))

	# ind_pred = Z_test.argmin(axis=1)
	# acc_det_test = 0
	# for i in range(Z_test.shape[0]):
	# 	acc_det_test += (Z_test[i, ind_pred[i]] < thresh[ind_pred[i]])
	# acc_det_test = 100*acc_det_test/Z_test.shape[0]

	# print((Z_novel > thresh).mean(axis=1))

	# acc_z_i = np.zeros([num_classes])
	# for class_idx in range(num_classes):
	# 	idx_inclass = np.where(label_act==class_idx)[0]
	# 	ind_correct = np.where(label_z[idx_inclass]==label_act[idx_inclass])[0] 
	# 	acc_z_i[class_idx] = 100*ind_correct.size / idx_inclass.size

	from IPython import embed; embed()
	ipdb.set_trace()

	'''
	from IPython import embed; embed()
	ipdb.set_trace()

	num_classes_plot = 18

	plt.figure(figsize=[15, 6])
	num_rows = 4
	num_cols = 5
	for i in range(num_classes_plot):
		plt.subplot(num_rows, num_cols, i+1)
		plt.hist(Z_novel[:, i], density=True, bins=15)
		plt.axvline(thresh[i], color='r', linestyle='dashed', linewidth=1)
		plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
		plt.title('Z_{}'.format(i), fontsize=12)
	plt.suptitle('Novel images')	

	label_test = label_act.copy()


	for j in range(2):
		idx_inclass = np.where(label_test==j)[0]
		Z_j = Z_test[idx_inclass]

		plt.figure(figsize=[15, 6])
		num_rows = 4
		num_cols = 5
		for i in range(num_classes_plot):
			plt.subplot(num_rows, num_cols, i+1)
			plt.hist(Z_j[:, i], density=True, bins=15)
			plt.title('Z_{}'.format(i), fontsize=12)
			plt.axvline(thresh[i], color='r', linestyle='dashed', linewidth=1)
			plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 
		plt.suptitle('Images of class {}'.format(j))

	j=6
	idx_inclass = np.where(label_test==j)[0]
	Z_j = Z_test[idx_inclass]

	plt.figure(figsize=[15, 6])
	num_rows = 4
	num_cols = 5
	for i in range(num_classes_plot):
		plt.subplot(num_rows, num_cols, i+1)
		plt.hist(Z_j[:, i], density=True, bins=15)
		plt.title('Z_{}'.format(i), fontsize=12)
		plt.axvline(thresh[i], color='r', linestyle='dashed', linewidth=1)
		plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 
	plt.suptitle('Images of class {}'.format(j))		

	z_bias = np.array([100*(label_z==i).mean() for i in np.arange(100)])
	plt.figure()
	plt.plot(z_bias)
	plt.xlabel('Classes')
	plt.title('No of times each class is predicted by decision rule')

	plt.figure()
	plt.plot(Z_test.mean(axis=0))
	plt.xlabel('Classes')
	plt.title('Mean value of decision statistic for each class')



	# k = 100
	# Z_top_k, evals_cov, snr_cov = Z_calc_smallest_k(logits_test, mu_train, cov_train_diag, k)


	# layer_name = 'dense_1'
	# # layer_name = 'dense_2'
	# # layer_name = 'dense_3'
	# # layer_name = 'dense_5'
	# # layer_name = 'dense_35'
	# intermediate_layer_model = Model(inputs=densenet.input,
	#                                  outputs=densenet.get_layer(layer_name).output)
	# features_test = intermediate_layer_model.predict(x_test, batch_size=batch_size)
	# features_train = intermediate_layer_model.predict(x_train, batch_size=batch_size)
	
	# for j in range(2):
	# 	# w_proj = vh_test_class[j][:np.int(n_comp_test_class[j])].dot(weight)
	# 	# w_proj = vh_test_class[j][:10].dot(weight)
	# 	idx_inclass = np.where(label_test==j)[0]
	# 	Z_j = Z_top_k[idx_inclass, :]

	# 	plt.figure(figsize=(15,2))
	# 	for i in range(5):
	# 		plt.subplot(1, 5, i+1)
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
	# 		plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 
	# 		# plt.ylim([0, 5000])

	# 	plt.suptitle('Smallest-{} decision rule for images of class {})'.format(k, j))	

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


	num_classes_plot = 18
	plt.figure(figsize=[15, 6])
	num_rows = 4
	num_cols = 5
	for i in range(num_classes_plot):
		plt.subplot(num_rows, num_cols, i+1)
		plt.bar(np.arange(num_neurons), height = mu_train[i]- mu_train[i].min())   
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

	novel_classes = np.array([0])
	# novel_classes = np.array([1])
	# novel_classes = np.array([2])
	# novel_classes = np.array([3])
	# novel_classes = np.array([4])
	# novel_classes = np.array([1, 7])
	# novel_classes = np.array([0, 1])
	# novel_classes = np.array([0, 1, 2])
	# novel_classes = np.array([0, 1, 2, 3])
	# novel_classes = np.array([0, 1, 2, 3, 4])


	exp_dir = '/home/rfml/wifi/experiments/exp19'
	# exp_dir = '/home/rfml/wifi/experiments/exp19_S1'
	# exp_dir = '/home/rfml/wifi/experiments/exp19_S2'
	# exp_dir = '/home/rfml/wifi/experiments/exp100_S1'
	# exp_dir = '/home/rfml/wifi/experiments/exp100_S2'
	# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Av2'
	# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2'
	# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2'
	# exp_dir = '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2'

	sample_rate = 20
	preprocess_type = 1
	sample_duration = 16

	noise = False
	snr_train = 30
	snr_test = 500

	channel = False
	diff_day = False
	num_ch_train = 1
	num_ch_test = 0
	beta = 2
	seed = 0

	if channel is True:
		data_format = 'dd-{:}-snr-{:.0f}-{:.0f}-b-{:.0f}-n-{:}-{:}-{:.0f}-pp-{:.0f}-fs-{:.0f}-s-{:}'.format(int(diff_day), snr_train, snr_test, 100*beta, num_ch_train, num_ch_test, sample_duration, preprocess_type, sample_rate, seed)
	elif noise is True:
		data_format = 'snr-{:.0f}-{:.0f}-l-{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(snr_train, snr_test, sample_duration, preprocess_type, sample_rate)
	else:
		data_format = '{:.0f}-pp-{:.0f}-fs-{:.0f}'.format(sample_duration, preprocess_type, sample_rate)

	outfile = exp_dir + '/sym-' + data_format + '.npz'
	data_format = 'l-' + data_format
	np_dict = np.load(outfile)
	x_train = np_dict['arr_0']
	y_train = np_dict['arr_1']
	x_test = np_dict['arr_2']
	y_test = np_dict['arr_3']
	num_classes = y_test.shape[1]

	print('\n-----------------\nOriginal data\n-----------------')
	print('x_train.shape = {}'.format(x_train.shape))
	print('y_train.shape = {}'.format(y_train.shape))
	print('x_test.shape = {}'.format(x_test.shape))
	print('y_test.shape = {}'.format(y_test.shape))
	ind_novel = np.empty([0], dtype=np.int)
	for n in novel_classes:
		ind_n = np.where(y_train.argmax(axis=1)==n)[0]
		print('ind_{}.shape = {}'.format(n, ind_n.shape))
		ind_novel = np.concatenate((ind_novel, ind_n))
		data_format = '-{}'.format(n) + data_format
	data_format = 'novel' + data_format

	print('ind_novel.shape = {}'.format(ind_novel.shape))


	ind_novel_test = np.empty([0], dtype=np.int)
	for n in novel_classes:
		ind_n = np.where(y_test.argmax(axis=1)==n)[0]
		print('ind_{}.shape = {}'.format(n, ind_n.shape))
		ind_novel_test = np.concatenate((ind_novel_test, ind_n))


	x_novel = np.concatenate((x_train[list(ind_novel)], x_test[list(ind_novel_test)]), axis=0)

	mask = np.ones(x_train.shape[0], dtype=bool)
	mask[list(ind_novel)] = False
	x_train = x_train[mask]
	y_train = y_train[mask]

	mask = np.ones(x_test.shape[0], dtype=bool)
	mask[list(ind_novel_test)] = False
	x_test = x_test[mask]
	y_test = y_test[mask]


	mask = np.ones(num_classes, dtype=bool)
	mask[list(novel_classes)] = False
	y_train = y_train[:, mask]
	y_test = y_test[:, mask]

	print('\n-----------------\nData without classes {}\n-----------------'.format(novel_classes))
	print('x_train.shape = {}'.format(x_train.shape))
	print('y_train.shape = {}'.format(y_train.shape))
	print('x_test.shape = {}'.format(x_test.shape))
	print('y_test.shape = {}'.format(y_test.shape))
	for n in novel_classes:
		ind_n = np.where(y_train.argmax()==n)[0]
		print('ind_{}.shape = {}'.format(n, ind_n.shape))

	dict_wifi = {}
	dict_wifi['x_train'] = x_train
	dict_wifi['y_train'] = y_train
	dict_wifi['x_test'] = x_test
	dict_wifi['y_test'] = y_test
	dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]
	dict_wifi['x_novel'] = x_novel

	# Checkpoint path
	checkpoint = exp_dir + '/ckpt-' + data_format + '.h5'

	print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')
	train_output, summary = test(dict_wifi, checkpoint_in=checkpoint)
	print('-----------------------\nExperiment:\n' + exp_dir + '\n-----------------------')