import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np 
import os



# graph = 'cfo_channel'
# graph = 'channel'
# graph = 'cfo_channel_variance'
# graph = 'cfo_channel_num_aug'
graph = 'channel_eq'

signals_directory = "/home/rfml/wifi/scripts/images/"
if not os.path.exists(signals_directory):
	os.makedirs(signals_directory)

if graph == 'channel':
	
	Accuracies_10 = np.array([15.79, 39.16, 45.63,	71.53, 80.37, 75.84, 90.26, 88.79, 91.84, 93.32, 93.74, 93.05])
	Accuracies_0 = np.array([5.05, 6.32, 8.53, 7.37, 10.42, 12.84, 11.63, 12.05, 23.32, 17.11, 25.05, 28.0])
	Number_days = np.array([1,2,3,4,5,6,7,8,9,10,15,20])

	fig = plt.figure()

	ax = plt.gca()

	ax.plot(Number_days, Accuracies_10, linestyle='--', marker='o', color='b', label='10 Augmentations')
	ax.plot(Number_days, Accuracies_0, linestyle='--', marker='o', color='r', label='No Augmentation')
	ax.legend(loc='upper left')
	plt.ylim(0, 100)
	ax.set_xticks([1,2,3,4,5,6,7,8,9,10,15,20], minor=False)
	ax.set_yticks([20, 40, 60, 80, 100], minor=False)

	ax.xaxis.grid(True, which='major')
	ax.yaxis.grid(True, which='major')

	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	fig_name = os.path.join(signals_directory, "DiffDaysCh-Acc" + '.pdf')
	plt.title(" Accuracy vs #(days data collected) (Ch)")

	fig.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')

elif graph == 'cfo_channel':

	Accuracies_10_2 = np.array([4.84, 14.74, 18.37, 32.32, 28.32, 48.89, 47, 41.16, 51.95, 50.26, 53.42, 61.89])
	Accuracies_20_2 = np.array([11.89, 15, 21.05, 29.95, 48.11, 50.11, 49.11, 48.11, 52.37, 59.11, 61.89, 65.11])
	Accuracies_0_0 = np.array([5.26, 0.0, 1.84, 6.37, 5.26, 3.26, 0.0, 0.0, 5.26, 4.63, 4.89, 5.05])
	Number_days = np.array([1,2,3,4,5,6,7,8,9,10,15,20])

	fig = plt.figure()

	ax = plt.gca()

	ax.plot(Number_days, Accuracies_10_2, linestyle='--', marker='o', color='b', label='10*2 Augmentations')
	ax.plot(Number_days, Accuracies_20_2, linestyle='--', marker='o', color='r', label='20*2 Augmentations')
	ax.plot(Number_days, Accuracies_0_0, linestyle='--', marker='o', color='k', label='No Augmentation')
	ax.legend(loc='upper left')
	plt.ylim(0, 100)
	ax.set_xticks([1,2,3,4,5,6,7,8,9,10,15,20], minor=False)
	ax.set_yticks([20, 40, 60, 80, 100], minor=False)

	ax.xaxis.grid(True, which='major')
	ax.yaxis.grid(True, which='major')

	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	fig_name = os.path.join(signals_directory, "DiffDaysChCfo-Acc" + '.pdf')
	plt.title(" Accuracy vs #(days data collected) (Ch + CFO)")

	fig.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')

elif graph == 'cfo_channel_comp':

	Accuracies_20_0 = np.array([14.95, 29.11, 38.89, 56.16, 48.05, 71.21, 72.16, 72.58, 73.95, 83.58, 88.84, 91.74])
	Accuracies_20_1 = np.array([5.84, 27.89, 28.79, 42.53, 41.00, 46.21, 57.16, 60.00, 56.68, 56.95, 74.11, 71.63])
	# Accuracies_0_0 = np.array([5.26, 0.0, 1.84, 6.37, 5.26, 3.26, 0.0, 0.0, 5.26, 4.63, 4.89, 5.05])
	Number_days = np.array([1,2,3,4,5,6,7,8,9,10,15,20])

	fig = plt.figure()

	ax = plt.gca()

	ax.plot(Number_days, Accuracies_20_0, linestyle='--', marker='o', color='b', label='20-0 Augmentations - CFO Comp ')
	ax.plot(Number_days, Accuracies_20_1, linestyle='--', marker='o', color='r', label='20-1 Augmentations')
	# ax.plot(Number_days, Accuracies_0_0, linestyle='--', marker='o', color='k', label='No Augmentation')
	ax.legend(loc='upper left')
	plt.ylim(0, 100)
	ax.set_xticks([1,2,3,4,5,6,7,8,9,10,15,20], minor=False)
	ax.set_yticks([20, 40, 60, 80, 100], minor=False)

	ax.xaxis.grid(True, which='major')
	ax.yaxis.grid(True, which='major')

	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	fig_name = os.path.join(signals_directory, "DiffDaysChCfo-Acc-with-cfo-comp" + '.pdf')
	plt.title(" Accuracy vs #(days data collected) (Ch + CFO)")

	fig.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')

elif graph == 'cfo_channel_variance':

	# Accuracies_20_0 = np.array([14.95, 29.11, 38.89, 56.16, 48.05, 71.21, 72.16, 72.58, 73.95, 83.58, 88.84, 91.74])
	# Accuracies_20_1 = np.array([5.84, 27.89, 28.79, 42.53, 41.00, 46.21, 57.16, 60.00, 56.68, 56.95, 74.11, 71.63])
	# Accuracies_0_0 = np.array([5.26, 0.0, 1.84, 6.37, 5.26, 3.26, 0.0, 0.0, 5.26, 4.63, 4.89, 5.05])
	days_accuracies = np.array([[4.68, 7.11, 4.21, 5.11, 12.74],
					   [15.11, 6.53, 13.11, 18.00, 24.11],
					   [20.16, 8.63, 20.11, 19.74, 17.79],
					   [30.42, 22.58, 27.53, 25.47, 17.42],
					   [37.26, 23.16, 28.32, 30.89, 20.00],
					   [43.16, 29.21, 44.42, 41.63, 31.68],
					   [41.00, 40.47, 42.05, 33.63, 37.79],
					   [45.53, 31.00, 52.00, 38.32, 33.84],
					   [45.32, 48.68, 51.89, 45.32, 41.84],
					   [53.21, 48.84, 48.00, 48.89, 36.37],
					   [60.47, 56.16, 64.84, 37.42, 54.63],
					   [55.79, 53.63, 64.37, 50.37, 47.53]])

	days_accuracies_10 = np.array([[5.58, 11.37, 16.79, 7.00, 15.00],
								   [21.53, 11.84, 17.11, 23.26, 25.11],
								   [30.21, 12.95, 27.26, 22.32, 23.11],
								   [34.74, 32.47, 35.95, 37.05, 33.05],
								   [48.11, 30.74, 41.26, 36.11, 41.79],
								   [65.79, 50.74, 48.53, 47.21, 40.42],
								   [56.26, 51.95, 62.84, 52.11, 46.95],
								   [62.42, 62.95, 61.79, 66.00, 62.63],
								   [62.63, 70.58, 72.16, 51.68, 56.42],
								   [78.21, 50.68, 71.26, 60.63, 65.47],
								   [77.95, 80.21, 82.47, 72.00, 70.53],
								   [82.47, 83.68, 88.11, 80.21, 80.74]])

	days_accuracies_comp = np.array([[5.68, 10.21, 17.95, 6.32, 18.63],
									 [26.84, 17.21, 14.58, 20.11, 36.89],
									 [31.32, 25.00, 33.05, 31.11, 40.58],
									 [49.68, 30.84, 42.84, 43.47, 54.84],
									 [46.74, 44.42, 53.42, 51.42, 55.26],
									 [57.89, 48.68, 64.95, 50.47, 58.53],
									 [74.16, 64.58, 74.00, 56.05, 62.53],									 
									 [63.63, 48.89, 66.74, 62.32, 63.21],
									 [64.79, 61.42, 75.74, 68.11, 61.21],
									 [75.16, 65.26, 75.84, 70.63, 71.63],
									 [78.53, 82.32, 83.74, 71.47, 68.68],
									 [74.47, 84.63, 88.79, 73.16, 81.32]])

	Number_days = np.array([1,2,3,4,5,6,7,8,9,10,15,20])
	# Number_days = np.array([1,2,3,4,5,6,7,8,9,10,15])
	# [82.47, 83.68, 88.11, -, -]

	mean_acc = np.mean(days_accuracies, axis=1)
	max_acc = np.max(days_accuracies, axis=1)
	min_acc = np.min(days_accuracies, axis=1)

	mean_acc_10 = np.mean(days_accuracies_10, axis=1)
	max_acc_10 = np.max(days_accuracies_10, axis=1)
	min_acc_10 = np.min(days_accuracies_10, axis=1)

	mean_acc_comp = np.mean(days_accuracies_comp, axis=1)
	max_acc_comp = np.max(days_accuracies_comp, axis=1)
	min_acc_comp = np.min(days_accuracies_comp, axis=1)


	fig = plt.figure()

	ax = plt.gca()

	ax.errorbar(Number_days, mean_acc, linestyle='--', marker='o', capsize=5, capthick=2, color='b', ecolor = 'r', label = '10-1 Augmentations')
	ax.errorbar(Number_days, mean_acc_10, linestyle='--', marker='o', capsize=5, capthick=2, color='k', ecolor = 'm', label = '10-10 Augmentations')
	ax.errorbar(Number_days, mean_acc_comp, linestyle='--', marker='o', capsize=5, capthick=2, color='g', ecolor = 'y', label = '10-0 Augmentations - CFO Compensation')
	# ax.plot(Number_days, mean_acc, linestyle='--', marker='o', color='b', label='10-1 Augmentations')
	# ax.errorbar(Number_days, mean_acc, yerr = [min_acc, max_acc], lw=2, capsize=5, capthick=2, color='r')
	# ax.plot(Number_days, Accuracies_20_1, linestyle='--', marker='o', color='r', label='20-1 Augmentations')
	# ax.plot(Number_days, Accuracies_0_0, linestyle='--', marker='o', color='k', label='No Augmentation')
	ax.legend(loc='upper left')
	plt.ylim(0, 100)
	ax.set_xticks([1,2,3,4,5], minor=False)
	ax.set_yticks([20, 40, 60, 80, 100], minor=False)

	ax.xaxis.grid(True, which='major')
	ax.yaxis.grid(True, which='major')

	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	fig_name = os.path.join(signals_directory, "DiffDaysChCfo-Acc-Var" + '.pdf')
	plt.title(" Accuracy vs #(days data collected) (Ch + CFO)")

	fig.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')

elif graph == 'cfo_channel_num_aug':


	aug_accuracies = np.array([[5.26, 5.32, 1.68, 1.32, 3.84],
								[4.05, 18.68, 21.11, 12.37, 9.79],
								[56.89, 41.53, 45.05, 44.58, 57.53],
								[57.21, 62.89, 77.47, 46.32, 60.32],
								[60.63, 65.11, 70.05, 55.84, 64.58],
								[78.21, 82.58, 77.16, 66.11, 79.11],
								[79.53, 77.68, 73.53, 79.21, 80.79],
								[80.05, 80.05, 80.05, 80.05, 80.05],
								[82.47, 83.68, 88.11, 80.21, 80.74]])

	

	Number_augs = np.array([1,2,3,4,5,6,7,8,10])
	# Number_days = np.array([1,2,3,4,5,6,7,8,9,10,15])
	# [82.47, 83.68, 88.11, -, -]

	mean_acc = np.mean(aug_accuracies, axis=1)
	max_acc = np.max(aug_accuracies, axis=1)
	min_acc = np.min(aug_accuracies, axis=1)


	fig = plt.figure()

	ax = plt.gca()

	ax.errorbar(Number_augs, mean_acc, linestyle='--', marker='o', capsize=5, capthick=2, color='b', ecolor = 'r', label = '20 days training, 1 day testing')
	
	ax.legend(loc='upper left')
	plt.ylim(0, 100)
	ax.set_xticks([1,2,3,4,5], minor=False)
	ax.set_yticks([20, 40, 60, 80, 100], minor=False)

	ax.xaxis.grid(True, which='major')
	ax.yaxis.grid(True, which='major')

	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	fig_name = os.path.join(signals_directory, "DiffDaysChCfo-Acc-Aug" + '.pdf')
	plt.title(" Accuracy vs #Augmentation for both (Ch + CFO)")

	fig.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')

elif graph == 'channel_eq':


	ch_eq = np.array([[33.79, 23.68, 29.47, 33.89, 21.74],
							   [42.26, 49.05, 51.63, 32.21, 39.00],
							   [54.89, 41.89, 61.00, 57.00, 44.11],
							   [57.11, 46.26, 65.95, 53.05, 54.11],
							   [63.16, 51.68, 57.74, 55.68, 62.37],
							   [56.68, 47.16, 67.37, 50.42, 55.95],  # 1
							   [50.63, 51.53, 67.84, 63.74, 58.63],  # 1
							   [67.37, 58.68, 62.42, 57.32, 57.26],  # 3
							   [63.32, 62.95, 59.53, 57.05, 75.00],  # 3,4
							   [65.68, 65.32, 69.47, 66.84, 70.16],  # 3
							   [67.68, 68.74, 61.53, 67.53, 68.05],
							   [74.21, 69.79, 71.84, 61.05, 78.89]])

	ch_augch = np.array([[23.32,22.74, 25.95, 26.05, 19.42],
						[45.68, 57.16, 61.32, 48.00, 40.37],
						[73.11, 63.21, 72.68, 52.42, 70.89],
						[80.32, 74.53, 81.47, 78.95, 91.89],
						[80.32, 88.00, 87.00, 88.64, 86.79],
						[92.37, 82.84, 88.63, 83.21, 94.32],
						[92.74, 84.47, 93.21, 91.32, 94.05],
						[96.00, 94.95, 91.11, 96.32, 96.05],
						[95.00, 91.89, 94.11, 98.63, 93.11],
						[95.42, 90.68, 97.84, 96.16, 97.26],
						[93.47, 98.37, 98.74, 97.95, 92.74],
						[95.16, 92.16, 97.74, 98.68, 84.11]])  # 5

	ch_eq_augch = np.array([])  # 5

	

	Number_days = np.array([1,2,3,4,5,6,7,8,9,10,15,20])
	# Number_days = np.array([1,2,3,4,5,6,7,8,9,10,15])
	# [82.47, 83.68, 88.11, -, -]

	mean_ch_eq = np.mean(ch_eq, axis=1)
	max_ch_eq = np.max(ch_eq, axis=1)
	min_ch_eq = np.min(ch_eq, axis=1)

	mean_ch_augch = np.mean(ch_augch, axis=1)
	max_ch_augch = np.max(ch_augch, axis=1)
	min_ch_augch = np.min(ch_augch, axis=1)


	fig = plt.figure()

	ax = plt.gca()

	ax.errorbar(Number_days, mean_ch_eq, linestyle='--', marker='o', capsize=5, capthick=2, color='r', ecolor = 'r', label = 'Channel augmentation (20)')
	ax.errorbar(Number_days, mean_ch_augch, linestyle='--', marker='o', capsize=5, capthick=2, color='b', ecolor = 'r', label = 'Equalization')
	
	ax.legend(loc='upper left')
	plt.ylim(0, 100)
	ax.set_xticks(Number_days, minor=False)
	ax.set_yticks([20, 40, 60, 80, 100], minor=False)

	ax.xaxis.grid(True, which='major')
	ax.yaxis.grid(True, which='major')

	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	fig_name = os.path.join(signals_directory, "Eq_vs_aug_ch" + '.pdf')
	plt.title(" Equalization vs Augmentation (Channel)")

	fig.savefig(fig_name, format='pdf', dpi=1000, bbox_inches='tight')
