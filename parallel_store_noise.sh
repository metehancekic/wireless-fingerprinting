#!/bin/bash

time python store_wifi_20.py -n -snr_tr 5 -d '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2'
time python store_wifi_20.py -n -snr_tr 10 -d '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2'
time python store_wifi_20.py -n -snr_tr 15 -d '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2'
time python store_wifi_20.py -n -snr_tr 20 -d '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Bv2'

time python store_wifi_20.py -n -snr_tr 5 -d '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2'
time python store_wifi_20.py -n -snr_tr 10 -d '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2'
time python store_wifi_20.py -n -snr_tr 15 -d '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2'
time python store_wifi_20.py -n -snr_tr 20 -d '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Cv2'

time python store_wifi_20.py -n -snr_tr 5 -d '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2'
time python store_wifi_20.py -n -snr_tr 10 -d '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2'
time python store_wifi_20.py -n -snr_tr 15 -d '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2'
time python store_wifi_20.py -n -snr_tr 20 -d '/home/rfml/wifi/experiments/NSWC_Crane_Experiments/Test3/converted_3Dv2'


time python store_wifi_20.py -n -snr_tr 5 -snr_te 20 -d '/home/rfml/wifi/experiments/exp19'
time python store_wifi_20.py -n -snr_tr 10 -snr_te 20 -d '/home/rfml/wifi/experiments/exp19'
time python store_wifi_20.py -n -snr_tr 15 -snr_te 20 -d '/home/rfml/wifi/experiments/exp19'
time python store_wifi_20.py -n -snr_tr 20 -snr_te 20 -d '/home/rfml/wifi/experiments/exp19'
