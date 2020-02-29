#!/bin/bash

# time python store_wifi.py -fs 200 -ch -snr 0 -b 0.5 -pp 1 -dd
# time python store_wifi.py -fs 200 -ch -snr 0 -b 0.5 -pp 3 -dd
# time python store_wifi.py -fs 200 -ch -snr 0 -b 0.5 -pp 1
# time python store_wifi.py -fs 200 -ch -snr 0 -b 0.5 -pp 3
# time python store_wifi.py -fs 200 -ch -snr 15 -b 0.5 -pp 1 -dd
# time python store_wifi.py -fs 200 -ch -snr 15 -b 0.5 -pp 3 -dd
# time python store_wifi.py -fs 200 -ch -snr 15 -b 0.5 -pp 1
# time python store_wifi.py -fs 200 -ch -snr 15 -b 0.5 -pp 3
# time python store_wifi.py -fs 200 -ch -snr 5 -b 0.5 -pp 1 -dd
# time python store_wifi.py -fs 200 -ch -snr 5 -b 0.5 -pp 3 -dd
# time python store_wifi.py -fs 200 -ch -snr 5 -b 0.5 -pp 1
# time python store_wifi.py -fs 200 -ch -snr 5 -b 0.5 -pp 3

# time python store_wifi_20.py -ch -s 0 -d '/home/rfml/wifi/experiments/exp19'
# time python store_wifi_20.py -ch -s 1 -d '/home/rfml/wifi/experiments/exp19'
# time python store_wifi_20.py -ch -s 2 -d '/home/rfml/wifi/experiments/exp19'
# time python store_wifi_20.py -ch -s 3 -d '/home/rfml/wifi/experiments/exp19'

# time python store_wifi_20.py -ch -s 0 -d '/home/rfml/wifi/experiments/exp19' -b 2
# time python store_wifi_20.py -ch -s 1 -d '/home/rfml/wifi/experiments/exp19' -b 2
# time python store_wifi_20.py -ch -s 2 -d '/home/rfml/wifi/experiments/exp19' -b 2
# time python store_wifi_20.py -ch -s 3 -d '/home/rfml/wifi/experiments/exp19' -b 2


time python store_wifi.py -fs 100 -pp 1 
time python store_wifi.py -fs 200 -pp 1 
time python store_wifi.py -fs 20 -pp 2