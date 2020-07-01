#!/bin/bash 

# # ./shell2.sh | tee shell2_out.txt
# pid=436088

# # check if the process is finished or not
# while [ -d /proc/$pid ] ; do
#     sleep 1
# done

#export CUDA_VISIBLE_DEVICES="1"


COMMAND="export path_to_config=/home/rfml/wifi-rebuttal/wifi-fingerprinting-journal/configs_train.json"
eval $COMMAND

COMMAND="export path_to_data=/home/rfml/wifi/experiments/exp19"
eval $COMMAND

COMMAND="python cfo_channel_training.py"  
echo $COMMAND
eval $COMMAND

# COMMAND="python test_attacks.py  \
# --dataset=$dataset  \
# --attack_method=BIM  \
# --loss_function=$loss_function" 
# echo $COMMAND
# eval $COMMAND




