#!/bin/bash 

# # ./shell2.sh | tee shell2_out.txt
# pid=436088

# # check if the process is finished or not
# while [ -d /proc/$pid ] ; do
#     sleep 1
# done

#export CUDA_VISIBLE_DEVICES="1"


COMMAND="python cfo_channel_training_simulations.py"  
echo $COMMAND
eval $COMMAND

# COMMAND="python test_attacks.py  \
# --dataset=$dataset  \
# --attack_method=BIM  \
# --loss_function=$loss_function" 
# echo $COMMAND
# eval $COMMAND




