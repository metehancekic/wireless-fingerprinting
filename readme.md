# Robust Wireless Fingerprinting: Generalizing Across Space and Time #


This repository contains scripts to simulate the effect of channel and CFO variations on wireless fingerprinting using complex-valued CNNs. This repo also has simulation-based dataset based on models of some typical nonlinearities. It includes augmentation techniques, estimation techniques, and combinations of the two. The repository consists of 4 folders; namely, cxnn, preproc, tests, data, and a number of scripts outside of these folders (in the main folder). 

## Simulated Dataset #

We have created a simulation-based WiFi dataset based on models of some typical nonlinearities. We implement two different kinds of circuit-level impairments: I/Q imbalance and power amplifier nonlinearity. Training dataset consists of 200 signals per device for 19 devices (classes). The validation and test sets contain 100 signals per device. Overall, the dataset contains 3800 signals for training, 1900 signals for validation and 1900 signals for the test set. The dataset can be downloaded as an npz file from [this Box link](https://ucsb.box.com/s/ddub4zlp2wbckk4l1v1785yw2aluzfru), and needs to be copied into the `data` subdirectory. 

Further details can be found in our paper at section 5e:


## Building the environment and running the code #

This repo can be installed via following command:

```bash
git clone https://github.com/metehancekic/wireless-fingerprinting.git
```

Since the implementation of complex valued neural networks is done on Keras with Theano backend, the modules inside the requirements.txt are needed to be installed to be able to run experiments. We strongly recommend to install miniconda, create a virtual environment, and run the following commands. These commands will build the environment which is necessary to run the codes in this repository.

```bash
conda create -n cxnn python=2.7
conda activate cxnn
pip install -r requirements.txt 
conda install mkl-service
conda install -c conda-forge resampy
```
For gpu usage:
```bash
conda install -c anaconda pygpu
```
With following CUDA and cuDNN versions:

> CUDA                    9.0.176\
> cuDNN                     7.3.1

The code with default parameters (without channel and CFO) can be run using: 

```bash
KERAS_BACKEND=theano python cfo_channel_training_simulations.py
KERAS_BACKEND=theano python cfo_channel_testing_simulations.py
```

Controlled experiments emulating the effect of frequency drift and channel variations is included via "experiment_setup.py" or can be explicitly called on terminal. All the hyper-parameters for these experiments are in "configs_train.json" and "configs_test.json" for training and testing codes respectively. 

For detailed information about arguments use following code:

```bash
KERAS_BACKEND=theano python cfo_channel_training_simulations.py --help
```

```
usage: cfo_channel_testing_simulations.py [-h]
                                          [-a {reim,reim2x,reimsqrt2x,magnitude,phase,re,im,modrelu,crelu}]
                                          [-phy_ch] [-phy_cfo] [-comp_cfo]
                                          [-eq_tr] [-aug_ch] [-aug_cfo] [-res]
                                          [-eq_test] [-comp_cfo_test]
                                          [-aug_ch_test] [-aug_cfo_test]

optional arguments:
  -h, --help            show this help message and exit
  -a {reim,reim2x,reimsqrt2x,magnitude,phase,re,im,modrelu,crelu}, --architecture {reim,reim2x,reimsqrt2x,magnitude,phase,re,im,modrelu,crelu}
                        Architecture

setup:
  Setup for experiments

  -phy_ch, --physical_channel
                        Emulate the effect of channel variations, default =
                        False
  -phy_cfo, --physical_cfo
                        Emulate the effect of frequency variations, default =
                        False
  -comp_cfo, --compensate_cfo
                        Compensate frequency of training set, default = False
  -eq_tr, --equalize_train
                        Equalize training set, default = False
  -aug_ch, --augment_channel
                        Augment channel for training set, default = False
  -aug_cfo, --augment_cfo
                        Augment cfo for training set, default = False
  -res, --obtain_residuals
                        Obtain residuals for both train and test set, default
                        = False
  -comp_cfo_test, --compensate_cfo_test
                        Compensate frequency of test set, default = False

test setup:
  Test Setup for experiments

  -eq_test, --equalize_test
                        Equalize test set, default = False
  -aug_ch_test, --augment_channel_test
                        Augment channel for test set, default = False
  -aug_cfo_test, --augment_cfo_test
                        Augment cfo for test set, default = False
```

Running code with different day scenario (channel, cfo):

```bash
KERAS_BACKEND=theano python cfo_channel_training_simulations.py --phsical_channel --physical_cfo --augment_channel --augment_cfo
KERAS_BACKEND=theano python cfo_channel_testing_simulations.py --phsical_channel --physical_cfo --augment_channel --augment_cfo --augment_channel_test --augment_cfo_test
```

## Module Structure #

```
project
│   README.md
│   cfo_channel_training_simulations.py     Training code for all the experiments
│   cfo_channel_testing_simulations.py      Testing code from checkpoints
│   configs_train.json          All hyper parameters for training
│   configs_test.json           All hyper parameters for testing
│   simulators.py               All simulations (CFO, channel, residuals, etc) as functions
│   experiment_setup.py         Experiment setup (use different day channel, cfo effects or not)
│
└───cxnn
│   │   models.py                   Neural network architectures
│   │   train.py                    Training function
│   │   train_network_reim_mag.py   Training function for real and complex networks
│   │ 
│   └───complexnn
│       │   complex-valued neural network implemantation codes
│       │   ...
│   
└───preproc   
│   │  fading_model.py      Signal processing tools (Fading models, etc)   
│   │  preproc_wifi         Preprocessing tools (Equalization, etc)
│
└───tests
│   │   test_aug_analysis.py        Signal processing tools (Fading models, etc)   
│   │   visualize_offset.py         Preprocessing tools (Equalization, etc)   
│
└───data
    │   simulations.npz     Simulated WiFi dataset
    │   
```

## Complex-valued CNNs

Complex layers are from Trabelsi et al, "Deep Complex Networks", *ICLR 2018* (MIT license), with the addition of cxnn/complexnn/activations.py which contains the ModReLU activation function.

## Citation

```
@article{fingerpinting2020,
  title={Robust Wireless Fingerprinting: Generalizing Across Space and Time},
  author={Cekic, Metehan and Gopalakrishnan, Soorya and Madhow, Upamanyu},
  journal={arXiv preprint arXiv:2002.10791},
  year={2020}
}
```

```
@inproceedings{fingerprinting2019globecom,
 author = {Gopalakrishnan, Soorya and Cekic, Metehan and Madhow, Upamanyu},
 booktitle = {IEEE Global Communications Conference (Globecom)},
 location = {Waikoloa, Hawaii},
 title = {Robust Wireless Fingerprinting via Complex-Valued Neural Networks},
 year = {2019}
}
```


