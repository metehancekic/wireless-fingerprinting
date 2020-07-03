# Robust Wireless Fingerprinting: Generalizing Across Space and Time #


This repository contains scripts to simulate the effect of channel and CFO variations on wireless fingerprinting using complex-valued CNNs. This repo also has simulation-based dataset based on models of some typical nonlinearities. It includes augmentation techniques, estimation techniques, and combinations of the two. The repository consists of 4 folders; namely, cxnn, preproc, tests, data, and a number of scripts outside of these folders (in the main folder). 

## Simulated Dataset #

We have created a simulation-based WiFi dataset based on models of some typical nonlinearities. We implement two different kinds of circuit-level impairments: I/Q imbalance and power amplifier nonlinearity. Training dataset consists of 200 signals per device for 19 devices (classes). The validation and test sets contain 100 signals per device. Overall, the dataset contains 3800 signals for training, 1900 signals for validation and 1900 signals for the test set.

Further details can be found in our [paper](https://arxiv.org/pdf/2002.10791.pdf):

## Module Structure #

```
project
│   README.md
│   cfo_channel_training_simulations.py     Training code for all the experiments
│   cfo_channel_testing_simulations.py      Testing code from checkpoints
│   configs_train.json          All hyper parameters for training
│   configs_test.json           All hyper parameters for testing
│   simulators.py               All simulations (CFO, channel, residuals, etc) as functions
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

## Prerequisites #

Since the implementation of complex valued neural networks is done on Keras with Theano backend, the following modules are needed to be installed to be able to run experiments.

> python                    2.7.1\
> numpy                     1.15.4\
> matplotlib                2.2.3\
> Keras                     2.2.4\
> Theano                    1.0.3\
> tqdm                      4.28.1\
> scipy                     1.1.0\
> resampy                   0.2.1\
> ipdb                      0.11 

CUDA and cuDNN versions:

> CUDA                  	9.0.176\
> cuDNN                     7.3.1

## Building the environment and running the code #

```bash
git clone https://github.com/metehancekic/wireless-fingerprinting.git
cd wireless-fingerprinting/
```

We strongly recommend to install miniconda and create a virtual environment and run the following commands.

```bash
conda create -n cxnn2 python=2.7
conda activate cxnn2
pip install -r requirements.txt 
conda install mkl-service
conda install -c conda-forge resampy
```

CFO and channel simulation parameters can be set in "configs_train.json" and "configs_test.json" for training and testing codes respectively. The code can then be run using: 


```bash
python cfo_channel_training_simulations.py
python cfo_channel_testing_simulations.py
python fingerprint_wifi_reim_mag.py
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
