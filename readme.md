### Robust Wireless Fingerprinting: Generalizing Across Space and Time #

This repository contains scripts to simulate the effect of channel and CFO variations on wireless fingerprinting using complex-valued CNNs. It includes augmentation techniques, estimation techniques, and combinations of the two. The repository consists of 3 folders; namely, cxnn, preproc, tests and a number of scripts outside of these folders (in the main folder). 

## Module Structure #

```
project
│   README.md
│   cfo_channel_training.py     Training code for all the experiments
│   cfo_channel_testing.py      Testing code from checkpoints
│   config_cfo_channel.json     All hyper parameters for the experiment
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
    │   test_aug_analysis.py        Signal processing tools (Fading models, etc)   
    │   visualize_offset.py         Preprocessing tools (Equalization, etc)   
```

## Prerequisites #

Since the implementation of complex valued neural networks is done on Keras with backend Theano, the following modules are needed to be installed to be able to run experiments.

> python                    2.7.1\
> numpy                     1.15.4\
> matplotlib                2.2.3\
> Keras                     2.2.4\
> Theano                    1.0.3\
> tqdm                      4.28.1\
> scikit-learn              0.20.3\
> scipy                     1.1.0\
> resampy                   0.2.1\
> ipdb                      0.11 

## Running the code #
After downloading the prerequisites, create environment variables named as 'path_to_data' and 'path_to_config' which have the paths to the dataset and the configuration file, e.g.,

```bash
export path_to_data='<path-to-data>'
export path_to_config='<path-to-config>'
```

CFO and channel simulation parameters can be set in config_cfo_channel.json. The code can then be run using: 


```bash
python cfo_channel_training.py
python fingerprint_wifi_reim_mag.py
```

## Complex-valued CNNs

Complex layers are from Trabelsi et al, "Deep Complex Networks", *ICLR 2018* (MIT license), with the addition of cxnn/complexnn/activations.py which contains the ModReLU activation function.




