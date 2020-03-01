### Robust Wireless Fingerprinting: Generalizing Across Space and Time #

This repository consists of codes that simulates different channels, CFO, augmentation techniques, estimation techniques, and different combinations of them. The repository consists of 3 folders; namely, cxnn, preproc, tests and a number of scripts outside of these folders (in the main folder). 

## Module Structure #

```
project
│   README.md
│   Codes to run experiments 
│	...
│
└───cxnn
│   │   train codes
│   │   test codes
│   │   architectures for adsb and wifi
│   │   ...
│   │ 
│   └───complexnn
│       │   complex-valued neural network implemantation codes
│       │   ...
│   
└───preproc   
│   │   preprocessing codes
│   │   ...
│
└───tests
    │   neural network analysis codes
    │   ...
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
> scipy                     1.1.0

## Running the code #




