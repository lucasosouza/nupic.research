Introduction
==============

This repository contains the code for experiments on dynamic sparse neural networks.

To get started, do `python runs/run_test.py`


Overview
==========

### Model 

A model class contains all code required to train and evaluate a predictive model. A model can contain one or more neural networks. As much as possible, particularities of each model should be agnostic to each neural network being used. Ideally, any torchvision (or related packages) default neural network could be used with a model. 

### Network

Networks are specific instances of neural networks. Can either be default, such as those loaded by torchvision.models, imported from a public available implementation, or customized for a particular task. 

### Utils

Several support files, which can support one or more experiments.
- Dataset: Datasets are loaders. As far as possible, should be agnostic to the dataset being loaded. The dataset name is a hyperparamenter defined in the run files. 
- Trainable: Required to integrate with Ray Tune. Intermediate class between Ray Tune and Model, calls run_one_epoch at each iteration.

### Runs

Each run file is a different experiment, conducted at some point. Stored to keep track of past runs. New runs can be modelled based on past runs. Not part of the source code. 

### Notebooks

Tests, explorations, and analysis. Not part of the source code.

### Tests

Include all related tests implemented so far.  Broken down into:
- Unit tests: Regular unit tests, to verify functionalities of models, networks, and common functions
- Blackbox tests: Tests to evaluate if the output of a model and network are as expected. For example, a network with three 100-neurons hidden layers, trained on MNIST for 100 iterations with batch size 128, must return a validation accuracy above 90%. To allow for stochasticity, it is best if blackbox tests are averaged over multiple runs (3,5 or more, depending on the computational complexity).

### Deprecated

Code no longer being used, but which might be stored temporarily during research period.

