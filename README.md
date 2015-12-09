# Probabilistic-Backpropagation
Implementation in C and Theano of the method Probabilistic Backpropagation for
scalable Bayesian inference in deep neural networks.

There are two folders "c" and "theano" containing implementations of the code
in "c" and in "theano". The "c" version of the code is between 20 and 4 times
faster than the theano version depending on the neural network size and the
dataset size.  To use the "c" version go to the folder "c/PBP_net" and run the
script "compile.sh". You will need to have installed the "openblas" library for
fast numerical algebra operations and "cython". For maximum speed, we recommend
you to compile yourself the "open blas" library in your own machine. To compile
the "c" code type

$ cd c/PBP_net ; ./compile.sh

In each folder, "c" and "theano", the python script test_PBP_net.py creates a
two-hidden-layer neural network with 50 hidden units in each layer and fits a
posterior approximation using the probabilistic backpropagation (PBP) method by
doing 40 ADF passes over the training data. The data used is from the Boston
Housing dataset.  After the training, the script "test_PBP_net.py" computes and
prints the test RMSE and the test log-likelihood. To run the script type

$ python test_PBP_net.py
