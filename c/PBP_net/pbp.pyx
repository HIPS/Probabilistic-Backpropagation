#
# Wrapper class for the C module network.c
#

import numpy as np
cimport numpy as np

#
# Definition of the C API for the module network.c in a pxd file.
# 
# Author: Jose Miguel Hernandez Lobato
#

cdef extern from "network.h":

    ctypedef struct NETWORK:
        pass

    NETWORK *init_network(int *size_hidden_layers, int n_hidden_layers,
        int d, double *random_noise)
    void destroy_network(NETWORK * network)
    NETWORK *predict(NETWORK *network, double *x_test, double *m_test,
        double *v_test, double *v_noise, int n_datapoints, int d)
    NETWORK *one_learning_epoch(NETWORK *network, double *x, double *y,
        int n_datapoints, int d, int *permutation)
    void get_params(NETWORK *n, double *sample_w_out, double *m_w_out,
        double *v_w_out, double *m_w_hat_nat_out, double *v_w_hat_nat_out,
        double *a_w_hat_nat_out, double *b_w_hat_nat_out,
        double *a_noise_out, double *b_noise_out,
        double *a_prior_out, double *b_prior_out, int *neurons_per_layer_out)
    void get_size_params(NETWORK *n, int *size_weihts, int *n_layers)
    void set_params(NETWORK *n, double *sample_w_in, double *m_w_in,
        double *v_w_in, double *m_w_hat_nat_in, double *v_w_hat_nat_in,
        double *a_w_hat_nat_in, double *b_w_hat_nat_in, double a_noise_in,
        double b_noise_in, double a_prior_in, double b_prior_in)
    NETWORK *predict_deterministic(NETWORK *n, double *x_test, double *y_test,
        int n_datapoints, int d)

#
# Definition of the wrapper class
#

cdef class PBP_net:
    
    # The structure encoding the network

    cdef NETWORK* _c_NETWORK

    def __cinit__(self, np.ndarray[ np.int32_t, ndim = 1 ] size_hidden_layers,
        d, np.ndarray[ np.float64_t, ndim = 1 ] random_noise):

        """
        Constructor for the class.
        """

        self._c_NETWORK = init_network(<int*> size_hidden_layers.data,
            len(size_hidden_layers), d, <double*> random_noise.data)

        if self._c_NETWORK is NULL:

            raise RuntimeError('Error creating Bayesian neural network.')

    def predict(self, np.ndarray[ np.float64_t, ndim = 2 ] x_test,
        np.ndarray[ np.float64_t, ndim = 1 ] m_test, 
        np.ndarray[ np.float64_t, ndim = 1 ] v_test, 
        np.ndarray[ np.float64_t, ndim = 1 ] v_noise):

        """
        Method that computes predictions of the Bayesian neural network.
        """

        if predict(self._c_NETWORK, <double*> x_test.data,
            <double*> m_test.data, <double*> v_test.data,
            <double*> v_noise.data, <int> x_test.shape[ 0 ],
            <int> x_test.shape[ 1 ]) is NULL:

            raise RuntimeError('Error generating predictions of '
                'Bayesian neural network.')

    def predict_deterministic(self, np.ndarray[ np.float64_t, ndim = 2 ] x_test,
        np.ndarray[ np.float64_t, ndim = 1 ] y_test):

        """
        Method that computes deterministic predictions of the Bayesian neural
        network.
        """

        if predict_deterministic(self._c_NETWORK, <double*> x_test.data,
            <double*> y_test.data, <int> x_test.shape[ 0 ],
            <int> x_test.shape[ 1 ]) is NULL:

            raise RuntimeError('Error generating deterministic predictions of '
                'Bayesian neural network.')


    def train_adf(self, np.ndarray[ np.float64_t, ndim = 2 ] x_train,
        np.ndarray[ np.float64_t, ndim = 1 ] y_train,
        np.ndarray[ np.int32_t, ndim = 1 ] permutation): 

        """
        Method that computes predictions of the Bayesian neural network.
        """

        if one_learning_epoch(self._c_NETWORK, <double*> x_train.data,
            <double*> y_train.data, <int> x_train.shape[ 0 ],
            <int> x_train.shape[ 1 ], <int*> permutation.data) is NULL:

            raise RuntimeError('Error training Bayesian neural network.')

    def get_size_params(self, np.ndarray[ np.int32_t, ndim = 1 ] size_w,
        np.ndarray[ np.int32_t, ndim = 1 ] n_layers):

        # We get the size of the weight parameters and the number of layers

        get_size_params(self._c_NETWORK, <int*> size_w.data, <int*> n_layers.data)

    def get_params(self, np.ndarray[ np.float64_t, ndim = 1 ] sample_w_out,
        np.ndarray[ np.float64_t, ndim = 1 ] m_w_out,
        np.ndarray[ np.float64_t, ndim = 1 ] v_w_out,
        np.ndarray[ np.float64_t, ndim = 1 ] m_w_hat_nat_out,
        np.ndarray[ np.float64_t, ndim = 1 ] v_w_hat_nat_out,
        np.ndarray[ np.float64_t, ndim = 1 ] a_w_hat_nat_out,
        np.ndarray[ np.float64_t, ndim = 1 ] b_w_hat_nat_out,
        np.ndarray[ np.float64_t, ndim = 1 ] a_noise_out,
        np.ndarray[ np.float64_t, ndim = 1 ] b_noise_out,
        np.ndarray[ np.float64_t, ndim = 1 ] a_prior_out,
        np.ndarray[ np.float64_t, ndim = 1 ] b_prior_out,
        np.ndarray[ np.int32_t, ndim = 1 ] neurons_per_layer_out):
 
        get_params(self._c_NETWORK, <double *> sample_w_out.data,
        <double *> m_w_out.data,
        <double *>v_w_out.data,
        <double *>m_w_hat_nat_out.data,
        <double *>v_w_hat_nat_out.data,
        <double *>a_w_hat_nat_out.data,
        <double *>b_w_hat_nat_out.data,
        <double *>a_noise_out.data,
        <double *>b_noise_out.data,
        <double *>a_prior_out.data,
        <double *>b_prior_out.data,
        <int *>neurons_per_layer_out.data)

    def set_params(self, 
        np.ndarray[ np.float64_t, ndim = 1 ] sample_w_in,
        np.ndarray[ np.float64_t, ndim = 1 ] m_w_in,
        np.ndarray[ np.float64_t, ndim = 1 ] v_w_in,
        np.ndarray[ np.float64_t, ndim = 1 ] m_w_hat_nat_in,
        np.ndarray[ np.float64_t, ndim = 1 ] v_w_hat_nat_in,
        np.ndarray[ np.float64_t, ndim = 1 ] a_w_hat_nat_in,
        np.ndarray[ np.float64_t, ndim = 1 ] b_w_hat_nat_in,
        a_noise_in, b_noise_in, a_prior_in, b_prior_in):

        set_params(self._c_NETWORK,
            <double *>sample_w_in.data, 
            <double *>m_w_in.data, <double *>v_w_in.data,
            <double *>m_w_hat_nat_in.data, <double *>v_w_hat_nat_in.data,
            <double *>a_w_hat_nat_in.data, <double *>b_w_hat_nat_in.data,
            <double> a_noise_in, <double> b_noise_in, <double> a_prior_in,
            <double> b_prior_in)

    #
    # Deallocator
    # 

    def __dealloc__(self):

        if self._c_NETWORK is not NULL:
           
            destroy_network(self._c_NETWORK)

def construct_PBP_network(size_hidden_layers, d):

    """
    Function that creates an PBP Bayesian neural network.
    """

    # We check the arguments

    if len(size_hidden_layers) < 0 or \
        type(size_hidden_layers) != list or type(d) != int:
        raise RuntimeError('construct_PBP_network::incorrect parameters.')

    # We obtain the random noise used to initialize the posterior means

    layer_sizes = [ d ]
    layer_sizes += size_hidden_layers
    layer_sizes.append(1)
    random_noise = np.array([])
    for size_out, size_in in zip(layer_sizes[ 1 : ], layer_sizes[ : -1 ]):
        random_noise = np.concatenate((random_noise,
            np.reshape(np.random.randn(size_out, size_in + 1), (1, -1))[ 0 ]))

    layer_sizes.append(1)
    random_noise = np.concatenate((random_noise,
        np.reshape(np.zeros((1, 1 + 1)), (1, -1))[ 0 ]))

    # We map the arrays so that they have the correct data type

    size_hidden_layers = np.array(size_hidden_layers, dtype = np.int32)
    random_noise = np.array(random_noise, dtype = np.float64)

    # We create the MSN

    net = PBP_net(size_hidden_layers, d, random_noise)

    return net

def predict_PBP_network(net, x_test):

    """
    Function that computes the output of a Bayesian nerual network fitted
    with PBP.
    """

    # We check the arguments

    if type(net) != PBP_net or type(x_test) != np.ndarray:
        raise RuntimeError('predict_PBP_network::incorrect parameters.')

    # We convert the input to the correct type if necessary

    if x_test.dtype.type != np.float64:
        x_test = np.array(x_test, dtype = np.float64)

    x_test = np.array(x_test, ndmin = 2)

    # We create the vector were the output values will be stored

    m_test = np.array([ 0 ] * x_test.shape[ 0 ], dtype = np.float64)
    v_test = np.array([ 0 ] * x_test.shape[ 0 ], dtype = np.float64)
    v_noise = np.array([ 0 ], dtype = np.float64)

    # We compute the output of the network

    net.predict(x_test, m_test, v_test, v_noise)

    # We return the output

    return m_test, v_test, v_noise

def predict_deterministic_PBP_network(net, x_test):

    """
    Function that computes the output of a Bayesian nerual network fitted
    with PBP.
    """

    # We check the arguments

    if type(net) != PBP_net or type(x_test) != np.ndarray:
        raise RuntimeError('predict_PBP_network::incorrect parameters.')

    # We convert the input to the correct type if necessary

    if x_test.dtype.type != np.float64:
        x_test = np.array(x_test, dtype = np.float64)

    x_test = np.array(x_test, ndmin = 2)

    # We create the vector were the output values will be stored

    y_test = np.array([ 0 ] * x_test.shape[ 0 ], dtype = np.float64)

    # We compute the output of the network

    net.predict_deterministic(x_test, y_test)

    # We return the output

    return y_test

def train_PBP_network(net, x_train, y_train, n_epochs):

    """
    Function that performs 40 training epochs of an ADF pass on a Bayesian
    nerual network fitted with PBP.
    """

    # We check the arguments

    if type(net) != PBP_net or type(x_train) != np.ndarray or \
        type(y_train) != np.ndarray:
        raise RuntimeError('do_one_epoch_PBP_network::incorrect parameters.')

    # We convert the input to the correct type if necessary

    if x_train.dtype.type != np.float64:
        x_train = np.array(x_train, dtype = np.float64)

    if y_train.dtype.type != np.float64:
        y_train = np.array(y_train, dtype = np.float64)

    # We perform one ADF pass

    for i in range(n_epochs):
        permutation = np.array(np.random.choice(range(x_train.shape[ 0 ]),
            x_train.shape[ 0 ], replace = False), dtype = np.int32)
        net.train_adf(x_train, y_train, permutation)
        print i

def map_to_dictionary_PBP_network(net):

    # We obtain the number of parameters and layers in the network

    size_w = np.array([ 0 ], dtype = np.int32)
    n_layers = np.array([ 0 ], dtype = np.int32)
    
    net.get_size_params(size_w, n_layers)
    
    # We generate arrays to store the parameters

    sample_w = np.zeros(size_w[ 0 ], dtype = np.float64)
    m_w = np.zeros(size_w[ 0 ], dtype = np.float64)
    v_w = np.zeros(size_w[ 0 ], dtype = np.float64)
    m_w_hat_nat = np.zeros(size_w[ 0 ], dtype = np.float64)
    v_w_hat_nat = np.zeros(size_w[ 0 ], dtype = np.float64)
    a_w_hat_nat = np.zeros(size_w[ 0 ], dtype = np.float64)
    b_w_hat_nat = np.zeros(size_w[ 0 ], dtype = np.float64)
    a_prior = np.zeros(1, dtype = np.float64)
    b_prior = np.zeros(1, dtype = np.float64)
    a_noise = np.zeros(1, dtype = np.float64)
    b_noise = np.zeros(1, dtype = np.float64)
    neurons_per_layer = np.zeros(n_layers[ 0 ], dtype = np.int32)

    # We get the parameters

    net.get_params(sample_w, m_w, v_w, m_w_hat_nat, v_w_hat_nat, a_w_hat_nat,
        b_w_hat_nat, a_noise, b_noise, a_prior, b_prior, neurons_per_layer)
 
    # We create a dictionary with all the network parameters

    network_dict = { 'sample_w': sample_w,
        'm_w': m_w, 'v_w': v_w, 'm_w_hat_nat': m_w_hat_nat,
        'v_w_hat_nat': v_w_hat_nat, 'a_w_hat_nat': a_w_hat_nat, 
        'b_w_hat_nat': b_w_hat_nat, 'a_prior': a_prior, 'b_prior': b_prior,
        'a_noise': a_noise, 'b_noise': b_noise,
        'neurons_per_layer': neurons_per_layer }

    return network_dict

def construct_from_dictionary_PBP_network(network_dict):

    # We create a network from scratch
    
    d = network_dict[ 'neurons_per_layer' ][ 0 ]
    size_hidden_layers = network_dict[ 'neurons_per_layer' ][ 1 : -1  ].tolist()
    net = construct_PBP_network(size_hidden_layers, int(d))

    # We fix the parameters of the newly created network

    net.set_params(network_dict[ 'sample_w' ],
        network_dict[ 'm_w' ], network_dict[ 'v_w' ],
        network_dict[ 'm_w_hat_nat' ], network_dict[ 'v_w_hat_nat' ],
        network_dict[ 'a_w_hat_nat' ], network_dict[ 'b_w_hat_nat' ],
        network_dict[ 'a_noise' ], network_dict[ 'b_noise' ],
        network_dict[ 'a_prior' ], network_dict[ 'b_prior' ])

    return net

def sample_weights_PBP_network(net):

    # We obtain the number of parameters and layers in the network

    size_w = np.array([ 0 ], dtype = np.int32)
    n_layers = np.array([ 0 ], dtype = np.int32)
    
    net.get_size_params(size_w, n_layers)
    
    # We generate arrays to store the parameters

    sample_w = np.zeros(size_w[ 0 ], dtype = np.float64)
    m_w = np.zeros(size_w[ 0 ], dtype = np.float64)
    v_w = np.zeros(size_w[ 0 ], dtype = np.float64)
    m_w_hat_nat = np.zeros(size_w[ 0 ], dtype = np.float64)
    v_w_hat_nat = np.zeros(size_w[ 0 ], dtype = np.float64)
    a_w_hat_nat = np.zeros(size_w[ 0 ], dtype = np.float64)
    b_w_hat_nat = np.zeros(size_w[ 0 ], dtype = np.float64)
    a_prior = np.zeros(1, dtype = np.float64)
    b_prior = np.zeros(1, dtype = np.float64)
    a_noise = np.zeros(1, dtype = np.float64)
    b_noise = np.zeros(1, dtype = np.float64)
    neurons_per_layer = np.zeros(n_layers[ 0 ], dtype = np.int32)

    # We get the parameters

    net.get_params(sample_w, m_w, v_w, m_w_hat_nat, v_w_hat_nat, \
        a_w_hat_nat, b_w_hat_nat, a_noise, b_noise, a_prior, b_prior, \
        neurons_per_layer)
 
    # We sample the weights

    sample_w = np.array(m_w + np.sqrt(v_w) * np.random.randn(len(m_w)),
        dtype = np.float64)

    # We restore the parameters

    # We fix the parameters of the newly created network

    net.set_params(sample_w, m_w, v_w, m_w_hat_nat, v_w_hat_nat, \
        a_w_hat_nat, b_w_hat_nat, a_noise, b_noise, a_prior, b_prior)
