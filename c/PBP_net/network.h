/*
 * Module that implements in c a Bayesian network trained with probabilistic
 * backpropagation (PBP).
 *
 * Author: Jose Miguel Hernandez Lobato
 * Date: 10 March 2015
 *
 */

#ifndef __NETWORK
#define __NETWORK

/* Structure with all the info for the network */

typedef struct {

    int n_hidden_layers;    /* The number of hidden layers */
    int *neurons_per_layer; /* The number of neurons per layer without bias */
    int *linear;            /* Type of activation function in each layer */
    double *scaling_a;      /* Scaling of the activation function per layer */

    int size_w;             /* Size of weight vector */
    int size_a;             /* Size of activation vector */
    int size_z;             /* Size of non-linear activation vector */

    int *start_w; /* The starting position for weight array per layer */
    int *start_a; /* The starting position for activation array per layer */
    int *start_z; /* The starting position for non-linear activation array */

    /* The parameters of the posterior approximation */

    double a_noise;       /* First parameter for Gamma on noise */
    double b_noise;       /* Second parameter for Gamma on noise */
    double a_prior;       /* First parameter for Gamma on the prior */
    double b_prior;       /* Second parameter for Gamma on the prior */

    double *sample_w;     /* A sample from the weights */
    double *m_w;          /* The mean for the weights */
    double *m_w_new;      /* The new mean for the weights */
    double *m_w_squared;  /* The mean for the weights squared */
    double *v_w;          /* The variances for the weights */
    double *v_w_new;      /* The new variances for the weights */

    /* Variables needed for the forward pass */

    double *m_z;          /* The mean for the neural output */
    double *m_z_squared;  /* The mean for the neural output squared */
    double *m_z_scaled;   /* The mean for the neural output scaled */
    double *v_z;          /* The variance for the neural output */
    double *v_z_scaled;   /* The variance for the neural output */
    double *m_a;          /* The mean for the activation */
    double *v_a;          /* The variance for activation */
    double *alpha;        /* The alpha for the neurons */
    double *gamma;        /* The gamma for the neurons */

    /* Variables needed for the backward pass */

    double *delta_m;      /* The delta_m for the neurons */
    double *delta_v;      /* The delta_v for the neurons */

    double *dm_z_d_m_a;   /* Auxiliary variables for the gradient */
    double *dv_z_d_m_a;   /* Auxiliary variables for the gradient */
    double *dm_z_d_v_a;   /* Auxiliary variables for the gradient */
    double *dv_z_d_v_a;   /* Auxiliary variables for the gradient */
    double *dm_a_d_m_a;   /* Auxiliary variables for the gradient */
    double *dv_a_d_m_a;   /* Auxiliary variables for the gradient */
    double *dm_a_d_v_a;   /* Auxiliary variables for the gradient */
    double *dv_a_d_v_a;   /* Auxiliary variables for the gradient */
    double *dm_a_d_m_z;   /* Auxiliary variables for the gradient */
    double *dv_a_d_m_z;   /* Auxiliary variables for the gradient */
    double *dm_a_d_v_z;   /* Auxiliary variables for the gradient */
    double *dv_a_d_v_z;   /* Auxiliary variables for the gradient */
    double *dm_a_d_m_w;   /* Auxiliary variables for the gradient */
    double *dv_a_d_m_w;   /* Auxiliary variables for the gradient */
    double *dm_a_d_v_w;   /* Auxiliary variables for the gradient */
    double *dv_a_d_v_w;   /* Auxiliary variables for the gradient */

    double *grad_m_w;     /* The gradient for mean of the weights */
    double *grad_v_w;     /* The gradient for the variances of the weights */

    /* Variables needed for refining the prior */

    double *m_w_hat_nat;  /* The 1st nat. param. for the prior approximation */
    double *v_w_hat_nat;  /* The 2st nat. param. for the prior approximation */
    double *a_w_hat_nat;  /* The 1st parameter for the prior approximation */
    double *b_w_hat_nat;  /* The 2nd parameter for the prior approximation */

    double *m_w_old;      /* The old mean for the weights */
    double *v_w_old;      /* The old variances for the weights */

    } NETWORK;

NETWORK *init_network(int *size_hidden_layers, int n_hidden_layers, int d, double *random_noise);
void destroy_network(NETWORK * network);
NETWORK *predict(NETWORK *network, double *x_test, double *m_test, double *v_test, double *v_noise, int n_datapoints, int d);
NETWORK *one_learning_epoch(NETWORK *network, double *x, double *y, int n_datapoints, int d, int *permutation);
void get_params(NETWORK *n, double *sample_w_out, double *m_w_out, double *v_w_out, double *m_w_hat_nat_out, double *v_w_hat_nat_out,
    double *a_w_hat_nat_out, double *b_w_hat_nat_out, double *a_noise_out, double *b_noise_out, double *a_prior_out,
    double *b_prior_out, int *neurons_per_layer_out);
void get_size_params(NETWORK *n, int *size_weihts, int *n_layers);
void set_params(NETWORK *n, double *sample_w_in, double *m_w_in, double *v_w_in, double *m_w_hat_nat_in, double *v_w_hat_nat_in,
    double *a_w_hat_nat_in, double *b_w_hat_nat_in, double a_noise_in, double b_noise_in, double a_prior_in, double b_prior_in);
NETWORK *predict_deterministic(NETWORK *n, double *x_test, double *y_test, int n_datapoints, int d);

#endif
