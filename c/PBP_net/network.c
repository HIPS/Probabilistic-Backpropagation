/*
 * Module that implements in c a Bayesian network trained with probabilistic
 * backpropagation (PBP).
 *
 * Author: Jose Miguel Hernandez Lobato
 * Date: 10 March 2015
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <cblas.h>

#include "network.h"
#include "pnorm.h"

/* Prototipes of private functions */

void refine_prior(NETWORK *network);
void do_ADF_update(NETWORK * network, double *x, double y);
double randn(double mu, double sigma);

/**
 * Generates a sample from a Gaussian distribution
 * @param mu    Mean of the Gaussian.
 * @param sigma Standard deviation of the Gaussian
 *
 */

double randn(double mu, double sigma) {

  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;

  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }

  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);

  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (double) X1);
}

/**
 * Constructor for a network.
 *
 * @param size_hidden_layers    The size of each hidden layer.
 * @param n_hidden_layers       The number of hidden layers.
 * @param d                     The input dimensionality.
 * @param random_noise          Random noise for the initialization of the
 *                              posterior means.
 *
 */

NETWORK *init_network(int *size_hidden_layers, int n_hidden_layers, int d,double *random_noise) {

    NETWORK *ret;
    int n, i, j, k;

    /* We save memory for the structure */
    
    ret = malloc(sizeof(NETWORK));

    /* We initialize the number of hidden layers */

    ret->n_hidden_layers = n_hidden_layers;

    /* We refine the prior on the noise variance and on the prior variance */

    ret->a_noise = 2.0 * 3;
    ret->b_noise = 2.0 * 3;
    ret->a_prior = 2.0 * 3;
    ret->b_prior = 2.0 * 3;

    /* We save memory for the array with the number of neurons per layer,
     * the activation type, the scaling of the activation per layer,
       and he starting position for the different arrays */

    ret->neurons_per_layer = malloc(sizeof(int) * (n_hidden_layers + 3));
    ret->linear = malloc(sizeof(int) * (n_hidden_layers + 3));
    ret->scaling_a = malloc(sizeof(double) * (n_hidden_layers + 3));
    ret->start_w = malloc(sizeof(int) * (n_hidden_layers + 3));
    ret->start_z = malloc(sizeof(int) * (n_hidden_layers + 3));
    ret->start_a = malloc(sizeof(int) * (n_hidden_layers + 3));

    /* We initialize the number of neurons per layer, the activation type and
     * compute the size of the auxiliary variables */

    /* The input layer */

    ret->start_w[ 0 ] = -1;
    ret->start_z[ 0 ] = 0;
    ret->start_a[ 0 ] = 0;

    ret->neurons_per_layer[ 0 ] = d;
    ret->linear[ 0 ] = 0;
    ret->scaling_a[ 0 ] = 1.0;

    ret->size_w = 0;
    ret->size_a = d;
    ret->size_z = d + 1;

    /* The hidden layers */

    for (i = 1 ; i < n_hidden_layers + 1 ; i ++) {
        ret->start_w[ i ] = ret->size_w;
        ret->start_z[ i ] = ret->size_z;
        ret->start_a[ i ] = ret->size_a;

        ret->neurons_per_layer[ i ] = size_hidden_layers[ i - 1 ];
        ret->linear[ i ] = 0;
        ret->scaling_a[ i ] = 1.0 / (ret->neurons_per_layer[ i - 1 ] + 1);

        ret->size_w += (ret->neurons_per_layer[ i - 1 ] + 1) * ret->neurons_per_layer[ i ];
        ret->size_a += ret->neurons_per_layer[ i ];
        ret->size_z += ret->neurons_per_layer[ i ] + 1;
    }

    /* The output layer */

    ret->start_w[ i ] = ret->size_w;
    ret->start_z[ i ] = ret->size_z;
    ret->start_a[ i ] = ret->size_a;

    ret->neurons_per_layer[ i ] = 1;
    ret->linear[ i ] = 1;
    ret->scaling_a[ i ] = 1.0 / (ret->neurons_per_layer[ i - 1 ] + 1);

    ret->size_w += (ret->neurons_per_layer[ i - 1 ] + 1) * ret->neurons_per_layer[ i ];
    ret->size_a += ret->neurons_per_layer[ i ];
    ret->size_z += ret->neurons_per_layer[ i ] + 1;
    
    /* The fake output layer */

    i++;
    ret->start_w[ i ] = ret->size_w;
    ret->start_z[ i ] = ret->size_z;
    ret->start_a[ i ] = ret->size_a;

    ret->neurons_per_layer[ i ] = 1;
    ret->linear[ i ] = 1;
    ret->scaling_a[ i ] = 1.0;

    ret->size_w += (ret->neurons_per_layer[ i - 1 ] + 1) * ret->neurons_per_layer[ i ];
    ret->size_a += ret->neurons_per_layer[ i ];
    ret->size_z += ret->neurons_per_layer[ i ] + 1;
 
    /* We save memory for the auxiliary variables */

    ret->sample_w = (double *) malloc(sizeof(double) * ret->size_w);
    ret->m_w = (double *) malloc(sizeof(double) * ret->size_w);
    ret->m_w_new = (double *) malloc(sizeof(double) * ret->size_w);
    ret->m_w_squared = (double *)  malloc(sizeof(double) * ret->size_w);
    ret->v_w = (double *) malloc(sizeof(double) * ret->size_w);
    ret->v_w_new = (double *) malloc(sizeof(double) * ret->size_w);
    ret->m_z = (double *) malloc(sizeof(double) * ret->size_z);
    ret->m_z_scaled = (double *) malloc(sizeof(double) * ret->size_z);
    ret->m_z_squared = (double *) malloc(sizeof(double) * ret->size_z);
    ret->v_z = (double *) malloc(sizeof(double) * ret->size_z);
    ret->v_z_scaled = (double *) malloc(sizeof(double) * ret->size_z);
    ret->m_a = (double *) malloc(sizeof(double) * ret->size_a);
    ret->v_a = (double *) malloc(sizeof(double) * ret->size_a);
    ret->alpha = (double *) malloc(sizeof(double) * ret->size_a);
    ret->gamma = (double *) malloc(sizeof(double) * ret->size_a);
    ret->delta_m = (double *) malloc(sizeof(double) * ret->size_z);
    ret->delta_v = (double *) malloc(sizeof(double) * ret->size_z);

    ret->dm_z_d_m_a = (double *) malloc(sizeof(double) * ret->size_z);
    ret->dv_z_d_m_a = (double *) malloc(sizeof(double) * ret->size_z);
    ret->dm_z_d_v_a = (double *) malloc(sizeof(double) * ret->size_z);
    ret->dv_z_d_v_a = (double *) malloc(sizeof(double) * ret->size_z);
    ret->dm_a_d_m_a = (double *) malloc(sizeof(double) * ret->size_w);
    ret->dv_a_d_m_a = (double *) malloc(sizeof(double) * ret->size_w);
    ret->dm_a_d_v_a = (double *) malloc(sizeof(double) * ret->size_w);
    ret->dv_a_d_v_a = (double *) malloc(sizeof(double) * ret->size_w);
    ret->dm_a_d_m_z = (double *) malloc(sizeof(double) * ret->size_w);
    ret->dv_a_d_m_z = (double *) malloc(sizeof(double) * ret->size_w);
    ret->dm_a_d_v_z = (double *) malloc(sizeof(double) * ret->size_w);
    ret->dv_a_d_v_z = (double *) malloc(sizeof(double) * ret->size_w);
    ret->dm_a_d_m_w = (double *) malloc(sizeof(double) * ret->size_w);
    ret->dv_a_d_m_w = (double *) malloc(sizeof(double) * ret->size_w);
    ret->dm_a_d_v_w = (double *) malloc(sizeof(double) * ret->size_w);
    ret->dv_a_d_v_w = (double *) malloc(sizeof(double) * ret->size_w);
    ret->grad_m_w = (double *) malloc(sizeof(double) * ret->size_w);
    ret->grad_v_w = (double *) malloc(sizeof(double) * ret->size_w);

    ret->m_w_hat_nat = (double *) malloc(sizeof(double) * ret->size_w);
    ret->v_w_hat_nat = (double *) malloc(sizeof(double) * ret->size_w);
    ret->a_w_hat_nat = (double *) malloc(sizeof(double) * ret->size_w);
    ret->b_w_hat_nat = (double *) malloc(sizeof(double) * ret->size_w);

    ret->m_w_old = (double *) malloc(sizeof(double) * ret->size_w);
    ret->v_w_old = (double *) malloc(sizeof(double) * ret->size_w);

    /* We initialize the posterior approximation */
    
    k = 0;
    for (i = 1 ; i < n_hidden_layers + 3 ; i ++) {
        n = (ret->neurons_per_layer[ i - 1 ] + 1) * ret->neurons_per_layer[ i ];
        for (j = 0 ; j < n ; j++) {
            ret->m_w[ k ] = 1.0 / sqrt(ret->neurons_per_layer[ i - 1 ] + 1) * random_noise[ k ];
            ret->v_w[ k ] = ret->b_prior / (ret->a_prior - 1);
            ret->m_w_hat_nat[ k ] = 0;
            ret->v_w_hat_nat[ k ] = (ret->a_prior - 1) / ret->b_prior;
            ret->a_w_hat_nat[ k ] = 0;
            ret->b_w_hat_nat[ k ] = 0;
            k++;
        }
    }

    /* We initialize the weights of the fake ouput layer */

    ret->m_w[ ret->start_w[ n_hidden_layers + 2 ] ] = 1.0;
    ret->m_w[ ret->start_w[ n_hidden_layers + 2 ] + 1 ] = 0.0;
    ret->v_w[ ret->start_w[ n_hidden_layers + 2 ] ] = 0.0;
    ret->v_w[ ret->start_w[ n_hidden_layers + 2 ] + 1 ] = 0.0;

    return ret;
}

/**
 * Constructor for a network.
 *
 * @param network   Pointer to the network to destroy.
 *
 */

void destroy_network(NETWORK *n) {

    free(n->neurons_per_layer);
    free(n->linear);
    free(n->scaling_a);
    free(n->start_w);
    free(n->start_a);
    free(n->start_z);
    free(n->sample_w);
    free(n->m_w);
    free(n->m_w_new);
    free(n->m_w_squared);
    free(n->v_w);
    free(n->v_w_new);
    free(n->m_z);
    free(n->m_z_squared);
    free(n->m_z_scaled);
    free(n->v_z);
    free(n->m_a);
    free(n->v_a);
    free(n->alpha);
    free(n->gamma);
    free(n->delta_m);
    free(n->delta_v);
    free(n->dm_z_d_m_a);
    free(n->dv_z_d_m_a);
    free(n->dm_z_d_v_a);
    free(n->dv_z_d_v_a);
    free(n->dm_a_d_m_a);
    free(n->dv_a_d_m_a);
    free(n->dm_a_d_v_a);
    free(n->dv_a_d_v_a);
    free(n->dm_a_d_m_z);
    free(n->dv_a_d_m_z);
    free(n->dm_a_d_v_z);
    free(n->dv_a_d_v_z);
    free(n->dm_a_d_m_w);
    free(n->dv_a_d_m_w);
    free(n->dm_a_d_v_w);
    free(n->dv_a_d_v_w);
    free(n->grad_m_w);
    free(n->grad_v_w);
    free(n->m_w_hat_nat);
    free(n->v_w_hat_nat);
    free(n->a_w_hat_nat);
    free(n->b_w_hat_nat);
    free(n->m_w_old);
    free(n->v_w_old);

    return;
}

/**
 * Deterministic forward propagation of the data.
 *
 */

void deterministc_forward_PBP(NETWORK *n, int index_layer) {

    int i;
    double sqrt_aux;

    /* Auxiliary pointers */

    double * __restrict__ sample_w;
    double * __restrict__ m_z;
    double * __restrict__ m_z_previous_layer;
    double * __restrict__ m_a;

    /* Auxiliary variables */

    double scaling_a;
    int n_neurons;
    int input_size;
    int linear;
    
    /* We initialize the pointers */

    sample_w = n->sample_w + n->start_w[ index_layer ];
    m_z = n->m_z + n->start_z[ index_layer ];
    m_z_previous_layer = n->m_z + n->start_z[ index_layer - 1 ];
    m_a = n->m_a + n->start_a[ index_layer ];

    /* We update the number of neurons, input size, type of activation and scaling constant */

    n_neurons = n->neurons_per_layer[ index_layer ];
    input_size = n->neurons_per_layer[ index_layer - 1 ] + 1;
    linear = n->linear[ index_layer ];
    scaling_a = n->scaling_a[ index_layer ];

    /* We compute the scaled activation mean: sqrt_aux * sample_w x m_z */

    sqrt_aux = sqrt(scaling_a);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n_neurons, input_size, sqrt_aux, sample_w, input_size, m_z_previous_layer, 1, 0.0, m_a, 1);

    /* We apply the non-linearity or compute a linear activation function */

    if (linear) {
        for (i = 0 ; i < n_neurons ; i++) {
            m_z[ i ] = m_a[ i ];
        }
    } else {
        for (i = 0 ; i < n_neurons ; i++) {
            m_z[ i ] = (m_a[ i ] > 0) ? m_a[ i ] : 0;
        }
    }

    /* We add the bias if we are not in the two last layers */

    if (index_layer < n->n_hidden_layers + 1) {
        m_z[ n_neurons ] = 1.0;
    }
}

/**
 * Forward propagation of probabilities.
 *
 */

void forward_PBP(NETWORK *n, int index_layer) {

    int i, limit;
    double a, a2, p, g, sqrt_aux;

    /* Auxiliary pointers */

    double * __restrict__ m_w;
    double * __restrict__ m_w_squared;
    double * __restrict__ v_w;
    double * __restrict__ m_z;
    double * __restrict__ v_z;
    double * __restrict__ m_a;
    double * __restrict__ v_a;
    double * __restrict__ m_z_previous_layer;
    double * __restrict__ m_z_previous_layer_squared;
    double * __restrict__ v_z_previous_layer;
    double * __restrict__ alpha;
    double * __restrict__ gamma;

    /* Auxiliary variables */

    double scaling_a;
    int n_neurons;
    int input_size;
    int linear;
    
    /* We initialize the pointers */

    m_w = n->m_w + n->start_w[ index_layer ];
    m_w_squared = n->m_w_squared + n->start_w[ index_layer ];
    v_w = n->v_w + n->start_w[ index_layer ];
    m_z = n->m_z + n->start_z[ index_layer ];
    v_z = n->v_z + n->start_z[ index_layer ];
    m_a = n->m_a + n->start_a[ index_layer ];
    v_a = n->v_a + n->start_a[ index_layer ];
    m_z_previous_layer = n->m_z + n->start_z[ index_layer - 1 ];
    v_z_previous_layer = n->v_z + n->start_z[ index_layer - 1 ];
    m_z_previous_layer_squared = n->m_z_squared + n->start_z[ index_layer - 1 ];
    alpha = n->alpha + n->start_a[ index_layer ];
    gamma = n->gamma + n->start_a[ index_layer ];

    /* We update the number of neurons, input size, type of activation and scaling constant */

    n_neurons = n->neurons_per_layer[ index_layer ];
    input_size = n->neurons_per_layer[ index_layer - 1 ] + 1;
    linear = n->linear[ index_layer ];
    scaling_a = n->scaling_a[ index_layer ];

    /* We compute the m_z_previous_layer_squared and the m_w_squared */

    limit = n_neurons * input_size;
    for (i = 0 ; i < limit ; i++)
        m_w_squared[ i ] = m_w[ i ] * m_w[ i ];
    for (i = 0 ; i < input_size ; i++)
        m_z_previous_layer_squared[ i ] = m_z_previous_layer[ i ] * m_z_previous_layer[ i ];

    /* We compute the scaled activation mean: sqrt_aux * m_w x m_z */

    sqrt_aux = sqrt(scaling_a);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n_neurons, input_size, sqrt_aux, m_w, input_size, m_z_previous_layer, 1, 0.0, m_a, 1);

    /* We compute the scaled activation variance: (m_w_squared x v_z_previous_layer +
       v_w x m_z_previous_layer_squared + v_w x v_z_previous_layer) */

    cblas_dgemv(CblasRowMajor, CblasNoTrans, n_neurons, input_size, scaling_a, m_w_squared, input_size, v_z_previous_layer, 1, 0, v_a, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n_neurons, input_size, scaling_a, v_w, input_size, m_z_previous_layer_squared, 1, 1.0, v_a, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n_neurons, input_size, scaling_a, v_w, input_size, v_z_previous_layer, 1, 1.0, v_a, 1);

    /* We apply the non-linearity or compute a linear activation function */

    if (linear) {
        for (i = 0 ; i < n_neurons ; i++) {
            m_z[ i ] = m_a[ i ];
            v_z[ i ] = v_a[ i ];
        }
    } else {
        for (i = 0 ; i < n_neurons ; i++) {

            sqrt_aux = sqrt(v_a[ i ]);
            a = m_a[ i ] / sqrt_aux;
            a2 = a * a;
            p = pnorm(a);
            g = (a < -30) ? -a - 1.0 / a + 2.0 / (a * a2) : 0.398942280401432677939946059 * exp(-0.5 * a2) / p;

            /* We compute the output mean and variance */

            m_z[ i ] = p * (m_a[ i ] + sqrt_aux * g);
            v_z[ i ] = m_z[ i ] * (m_a[ i ] + sqrt_aux * g) * (1 - p) + p * v_a[ i ] * (1 - g * g - g * a);

            alpha[ i ] = a;
            gamma[ i ] = g;
        }
    }

    /* We add the bias if we are not in the two last layers */

    if (index_layer < n->n_hidden_layers + 1) {
        m_z[ n_neurons ] = 1.0;
        v_z[ n_neurons ] = 0.0;
    }
}

/**
 * Backward computation of gradients.
 *
 */

void backward_PBP(NETWORK *n, int index_layer) {

    double g, a, da_dm_a, da_dv_a, dg_dm_a, dg_dv_a, d, p;
    int i, j, k, limit_1, limit_2;

    /* Auxiliary pointers */

    double * __restrict__ m_w;
    double * __restrict__ m_w_layer_above;
    double * __restrict__ v_w_layer_above;
    double * __restrict__ m_z;
    double * __restrict__ m_z_scaled;
    double * __restrict__ m_a;
    double * __restrict__ v_a;
    double * __restrict__ m_z_layer_below;
    double * __restrict__ m_z_layer_below_scaled;
    double * __restrict__ v_z_layer_below;
    double * __restrict__ v_z_layer_below_scaled;
    double * __restrict__ alpha;
    double * __restrict__ gamma;
    double * __restrict__ delta_m_layer_above;
    double * __restrict__ delta_v_layer_above;
    double * __restrict__ delta_m;
    double * __restrict__ delta_v;
    double * __restrict__ dm_z_d_m_a;
    double * __restrict__ dv_z_d_m_a;
    double * __restrict__ dm_z_d_v_a;
    double * __restrict__ dv_z_d_v_a;
    double * __restrict__ dm_a_d_m_a;
    double * __restrict__ dv_a_d_m_a;
    double * __restrict__ dm_a_d_v_a;
    double * __restrict__ dv_a_d_v_a;
    double * __restrict__ dm_a_d_m_z;
    double * __restrict__ dv_a_d_m_z;
    double * __restrict__ dm_a_d_v_z;
    double * __restrict__ dv_a_d_v_z;
    double * __restrict__ dm_a_d_m_w;
    double * __restrict__ dv_a_d_m_w;
    double * __restrict__ dm_a_d_v_w;
    double * __restrict__ dv_a_d_v_w;
    double * __restrict__ grad_m_w;
    double * __restrict__ grad_v_w;

    /* Auxiliary variables */

    double scaling_a, scaling_a_layer_above, sqrt_aux, sqrt_v_a;
    double *p_aux_1, *p_aux_2, *p_aux_3, *p_aux_4;
    double *p_aux_1_2, *p_aux_2_2, *p_aux_3_2, *p_aux_4_2;
    int n_neurons;
    int linear;

    /* We initialize the auxiliary pointers */

    m_w = n->m_w + n->start_w[ index_layer ];
    m_w_layer_above = n->m_w + n->start_w[ index_layer + 1 ];
    v_w_layer_above = n->v_w + n->start_w[ index_layer + 1 ];
    m_z = n->m_z + n->start_z[ index_layer ];
    m_z_scaled = n->m_z_scaled + n->start_z[ index_layer ];
    m_a = n->m_a + n->start_a[ index_layer ];
    v_a = n->v_a + n->start_a[ index_layer ];
    m_z_layer_below = n->m_z + n->start_z[ index_layer - 1 ];
    v_z_layer_below = n->v_z + n->start_z[ index_layer - 1 ];
    m_z_layer_below_scaled = n->m_z_scaled + n->start_z[ index_layer - 1 ];
    v_z_layer_below_scaled = n->v_z_scaled + n->start_z[ index_layer - 1 ];
    alpha = n->alpha + n->start_a[ index_layer ];
    gamma = n->gamma + n->start_a[ index_layer ];
    delta_m_layer_above = n->delta_m + n->start_z[ index_layer + 1 ];
    delta_v_layer_above = n->delta_v + n->start_z[ index_layer + 1 ];
    delta_m = n->delta_m + n->start_z[ index_layer ];
    delta_v = n->delta_v + n->start_z[ index_layer ];

    dm_z_d_m_a = n->dm_z_d_m_a + n->start_z[ index_layer ];
    dv_z_d_m_a = n->dv_z_d_m_a + n->start_z[ index_layer ];
    dm_z_d_v_a = n->dm_z_d_v_a + n->start_z[ index_layer ];
    dv_z_d_v_a = n->dv_z_d_v_a + n->start_z[ index_layer ];
    dm_a_d_m_a = n->dm_a_d_m_a + n->start_w[ index_layer ];
    dv_a_d_m_a = n->dv_a_d_m_a + n->start_w[ index_layer ];
    dm_a_d_v_a = n->dm_a_d_v_a + n->start_w[ index_layer ];
    dv_a_d_v_a = n->dv_a_d_v_a + n->start_w[ index_layer ];
    dm_a_d_m_z = n->dm_a_d_m_z + n->start_w[ index_layer ];
    dv_a_d_m_z = n->dv_a_d_m_z + n->start_w[ index_layer ];
    dm_a_d_v_z = n->dm_a_d_v_z + n->start_w[ index_layer ];
    dv_a_d_v_z = n->dv_a_d_v_z + n->start_w[ index_layer ];
    dm_a_d_m_w = n->dm_a_d_m_w + n->start_w[ index_layer ];
    dv_a_d_m_w = n->dv_a_d_m_w + n->start_w[ index_layer ];
    dm_a_d_v_w = n->dm_a_d_v_w + n->start_w[ index_layer ];
    dv_a_d_v_w = n->dv_a_d_v_w + n->start_w[ index_layer ];
    grad_m_w = n->grad_m_w + n->start_w[ index_layer ];
    grad_v_w = n->grad_v_w + n->start_w[ index_layer ];

    /* We update the number of neurons, input size, type of activation and scaling constant */

    n_neurons = n->neurons_per_layer[ index_layer ];
    linear = n->linear[ index_layer ];
    scaling_a = n->scaling_a[ index_layer ];
    scaling_a_layer_above = n->scaling_a[ index_layer + 1 ];

    /* We compute the gradient of the non-linear activations with respect to the activations */

    if (linear) {
        for (i = 0 ; i < n_neurons ; i++) {
            dm_z_d_m_a[ i ] = 1.0;
            dm_z_d_v_a[ i ] = 0.0;
            dv_z_d_m_a[ i ] = 0.0;
            dv_z_d_v_a[ i ] = 1.0;

        }
    } else {
        for (i = 0 ; i < n_neurons ; i++) {
            g = gamma[ i ];
            a = alpha[ i ];
            da_dm_a = 1.0 / sqrt(v_a[ i ]);
            da_dv_a = m_a[ i ] / (2 * v_a[ i ]) * da_dm_a;
            if (a < -30) {
                g = -a - 1.0 / a + 2.0 / (a * a * a);
                dg_dm_a = -da_dm_a + 1.0 / (a * a) * da_dm_a - 6.0 / (a * a * a * a) * da_dm_a;
                dg_dv_a = -da_dv_a + 1.0 / (a * a) * da_dv_a - 6.0 / (a * a * a * a) * da_dv_a;
            } else {
                dg_dm_a = -(g * a + g * g) * da_dm_a;
                dg_dv_a = -(g * a + g * g) * da_dv_a;
            }
            d = 0.398942280401432677939946059 * exp(-0.5 * a * a);
            p = pnorm(a);
            sqrt_v_a = sqrt(v_a[ i ]);
            dm_z_d_m_a[ i ] = da_dm_a * d * (m_a[ i ] + sqrt_v_a * g) + p * (1 + sqrt_v_a * dg_dm_a);
            dm_z_d_v_a[ i ] = da_dv_a * d * (m_a[ i ] + sqrt_v_a * g) + p * (g / (2 * sqrt_v_a) + sqrt_v_a * dg_dv_a);
            dv_z_d_m_a[ i ] = dm_z_d_m_a[ i ] * (m_a[ i ] + sqrt_v_a * g) * (1 - p) + m_z[ i ] * ((1 + sqrt_v_a * dg_dm_a) * (1 - p) -
                (m_a[ i ] + sqrt_v_a * g) * d * da_dm_a) + d * da_dm_a * v_a[ i ] * (1 - g * g - g * a) - p * v_a[ i ] *
                (2 * g * dg_dm_a + dg_dm_a * a + g * da_dm_a);
            dv_z_d_v_a[ i ] = dm_z_d_v_a[ i ] * (m_a[ i ] + sqrt_v_a * g) * (1 - p) + m_z[ i ] * ((0.5 / sqrt_v_a * g + dg_dv_a * sqrt_v_a) * (1 - p) -
                (m_a[ i ]+ sqrt_v_a * g) * d * da_dv_a) + d * da_dv_a * v_a[ i ] * (1 - g * g - g * a) + p * ((1 - g * g - g * a) + v_a[ i ] *
                (-2 * g * dg_dv_a - dg_dv_a * a - g * da_dv_a));
        }
    }

    /* We scale m_z */

    limit_1 = n_neurons + 1;
    for (i = 0 ; i < limit_1 ; i++) {
        m_z_scaled[ i ] = m_z[ i ] * scaling_a_layer_above;
    }

    /* We initialize the rows of dv_a_dm_z to m_z_scaled */

    limit_1 = n->neurons_per_layer[ index_layer + 1 ];
    limit_2 = n_neurons + 1;
    for (i = 0 ; i < limit_1 ; i++) {
        p_aux_1 = dv_a_d_m_z + i * limit_2;
        for (j = 0 ; j < limit_2 ; j++) {
            p_aux_1[ j ] = m_z_scaled[ j ];
        }
    }

    /* We compute the gradient of the activation of the layer above with respect to the non-linear activations in this layer */

    sqrt_aux = sqrt(scaling_a_layer_above);
    limit_1 = n->neurons_per_layer[ index_layer + 1 ] * (n_neurons + 1);
    for (i = 0 ; i < limit_1 ; i++) {
        dm_a_d_m_z[ i ] = m_w_layer_above[ i ] * sqrt_aux;
        dm_a_d_v_z[ i ] = 0;
        dv_a_d_m_z[ i ] = dv_a_d_m_z[ i ] * 2 * v_w_layer_above[ i ];
        dv_a_d_v_z[ i ] = (m_w_layer_above[ i ] * m_w_layer_above[ i ] + v_w_layer_above[ i ]) * scaling_a_layer_above;
    }

    /* We scale the activatons from the layer below */

    sqrt_aux = sqrt(scaling_a);
    limit_1 = n->neurons_per_layer[ index_layer - 1 ] + 1;
    for (i = 0 ; i < limit_1 ; i++) {
        m_z_layer_below_scaled[ i ] = m_z_layer_below[ i ] * sqrt_aux;
        v_z_layer_below_scaled[ i ] = v_z_layer_below[ i ] * scaling_a;
    }

    /* We initialize dm_a_dm_w, dv_a_dm_w and dv_a_dv_w */

    limit_1 = n_neurons;
    limit_2 = n->neurons_per_layer[ index_layer - 1 ] + 1;
    for (i = 0 ; i < limit_1 ; i++) {
        p_aux_1 = dm_a_d_m_w + i * limit_2;
        p_aux_2 = dv_a_d_m_w + i * limit_2;
        p_aux_3 = dv_a_d_v_w + i * limit_2;
        for (j = 0 ; j < limit_2 ; j++) {
            p_aux_1[ j ] = m_z_layer_below_scaled[ j ];
            p_aux_2[ j ] = v_z_layer_below_scaled[ j ];
            p_aux_3[ j ] = (m_z_layer_below_scaled[ j ] * m_z_layer_below_scaled[ j ] + v_z_layer_below_scaled[ j ]);
        }
    }

    limit_1 = n_neurons * (n->neurons_per_layer[ index_layer - 1 ] + 1);
    for (i = 0 ; i < limit_1 ; i++) {
         dm_a_d_v_w[ i ] = 0;
         dv_a_d_m_w[ i ] = 2 * dv_a_d_m_w[ i ] * m_w[ i ];
    }

    /* We compute the gradient of the activations of the top layer with respect to the activations of the current layer */

    limit_1 = n->neurons_per_layer[ index_layer + 1 ];
    limit_2 = n_neurons + 1;
    k = 0;
    for (i = 0 ; i < limit_1 ; i++) {
        p_aux_1 = dm_a_d_m_a + i * limit_2;
        p_aux_2 = dv_a_d_m_a + i * limit_2;
        p_aux_3 = dm_a_d_v_a + i * limit_2;
        p_aux_4 = dv_a_d_v_a + i * limit_2;
        p_aux_1_2 = dm_a_d_m_z + i * limit_2;
        p_aux_2_2 = dm_a_d_v_z + i * limit_2;
        p_aux_3_2 = dv_a_d_m_z + i * limit_2;
        p_aux_4_2 = dv_a_d_v_z + i * limit_2;
        for (j = 0 ; j < limit_2 - 1 ; j++) {
            p_aux_1[ j ] = p_aux_1_2[ j ] * dm_z_d_m_a[ j ] + p_aux_2_2[ j ] * dv_z_d_m_a[ j ];
            p_aux_2[ j ] = p_aux_3_2[ j ] * dm_z_d_m_a[ j ] + p_aux_4_2[ j ] * dv_z_d_m_a[ j ];
            p_aux_3[ j ] = p_aux_1_2[ j ] * dm_z_d_v_a[ j ] + p_aux_2_2[ j ] * dv_z_d_v_a[ j ];
            p_aux_4[ j ] = p_aux_3_2[ j ] * dm_z_d_v_a[ j ] + p_aux_4_2[ j ] * dv_z_d_v_a[ j ];
        }
    }

    /* We compute the deltas */
 
    limit_1 = n->neurons_per_layer[ index_layer + 1 ];
    cblas_dgemv(CblasRowMajor, CblasTrans, limit_1, n_neurons + 1, 1.0, dm_a_d_m_a, n_neurons + 1, delta_m_layer_above, 1, 0.0, delta_m, 1);
    cblas_dgemv(CblasRowMajor, CblasTrans, limit_1, n_neurons + 1, 1.0, dv_a_d_m_a, n_neurons + 1, delta_v_layer_above, 1, 1.0, delta_m, 1);
    cblas_dgemv(CblasRowMajor, CblasTrans, limit_1, n_neurons + 1, 1.0, dm_a_d_v_a, n_neurons + 1, delta_m_layer_above, 1, 0.0, delta_v, 1);
    cblas_dgemv(CblasRowMajor, CblasTrans, limit_1, n_neurons + 1, 1.0, dv_a_d_v_a, n_neurons + 1, delta_v_layer_above, 1, 1.0, delta_v, 1);

    /* We compute the gradients */

    limit_1 = n_neurons;
    limit_2 = n->neurons_per_layer[ index_layer - 1 ] + 1;
    k = 0;
    for (i = 0 ; i < limit_1 ; i++) {
        for (j = 0 ; j < limit_2 ; j++) {
            grad_m_w[ k ] = delta_m[ i ] * dm_a_d_m_w[ k ] + delta_v[ i ] * dv_a_d_m_w[ k ];
            grad_v_w[ k ] = delta_m[ i ] * dm_a_d_v_w[ k ] + delta_v[ i ] * dv_a_d_v_w[ k ];
            k++;
        }
    }
}

/**
 * Function that performs an ADF update.
 *
 * @param network   Pointer to the network.
 * @param x         Pointer to the input features.
 * @param y         Pointer to the target.
 *
 */

void do_ADF_update(NETWORK * n, double *x, double y) {

    int i, limit_1;

    double * __restrict__ m_z;
    double * __restrict__ v_z;
    double * __restrict__ delta_m;
    double * __restrict__ delta_v;

    double * __restrict__ m_w;
    double * __restrict__ v_w;
    double * __restrict__ grad_m_w;
    double * __restrict__ grad_v_w;

    double m, v, v1, v2, logZ, logZ1, logZ2, d_logZ_d_m, d_logZ_d_v, a_new, b_new, m_aux, v_aux;

    /* We initialize the non-linear activations of the input layer with the data */

    limit_1 = n->start_a[ 1 ];
    m_z = n->m_z;
    v_z = n->v_z;
    for (i = 0 ; i < limit_1 ; i++) {
        m_z[ i ] = x[ i ];
        v_z[ i ] = 0.0;
    }
    m_z[ i ] = 1.0;
    v_z[ i ] = 0.0;

    /* We do a forward pass */

    limit_1 = n->n_hidden_layers + 2;
    for (i = 1 ; i < limit_1 ; i++)
        forward_PBP(n, i);

    /* We obtain logZ, logZ1 and logZ2 and the updates for a and b */

    m_z = n->m_z + n->start_z[ n->n_hidden_layers + 1 ];
    v_z = n->v_z + n->start_z[ n->n_hidden_layers + 1 ];

    m = m_z[ 0 ];
    v = v_z[ 0 ] + n->b_noise / (n->a_noise - 1);
    v1 = v_z[ 0 ] + n->b_noise / (n->a_noise - 0);
    v2 = v_z[ 0 ] + n->b_noise / (n->a_noise + 1);
    logZ = -0.5 * (log(v) + (m - y) * (m - y) / v);
    logZ1 = -0.5 * (log(v1) + (m - y) * (m - y) / v1);
    logZ2 = -0.5 * (log(v2) + (m - y) * (m - y) / v2);

    a_new = 1.0 / (exp(logZ2 - 2 * logZ1 + logZ) * (n->a_noise + 1.0) / n->a_noise - 1.0);
    b_new = 1.0 / (exp(logZ2 - logZ1) * (n->a_noise + 1) / n->b_noise - exp(logZ1 - logZ) * n->a_noise / n->b_noise);

    /* We initialize the deltas for the output layer */

    d_logZ_d_m = -(m - y) / v;
    d_logZ_d_v = -0.5 / v + 0.5 * (m - y) * (m - y) / (v * v);
    delta_m = n->delta_m + n->start_z[ n->n_hidden_layers + 2 ];
    delta_v = n->delta_v + n->start_z[ n->n_hidden_layers + 2 ];
    delta_m[ 0 ] = d_logZ_d_m;
    delta_v[ 0 ] = d_logZ_d_v;

    /* We do a backward pass */

    limit_1 = n->n_hidden_layers + 1;
    for (i = limit_1 ; i >= 1 ; i--)
        backward_PBP(n, i);

    /* We update the mean and variance parameters, except those in the fake output layer */

    m_w = n->m_w;
    v_w = n->v_w;
    grad_m_w = n->grad_m_w;
    grad_v_w = n->grad_v_w;
    
    limit_1 = n->size_w - 2;
    for (i = 0 ; i < limit_1 ; i++) {
        v_aux = v_w[ i ] - v_w[ i ] * v_w[ i ] * (grad_m_w[ i ] * grad_m_w[ i ] - 2 * grad_v_w[ i ]);
        m_aux = m_w[ i ] + v_w[ i ] * grad_m_w[ i ];
        if (v_aux > 1e-100 && m_aux != NAN && v_aux != NAN) {
            m_w[ i ] = m_aux;
            v_w[ i ] = v_aux;
        }
    }

    /* We update the noise variables */

    n->b_noise = b_new;
    n->a_noise = a_new;
}

/**
 * Function that refines the approximate factor for the prior.
 *
 * @param network   Pointer to the network.
 *
 */

void refine_prior(NETWORK *n) {

    int i;

    double v_w_nat, m_w_nat, v_w_cav_nat, m_w_cav_nat, v_w_cav, m_w_cav,
        a_w_nat, b_w_nat, a_w_cav_nat, b_w_cav_nat, a_w_cav, b_w_cav,
        v, v1, v2, logZ, logZ1, logZ2, d_logZ_d_m_w_cav, d_logZ_d_v_w_cav,
        m_w_new, v_w_new, a_w_new, b_w_new, v_w_new_nat, m_w_new_nat,
        a_w_new_nat, b_w_new_nat;

    /* We iterate over the layers refining the prior */
    
    for (i = 0 ; i < n->size_w - 2 ; i++) {
        v_w_nat = 1.0 / n->v_w[ i ];
        m_w_nat = n->m_w[ i ] / n->v_w[ i ];
        v_w_cav_nat = v_w_nat - n->v_w_hat_nat[ i ];
        m_w_cav_nat = m_w_nat - n->m_w_hat_nat[ i ];

        v_w_cav = 1.0 / v_w_cav_nat;
        m_w_cav = m_w_cav_nat / v_w_cav_nat;
        a_w_nat = n->a_prior - 1;
        b_w_nat = -n->b_prior;
        a_w_cav_nat = a_w_nat - n->a_w_hat_nat[ i ];
        b_w_cav_nat = b_w_nat - n->b_w_hat_nat[ i ];
        a_w_cav = a_w_cav_nat + 1;
        b_w_cav = -b_w_cav_nat;

        if (v_w_cav > 0 && b_w_cav > 0 && a_w_cav > 1 && v_w_cav < 1e6) {

            v = v_w_cav + b_w_cav / (a_w_cav - 1);
            v1  = v_w_cav + b_w_cav / a_w_cav;
            v2  = v_w_cav + b_w_cav / (a_w_cav + 1);
            logZ = -0.5 * log(v) - 0.5 * m_w_cav * m_w_cav / v;
            logZ1 = -0.5 * log(v1) - 0.5 * m_w_cav * m_w_cav / v1;
            logZ2 = -0.5 * log(v2) - 0.5 * m_w_cav * m_w_cav / v2;
            d_logZ_d_m_w_cav = -m_w_cav / v;
            d_logZ_d_v_w_cav = -0.5 / v + 0.5 * m_w_cav * m_w_cav / (v * v);
            m_w_new = m_w_cav + v_w_cav * d_logZ_d_m_w_cav;
            v_w_new = v_w_cav - v_w_cav * v_w_cav * (d_logZ_d_m_w_cav * d_logZ_d_m_w_cav - 2 * d_logZ_d_v_w_cav) ;
            a_w_new = 1.0 / (exp(logZ2 - 2 * logZ1 + logZ) * (a_w_cav + 1) / a_w_cav - 1.0);
            b_w_new = 1.0 / (exp(logZ2 - logZ1) * (a_w_cav + 1) / (b_w_cav) - exp(logZ1 - logZ) * a_w_cav / b_w_cav);
            v_w_new_nat = 1.0 / v_w_new;
            m_w_new_nat = m_w_new / v_w_new;
            a_w_new_nat = a_w_new - 1;
            b_w_new_nat = -b_w_new;

            n->m_w_hat_nat[ i ] = m_w_new_nat - m_w_cav_nat;
            n->v_w_hat_nat[ i ] = v_w_new_nat - v_w_cav_nat;
            n->a_w_hat_nat[ i ] = a_w_new_nat - a_w_cav_nat;
            n->b_w_hat_nat[ i ] = b_w_new_nat - b_w_cav_nat;

            n->m_w[ i ] = m_w_new;
            n->v_w[ i ] = v_w_new;

            n->a_prior = a_w_new;
            n->b_prior = b_w_new;
        }
    }
}

/**
 * Function that performs one learning epoch.
 *
 * @param network       Pointer to the network.
 * @param x             Pointer to the input feature matrix.
 * @param y             Pointer to the target vector.
 * @param n_datapoints  Number of data points in the training set.
 * @param d             Dimensionality of the training data.
 * @param permutation   A random permuation of the input data.
 *
 */

NETWORK *one_learning_epoch(NETWORK *n, double *x, double *y, int n_datapoints, int d, int *permutation) {

    int i, index;

    /* We do one ADF upte for each datapoint */

    for (i = 0 ; i < n_datapoints ; i++) {
        index = permutation[ i ] * d;
        do_ADF_update(n, x + index, y[ permutation[ i ] ]);
        
        if (i % 1000 == 0) {
            printf(".");
            fflush(stdout);
        }
    }
    printf("\n");
    fflush(stdout);

    /* We refine the prior */

    refine_prior(n);

    /* We are done */

    return n;
}

/**
 * Function that predicts deterministically.
 *
 * @param network       Pointer to the network.
 * @param x_test        Pointer to the input feature matrix.
 * @param y_test        Pointer to store the predictions.
 * @param n_datapoints  Number of data points in the test set.
 * @param d             Dimensionality of the test data.
 *
 */

NETWORK *predict_deterministic(NETWORK *n, double *x_test, double *y_test, int n_datapoints, int d) {

    int i, j, index, limit_1;

    double *m_z;

    for (i = 0 ; i < n_datapoints ; i++) {

        /* We initialize the non-linear activations of the input layer with the data */

        limit_1 = n->start_a[ 1 ];
        m_z = n->m_z;
        for (j = 0 ; j < limit_1 ; j++) {
            index = i * d + j;
            m_z[ j ] = x_test[ index ];
        }
        m_z[ j ] = 1.0;

        /* We do a forward pass */

        limit_1 = n->n_hidden_layers + 2;
        for (j = 1 ; j < limit_1 ; j++)
            deterministc_forward_PBP(n, j);

        /* We store the results */

        m_z = n->m_z + n->start_z[ n->n_hidden_layers + 1 ];

        y_test[ i ] = m_z[ 0 ];
    }
    
    /* We are done */

    return n;
}

/**
 * Function that predicts.
 *
 * @param network       Pointer to the network.
 * @param x_test        Pointer to the input feature matrix.
 * @param m_test        Pointer to store the predictive mean.
 * @param v_test        Pointer to store the predictive variance.
 * @param v_noise       Pointer to store the noise variance.
 * @param n_datapoints  Number of data points in the test set.
 * @param d             Dimensionality of the test data.
 *
 */

NETWORK *predict(NETWORK *n, double *x_test, double *m_test, double *v_test, double *v_noise, int n_datapoints, int d) {

    int i, j, index, limit_1;

    double *m_z, *v_z;

    for (i = 0 ; i < n_datapoints ; i++) {

        /* We initialize the non-linear activations of the input layer with the data */

        limit_1 = n->start_a[ 1 ];
        m_z = n->m_z;
        v_z = n->v_z;
        for (j = 0 ; j < limit_1 ; j++) {
            index = i * d + j;
            m_z[ j ] = x_test[ index ];
            v_z[ j ] = 0.0;
        }
        m_z[ j ] = 1.0;
        v_z[ j ] = 0.0;

        /* We do a forward pass */

        limit_1 = n->n_hidden_layers + 2;
        for (j = 1 ; j < limit_1 ; j++)
            forward_PBP(n, j);

        /* We store the results */

        m_z = n->m_z + n->start_z[ n->n_hidden_layers + 1 ];
        v_z = n->v_z + n->start_z[ n->n_hidden_layers + 1 ];

        m_test[ i ] = m_z[ 0 ];
        v_test[ i ] = v_z[ 0 ];
    }
    
    /* We store the noise value */

    *v_noise = n->b_noise / (n->a_noise - 1);
    
    /* We are done */

    return n;
}

/* Function that returns the parameters of the network */

void get_params(NETWORK *n, double *sample_w_out, double *m_w_out, double *v_w_out, double *m_w_hat_nat_out, double *v_w_hat_nat_out,
    double *a_w_hat_nat_out, double *b_w_hat_nat_out, double *a_noise_out, double *b_noise_out, double *a_prior_out,
    double *b_prior_out, int *neurons_per_layer_out) {

    int i, limit_1;

    /* We copy the parameters */

    limit_1 = n->size_w - 2;
    for (i = 0 ; i < limit_1 ; i++) {
        sample_w_out[ i ] = n->sample_w[ i ];
        m_w_out[ i ] = n->m_w[ i ];
        v_w_out[ i ] = n->v_w[ i ];
        m_w_hat_nat_out[ i ] = n->m_w_hat_nat[ i ];
        v_w_hat_nat_out[ i ] = n->v_w_hat_nat[ i ];
        a_w_hat_nat_out[ i ] = n->a_w_hat_nat[ i ];
        b_w_hat_nat_out[ i ] = n->b_w_hat_nat[ i ];
    }
    
    *a_noise_out = n->a_noise;
    *b_noise_out = n->b_noise;
    *a_prior_out = n->a_prior;
    *b_prior_out = n->b_prior;

    limit_1 = n->n_hidden_layers + 2;
    for (i = 0 ; i < limit_1 ; i++) {
        neurons_per_layer_out[ i ] = n->neurons_per_layer[ i ];
    }
}

/* Function that returns the sizes of the parameters of the network */

void get_size_params(NETWORK *n, int *size_weihts, int *n_layers) {

    *size_weihts = n->size_w - 2;
    *n_layers = n->n_hidden_layers + 2;
}

/* Function that sets the parameters of the network */

void set_params(NETWORK *n, double *sample_w_in, double *m_w_in, double *v_w_in, double *m_w_hat_nat_in, double *v_w_hat_nat_in,
    double *a_w_hat_nat_in, double *b_w_hat_nat_in, double a_noise_in, double b_noise_in, double a_prior_in, double b_prior_in) {

    int i, limit_1;

    /* We copy the parameters */

    limit_1 = n->size_w - 2;
    for (i = 0 ; i < limit_1 ; i++) {
        n->sample_w[ i ] = sample_w_in[ i ];
        n->m_w[ i ] = m_w_in[ i ];
        n->v_w[ i ] = v_w_in[ i ];
        n->m_w_hat_nat[ i ] = m_w_hat_nat_in[ i ];
        n->v_w_hat_nat[ i ] = v_w_hat_nat_in[ i ];
        n->a_w_hat_nat[ i ] = a_w_hat_nat_in[ i ];
        n->b_w_hat_nat[ i ] = b_w_hat_nat_in[ i ];
    }
    
    n->a_noise = a_noise_in;
    n->b_noise = b_noise_in;
    n->a_prior = a_prior_in;
    n->b_prior = b_prior_in;
}
