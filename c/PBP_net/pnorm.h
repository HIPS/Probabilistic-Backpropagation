/*
 * Module that implements the cdf and density functions of the standard
 * Gaussian distribution. The code is extracted from the R implementation
 * of these functions.
 *
 * Author: Jose Miguel Hernandez Lobato
 * Date: 10 March 2015
 *
 */

#ifndef __PNORM
#define __PNORM

double log_pnorm(double x);
double log_dnorm(double x);
double dnorm(double x);
double pnorm(double x);

#endif
