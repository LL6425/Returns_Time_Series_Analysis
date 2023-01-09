function [loglik] = l_like(parameters,data)

% this function does compute the log-likelihood of lambda, based on the
% likelihood of a Student t distribution where the location is constant
% and the scale is an exponential function of a time varying parameter
% lambda_t whose evolutoion (along with the likelihood function) is
% contained in the file filter_scale.m 

phi = parameters(1);
nu = parameters(2); 
mu = parameters(3);
delta = parameters(4);
theta = parameters(5); 

 
[lambda, loglik] = filter_scale(data, phi, nu, mu, delta, theta); 
 
 

loglik = - loglik;