function [loglik] = l_like(parameters,data)

% Function computing the log-likelihood of lambda, based on the
% likelihood of a Student t distribution where the location is constant
% and the scale is an exponential function of a time varying parameter


phi = parameters(1);
nu = parameters(2); 
mu = parameters(3);
delta = parameters(4);
theta = parameters(5); 

 
[lambda, loglik] = filter_scale(data, phi, nu, mu, delta, theta); 
 
 

loglik = - loglik;
