
function [loglik] = lllh(parameters,vY)

%Loglikelihood

phi = parameters(1);
nu = parameters(2); 
mu = parameters(3);
delta = parameters(4);
kappa = parameters(5); 

 
[sigma, loglik] = fil(vY, phi, nu, mu, delta, kappa); 
 
 

loglik = - loglik;
