function [lambda, loglik] = filter_scale(data, phi, nu, mu, delta, theta)

% This function gives the filter for estimating the conditional (log-)scale of a 
% score driven model for the location - in particolar for a Beta-t-EGARCH model where the 
% scale is the exponential of a time varying parameter lambda 
% and specifies the likelihood function to be maximised  

% input: the data and the static parameters associated with the model 
% typically the static parameters associated with the conditional
% distrubution assumed as a data generating process plus tthe static
% parameters that characterise the filtering recursion 
% 
% note that No initial condition for lambda is required as an input: we
% shall fix it here in the function file 

T = length(data);

% allocate space

lambda = zeros(T,1);   % time varying log-scale
b = zeros(T,1);        % beta random variable defining the score 
u = zeros(T,1);        % score 

% initial values for the dynamic recursions 

lambda(1) = delta * (1-phi); % unconditional mean of lambda 
b(1)=(data(1)^2/(nu*exp(2*lambda(1))))/(1+(data(1)^2/(nu*exp(2*lambda(1))))); % as a function of the first observation 
u(1)=(nu+1)*b(1)-1; % as a function of b(1)

loglik =0; 

for t=1:T-1
    
    lambda(t+1) = delta + phi * lambda(t) + theta * u(t);
    b(t+1) = (data(t+1)^2/(nu*exp(2*lambda(t+1))))/(1+(data(t+1)^2/(nu*exp(2*lambda(t+1)))));
    u(t+1) = (nu+1) * b(t+1) - 1;
    
    loglik = loglik +log(gamma((nu+1)/2)) - log(gamma(nu/2))-0.5*log(pi)-0.5* log(nu) - 0.5 * 2* lambda(t)- ((nu+1)/2) * log(1+((data(t) - mu)^2)/(nu*exp(2*lambda(t))));
   
    
end
 