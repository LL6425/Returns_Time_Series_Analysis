function [sigma, loglik] = fil(vY, phi, nu, mu, delta, kappa)

%filter

[cT, cN] = size(vY);

sigma = zeros(cT,1); 
u = zeros(cT,1);        


sigma(1) = delta*(1-phi); 
u(1)=((((nu+1) * (vY(1))^2) / (((nu-2)*(sigma(1)))+(vY(1))^2))) -1;


loglik =0; 

for t=1:cT-1
     
    sigma(t+1) =  delta + phi * sigma(t) + kappa * sigma(t) * u(t);
  
    
    u(t+1)=((((nu+1) * (vY(t))^2) / (((nu-2)*(sigma(t)))+(vY(t))^2))) -1;
 
    
    loglik = loglik +log(gamma((nu+1)/2)) - log(gamma(nu/2))-0.5*log(pi)-0.5* log(nu-2) - 0.5* log(sigma(t))- ((nu+1)/2) * log(1+(((vY(t) - mu)^2)/((nu-2)*(sigma(t)))));
   
    
end
 

