% Time series selected : Visa stock


% 0. Clean the work space and Import the data

clear all
close all
clc

% Visa stock price in the last five years

% Data from Yahoo Finance

T=readtable('V.csv');  % Table columns: Date , Open , High , Low , Close , AdjClose , Volume

% selecting the AdjClose column for the time series

P=table2array(T(:,6));

% 1. Preliminary analysis

figure;
plot(P),title('Visa stock price in the period 27/06/2017-24/06/2022');

figure;
autocorr(P,120),title('ACF of Visa P');

% The plot of the time series highlights a non-stationary process in mean.
% Moreover, the ACF of the stock price Pt time series is very persistent
% and it decreases linearly and slowly.

% From the previous results we can hypothesise that the series behaves like
% a local level model or a random walk.

p=log(P);
y=diff(p);


figure;
plot(y),title('Visa stock returns as diff(log(P))');

figure;
subplot(3,1,1),autocorr(y,80),title('ACF Visa stock returns');
subplot(3,1,2),autocorr(abs(y),80),title('ACF absolute value of Visa stock returns');
subplot(3,1,3),autocorr(y.^2,80),title('ACF squared Visa stock returns');




%Facts:
% - Stationarity in mean
% - Considerable amount of extreme observations
% - Time varying volatility
% - Volatility clustering
% - Autocorrelation weak but significant  
% - Non-independence 



s=skewness(y);   % s = -0.1130

k=kurtosis(y);   % k = 12.1871 >> 3

h=jbtest(y);     % h = 1 so the null hypothesis of normality is rejected



% 2. Models


varTypes = ["string","double","double","double"];
varNames = ["Model","LogL","AIC","BIC"];
Results = table('Size',[4,4],'VariableTypes',varTypes,'VariableNames',varNames);

%  2.1 ARCH(1)

Mdl = garch(0,1);
[EstMdl,EstParamCov,logL_arch,info] = estimate(Mdl,y);
[s_arch,logL_arch] = infer(EstMdl,y);


[aic_arch,bic_arch]=aicbic(logL_arch,2,1257);

Results(1,:)={'ARCH(1,1)',logL_arch,aic_arch,bic_arch};

figure;
subplot(2,1,1),plot(y.^2),title('ARCH(1), estimates in red');
hold on;
plot(s_arch,'r');
subplot(2,1,2),plot(abs(y));
hold on;
plot(sqrt(s_arch),'r');

%From both the graphs we can notice a rough fitting


v_arch=y.^2-s_arch;

figure;
plot(v_arch),title('ARCH(1) innovation errors');

figure;
autocorr(v_arch,80),title('ARCH(1) ACF innovation errors');

% Innovation errors are autocorrelated so the model is not correctly
% specified



%  2.1 GARCH(1,1)

Mdl = garch(1,1);
[EstMdl,EstParamCov,logL_garch,info] = estimate(Mdl,y);
[s_garch,logL_garch] = infer(EstMdl,y);

% Stationarity condition satisfied since a+b < 1

[aic_garch,bic_garch]=aicbic(logL_garch,3,1257);

Results(2,:)={'GARCH(1,1)',logL_garch,aic_garch,bic_garch};


figure;
subplot(2,1,1),plot(y.^2),title('GARCH(1,1), estimates in red');
hold on;
plot(s_garch,'r');
subplot(2,1,2),plot(abs(y));
hold on;
plot(sqrt(s_garch),'r');


v_garch=y.^2-s_garch;

figure;
plot(v_garch),title('GARCH(1,1) innovation errors');

figure;
autocorr(v_garch,120),title('GARCH(1,1) ACF innovation errors');

% Low but significant autocorrelation of innovation errors for certain
% lags(3,4,6,9,12,15)


% 2.2 Beta-t-Garch(1,1)

% par_order=phi, nu, mu, delta, kappa
par0 = [0.4  4  0   0  0.1];

options = optimset('Algorithm','interior-point','Display','iter',...
   'AlwaysHonorConstraints','bounds','TolFun' ,...
     3000, 'TolFun', 1e-9, 'TolX', 1e-9);
 
lb = [0.00001 2 -Inf 0.00001 0.00001] ; 
ub = [1 Inf Inf  Inf 2];   

[par_btg, fval, exitflag] = fmincon('lllh',par0,[],[],[],[],lb, ub,[],...,
                        options, y);
                    


[s_btg,loglik_btg] = fil(y, par_btg(1), par_btg(2), par_btg(3), par_btg(4), par_btg(5));

likelihood_btg = -fval; 

[aic_btg,bic_btg]=aicbic(loglik_btg,5,1257);

Results(3,:)={'beta-t-GARCH(1,1)',loglik_btg,aic_btg,bic_btg};                  
                    
figure;
subplot(2,1,1),plot(y.^2),title('beta-t-GARCH(1,1), estimates in red');
hold on;
plot(s_btg,'r');
subplot(2,1,2),plot(abs(y));
hold on;
plot(sqrt(s_btg),'r');  

v_btg=y.^2-s_btg;

figure;
plot(v_btg),title('beta-t-GARCH(1,1) innovation errors');

figure;
autocorr(v_btg,120),title('beta-t-GARCH(1,1) ACF innovation errors');

hat_nu_btg=par_btg(2);

ut_btg=((hat_nu_btg+1).*y.^2)./((hat_nu_btg-2).*s_btg+y.^2)-1;

figure;
subplot(3,1,1),autocorr(ut_btg,120),title('ACF u_t beta-t-Garch(1,1)');
subplot(3,1,2),autocorr(ut_btg.^2,120),title('ACF u_t^2 beta-t-Garch(1,1)');
subplot(3,1,3),autocorr(abs(ut_btg),120),title('ACF |u_t| beta-t-Garch(1,1)');

%The code of the beta-t-Garch part is extremely unstable with respect to
%changes in par0

% 2.4 Beta-t-EGarch(1,1)

options = optimset('Algorithm','interior-point','Display','iter',...
   'AlwaysHonorConstraints','bounds','TolFun' ,...
     3000, 'TolFun', 1e-9, 'TolX', 1e-9);
 
initial_parameters = [0.1   14  0    0.1    0.2];

lb = [-1 0 -Inf -Inf -2] ; 
ub = [1 Inf Inf  Inf 2];

[par_btEg, fval, exitflag] = fmincon('l_like',initial_parameters,[],[],[],[],lb, ub,[],...,
             options, y);
 
[lambda,loglik_btEg] = filter_scale(y, par_btEg(1), par_btEg(2), par_btEg(3), par_btEg(4), par_btEg(5));  

likelihood_btEg=-fval;

[aic_btEg,bic_btEg]=aicbic(loglik_btEg,5,1257);

Results(4,:)={'beta-t-EGARCH(1,1)',loglik_btEg,aic_btEg,bic_btEg};

lambda_sel=lambda(10:end,1);
y_sel=y(10:end,1);

% This selection is helpful to have a 'proper' scale for the plot
% Otherwise the first ten estimates of exp(2*lambda) are out of scale

figure;
subplot(2,1,1),plot(y_sel.^2),title('beta-t-EGARCH(1,1), estimates in red');
hold on;
plot(exp(2*lambda_sel),'r');
subplot(2,1,2),plot(abs(y_sel));
hold on;
plot(exp(lambda_sel),'r');  

hat_phi = par_btEg(1);
hat_nu = par_btEg(2);
hat_mu = par_btEg(3);
hat_delta = par_btEg(4);
hat_theta = par_btEg(5);

% Estimated degrees of freedom hat_nu=4.3869(<<300) indicates that the Student-T 
% conditional distribution is a good assumption. 

b=(y.^2./(hat_nu * exp(2*lambda)))./(1+(y.^2./(hat_nu * exp(2*lambda))));
ut=(hat_nu+1) * b - 1;
wt=y.^2./exp(2*lambda) - 1; 

figure;
subplot(2,1,1),autocorr(ut,120),title('ACF u_t beta-t-EGarch');
subplot(2,1,2),autocorr(wt,120),title('ACF w_t beta-t-EGarch');

figure;
subplot(2,1,1),autocorr(ut.^2,120),title('ACF u_t^2 beta-t-EGarch');
subplot(2,1,2),autocorr(wt.^2,120),title('ACF w_t^2 beta-t-EGarch');

figure;
subplot(2,1,1),autocorr(abs(ut),120),title('ACF |u_t| beta-t-EGarch');
subplot(2,1,2),autocorr(abs(wt),120),title('ACF |w_t| beta-t-EGarch');

% Both w and u are uncorrelated and independent
 

% 3. Summary & Comparison

Results

% As we can see in the table the best-fitting model is the
% beta-t-Garch(1,1)

figure;
plot(y(10:end,:).^2),title('All models graph');
hold on;
plot(s_arch(10:end,:),'r');
hold on;
plot(s_garch(10:end,:),'y');
hold on;
plot(s_btg(10:end,:),'g');
hold on;
plot(exp(2*lambda(10:end,:)),'c');
legend('y^2','\sigma^2 ARCH(1)','\sigma^2 GARCH(1,1)','\sigma^2 beta-t-GARCH(1,1)','\sigma^2 beta-t-EGARCH(1,1)');


























