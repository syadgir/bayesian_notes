data {
  // number of observations
  int n;
  // response vector
  vector[n] y;
  // number of columns in the design matrix X
  int k;  ##number of betas
  // design matrix X
  matrix [n, k] X;
  // beta prior
  real b_loc;
  real<lower = 0.0> b_scale;
  // sigma prior
  real sigma_scale;
}
parameters {
  // regression coefficient vector
  vector[k] b;
  // scale of the regression errors
  real<lower = 0.0> sigma;
  real<lower = 1.0> nu;
}
transformed parameters {
  // mu is the observation fitted/predicted value
  // also called yhat
  vector[n] mu; 
  mu = X * b; ##sy: b is length k (multiple betas in there)
  ##sy: mu is a vector of the predicted values after the linear model/likelihood
}
model {
  // priors
  b ~ normal(b_loc, b_scale);
  sigma ~ cauchy(0, sigma_scale);
  nu ~ gamma(2, 0.1);
  // likelihood
  y ~ student_t(nu, mu, sigma);  ##the student t distribution is used often in bayesian models because it has larger tails than the normal distribution, which means that outliers will not be as influential.  W/ a normal distribtion, an extreme value will either pull the mean toward the outlier, or increase the standard error so that the fits on the central data points will be much less certain.  Cauchy is often used on sigma for this reason, and a double exponential/Laplace distribution has the similar effect, but does so by putting most weight on the median 
}
generated quantities {  ##sy: predictions--combine distributions of predictive variables (ie, variability in covariates) and distributions of output parameters (ie, uncertainty in the model) to get a predicted distribution for our outcome variable
  // simulate data from the posterior
  vector[n] y_rep;
  // log-likelihood values
  vector[n] log_lik;
  for (i in 1:n) {
    y_rep[i] = student_t_rng(nu, mu[i], sigma); ##this is a 'student t random number generator'-it's taking variance in mu (which is the predicted values of the posterior distribution)
    log_lik[i] = student_t_lpdf(y[i] | nu, mu[i], sigma);
  }

}
