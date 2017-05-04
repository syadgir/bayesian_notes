// Logit Model
//
  // y ~ Bernoulli(p)
// p = a + X B
// b0 \sim cauchy(0, 10)
// b \sim cauchy(0, 2.5)
data {
  // number of observations
  int N;
  // response
  // vectors are only real numbers
  // need to use an array
  int<lower = 0, upper = 1> y[N];
  // number of columns in the design matrix X
  int K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
}
transformed data {
  # default scales same as rstanarm
  # assume data is centered and scaled
  real<lower = 0.0> a_scale;
  vector<lower = 0.0>[K] b_scale;
  a_scale = 10.0;
  b_scale = rep_vector(2.5, K);
}
parameters {
  // regression coefficient vector
  real a;
  vector[K] b;
}
transformed parameters {
  vector<lower = 0.0, upper = 1.0>[N] p;
  p = inv_logit(a + X * b);
}
model {
  // priors
  a ~ normal(0.0, a_scale);
  b ~ normal(0.0, b_scale);
  // likelihood
  y ~ binomial(1, p);
}
generated quantities {
  // simulate data from the posterior
  vector[N] y_rep;
  // log-likelihood posterior
  vector[N] log_lik;
  for (i in 1:N) {
    y_rep[i] = binomial_rng(1, p[i]);
    log_lik[i] = binomial_lpmf(y[i] | 1, p[i]);
  }
}
