"""Test STAN."""
from utils import StanFit
from simulate import generate
import matplotlib.pyplot as pl
import numpy as np
np.random.seed(1234)


# The STAN code
stan_code = """
functions {
    // A trivial diff eq controlling the "tidal" evolution
    real[] tidal(real t, real[] y, real[] params, real[] x_r, int[] x_i) {

        // TODO!

        real dydt[2];

        // da / dt
        dydt[1] = 0;

        // de /dt
        dydt[2] = 0;

        return dydt;

    }
}
data {
    // Number of systems
    int<lower=1> N;

    // Observed eccentricity and semi-major axis, with uncertainty
    real dat_a[N];
    real<lower=0> dat_siga[N];
    real dat_e[N];
    real<lower=0> dat_sige[N];

    // Other observables, with uncertainty
    real dat_r[N];
    real<lower=0> dat_sigr[N];
    real dat_t[N];
    real<lower=0> dat_sigt[N];
    real dat_m[N];
    real<lower=0> dat_sigm[N];
    real dat_M[N];
    real<lower=0> dat_sigM[N];
    real dat_R[N];
    real<lower=0> dat_sigR[N];

    // Diff eq solution times
    int<lower=1> K;
    real solution_times[K];
}
transformed data {
    real x_r[0];
    int x_i[0];
}
parameters {
    // The population-level variables we're trying to constrain
    real par_alpha;
    real par_beta;
    real<lower=0> par_sigma;

    // The `true` parameters for each system
    real par_logtau[N];
    real par_a0[N];
    real par_e0[N];
    real par_t[N];
    real par_m[N];
    real par_r[N];
    real par_M[N];
    real par_R[N];
}
model {
    // Dummy variables
    real afinal;
    real efinal;
    real sig_tau;
    real ae_init[2];
    real ae_final[K,2];
    real params[6];

    // Enforce some generous priors on our hyperparameters
    par_alpha ~ uniform(0, 10);
    par_beta ~ uniform(0, 10);
    par_sigma ~ uniform(0, 1);

    // Go through the ensemble
    for (n in 1:N) {

        // Enforce the prior on the observables
        par_t[n] ~ normal(dat_t[n], dat_sigt[n]);
        par_r[n] ~ normal(dat_r[n], dat_sigr[n]);
        par_m[n] ~ normal(dat_m[n], dat_sigm[n]);
        par_R[n] ~ normal(dat_R[n], dat_sigR[n]);
        par_M[n] ~ normal(dat_M[n], dat_sigM[n]);

        // Enforce the prior on tau
        par_logtau[n] ~ normal(par_alpha - par_beta * par_r[n], par_sigma);

        // Enforce the prior on the initial conditions
        par_a0[n] ~ uniform(0.01, 0.3);
        par_e0[n] ~ beta(0.867, 3.03);

        // Evolve the system and compute the value of `a` and `e`
        // closest to the system age.
        // TODO: This can be optimized!
        params[1] = par_M[n];
        params[2] = par_m[n];
        params[3] = par_R[n];
        params[4] = 10 ^ par_logtau[n];
        params[5] = 1;
        params[6] = par_t[n];
        ae_init[1] = par_a0[n];
        ae_init[2] = par_e0[n];
        ae_final = integrate_ode_rk45(tidal, ae_init, 0,
                                      solution_times, params, x_r, x_i);
        for (k in 1:K) {
            if (solution_times[k] > par_t[n]) {
                afinal = ae_final[k,1];
                efinal = ae_final[k,2];
                break;
            }
        }

        // Compute the likelihood of observing `a` and `e`
        dat_a[n] ~ normal(afinal, dat_siga[n]);
        dat_e[n] ~ normal(efinal, dat_sige[n]);
    }
}
"""

data, init = generate(N=100)

# Go!
fit = StanFit(stan_code, data, warmup=2000, niter=5000, n_jobs=1, init=init)

# Plot the STAN output
print(fit)
fit.plot(('alpha', 'beta', 'sigma'))
pl.show()
