"""
Another Stan test.

Here I regress on data from 200 'planets' to
try to infer the dependence of the tidal
quality factor on the planet radius.
"""
from collections import OrderedDict
from utils import StanFit, normal
import matplotlib.pyplot as pl
import numpy as np
np.random.seed(1234)


# The STAN code
test_code = """
functions {
}
data {
    // Number of planets
    int<lower=1> N;

    // Observed eccentricity and semi-major axis, with uncertainty
    real a[N];
    real<lower=0> siga[N];
    real e[N];
    real<lower=0> sige[N];

    // Observed radius and age, with uncertainty
    real r[N];
    real<lower=0> sigr[N];
    real t[N];
    real<lower=0> sigt[N];

    // Initial distribution parameters
    real mua0;
    real siga0;
    real mue0;
    real sige0;
}
transformed data {
    real x_r[0];
    int x_i[0];
}
parameters {
    // The population-level variables we're trying to constrain
    real m;
    real b;
    real<lower=0> sigQ;

    // The `true` parameters for each planet
    real Q[N];
    real a0[N];
    real e0[N];
    real ttrue[N];
    real rtrue[N];
}
model {
    // Dummy variables
    real afinal;
    real efinal;

    // Enforce some generous priors on m, b, and sigQ
    m ~ uniform(-10, 10);
    b ~ uniform(-10, 10);

    // TODO: log uniform?
    sigQ ~ uniform(0.001, 1.);

    // Go through the ensemble
    for (n in 1:N) {

        // Enforce the prior on the radius and age
        ttrue[n] ~ normal(t[n], sigt[n]);
        rtrue[n] ~ normal(r[n], sigr[n]);

        // Enforce the prior on Q
        Q[n] ~ normal(m * rtrue[n] + b, sigQ);

        // Enforce the prior on the initial conditions
        a0[n] ~ normal(mua0, siga0);
        e0[n] ~ normal(mue0, sige0);

        // Evolve the system
        afinal = a0[n] - 0.1 * Q[n] * ttrue[n];
        efinal = e0[n] - 0.0025 * Q[n] * ttrue[n];

        // Compute the likelihood of observing `a` and `e`
        a[n] ~ normal(afinal, siga[n]);
        e[n] ~ normal(efinal, sige[n]);
    }
}
"""


def gen_data(m, b, sigQ, mua0, siga0, mue0, sige0):
    """
    Generate synthetic data.

    Observed variables
    ------------------
    - a, siga: Present-day semi-major axis and uncertainty
    - e, sige: Present-day eccentricity and uncertainty
    - r, sigr: Planet radius and uncertainty
    - t, sigt: System age and uncertainty

    Priors
    ------
    - mua0, siga0: Population semi-major axis distribution at t = 0:

                   a0 ~ N(mua0, siga0)

    - mue0, sige0: Population eccentricity distribution at t = 0:

                   e0 ~ N(mue0, sige0)

    Latent variables
    ----------------
    - m, b, sigQ: Model parameters for tidal evolution
                  We assume a planet's Q is drawn from

                        Q ~ N(m * r + b, sigQ)

                  We regress on all three.

    Model
    -----
    For simplicity, let's assume that Q simply controls the (linear) rate
    at which the semi-major axis and the eccentricity get damped, so:

        a ~ N(a0 - 0.1 * Q * t, siga)
        e ~ N(e0 - 0.0025 * Q * t, sige)

    """
    # Randomize a radius and an age
    t = 10 * np.random.random()
    r = 5 * np.random.random()

    # Draw Q
    Q = normal(m * r + b, sigQ)

    # Draw the initial values
    a0 = normal(mua0, siga0)
    e0 = normal(mue0, sige0)

    # Fix the uncertainty on the observed variables for now
    siga = 0.1
    sige = 0.01
    sigt = 0.1
    sigr = 0.1

    # Draw the observed variables
    a = normal(a0 - 0.1 * Q * t, siga)
    e = normal(e0 - 0.0025 * Q * t, sige)

    # Add noise to the age and radius
    t = normal(t, sigt)
    r = normal(r, sigr)

    return t, sigt, r, sigr, a, e, siga, sige, a0, e0, Q


# Values we're going to try to recover
m = 3.
b = 2.5
sigQ = 0.1

# Priors we know
mua0 = 30.
siga0 = 1.
mue0 = 0.75
sige0 = 0.05

# Number of planets
N = 200

# Generate our fake data
t = np.zeros(N)
r = np.zeros(N)
a = np.zeros(N)
e = np.zeros(N)
sigt = np.zeros(N)
sigr = np.zeros(N)
siga = np.zeros(N)
sige = np.zeros(N)
a0 = np.zeros(N)
e0 = np.zeros(N)
Q = np.zeros(N)
for n in range(N):
    t[n], sigt[n], r[n], sigr[n], a[n], e[n], siga[n], \
        sige[n], a0[n], e0[n], Q[n] = \
        gen_data(m, b, sigQ, mua0, siga0, mue0, sige0)

# Plot the distributions
fig, ax = pl.subplots(3, figsize=(4, 6))
fig.subplots_adjust(hspace=0.5)
ax[0].set_title('Semi-major axis')
ax[0].hist(a0, histtype='step', bins=30, label='Initial')
ax[0].hist(a, histtype='step', bins=30, label='Observed')
ax[0].legend(loc='best')
ax[1].set_title('Eccentricity')
ax[1].hist(e0, histtype='step', bins=30, label='Initial')
ax[1].hist(e, histtype='step', bins=30, label='Observed')
ax[1].legend(loc='best')
ax[2].set_title('Tidal Q')
ax[2].hist(Q, bins=30)
# pl.show()

# Initial values
init = [dict(m=1,
             b=1,
             sigQ=0.1,
             Q=np.array([normal(r[n] + 1, 0.1) for n in range(N)]),
             ttrue=t,
             rtrue=r,
             a0=np.array([normal(mua0, siga0) for n in range(N)]),
             e0=np.array([normal(mue0, sige0) for n in range(N)]))]

# Data dictionary
data = OrderedDict()
data['N'] = N
data['a'] = a
data['siga'] = siga
data['e'] = e
data['sige'] = sige
data['r'] = r
data['sigr'] = sigr
data['t'] = t
data['sigt'] = sigt
data['mua0'] = mua0
data['siga0'] = siga0
data['mue0'] = mue0
data['sige0'] = sige0

# Go!
fit = StanFit(test_code, data, warmup=2000, niter=10000, n_jobs=1, init=init)

# Plot the STAN output
print(fit)
fit.plot()
pl.show()
