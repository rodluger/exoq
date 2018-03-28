"""
Another Stan test. This time we actually solve a (trivial) diff eq.
"""
from collections import OrderedDict
from utils import StanFit, normal
import matplotlib.pyplot as pl
import numpy as np
np.random.seed(1234)


# The STAN code
test_code = """
functions {
    // A trivial diff eq controling the "tidal" evolution
    real[] tidal(real t, real[] y, real[] Q, real[] x_r, int[] x_i) {
        real dydt[2];

        // da / dt
        dydt[1] = -0.1 * Q[1];

        // de /dt
        dydt[2] = -0.0025 * Q[1];

        return dydt;
    }
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
    real m;
    real b;
    real logsigQ;

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
    real sigQ;
    real ae_init[2];
    real ae_final[K,2];
    real Qn[1];

    // Enforce some generous priors on m, b, and logsigQ
    m ~ uniform(-10, 10);
    b ~ uniform(-10, 10);
    logsigQ ~ uniform(-3, 0);
    sigQ = 10 ^ logsigQ;

    // Go through the ensemble
    for (n in 1:N) {

        // Enforce the prior on the radius and age
        ttrue[n] ~ normal(t[n], sigt[n]);
        rtrue[n] ~ normal(r[n], sigr[n]);

        // Enforce the prior on Q
        Q[n] ~ normal(m * rtrue[n] + b, sigQ);
        Qn[1] = Q[n];

        // Enforce the prior on the initial conditions
        a0[n] ~ normal(mua0, siga0);
        e0[n] ~ normal(mue0, sige0);

        // Evolve the system and compute the value of `a` and `e`
        // closest to the system age.

        ae_init[1] = a0[n];
        ae_init[2] = e0[n];
        ae_final = integrate_ode_rk45(tidal, ae_init, 0,
                                      solution_times, Qn, x_r, x_i);

        //aefinal[k,1] = a0[n] - 0.1 * Q[n] * solution_times[k];
        //aefinal[k,2] = e0[n] - 0.0025 * Q[n] * solution_times[k];

        for (k in 1:K) {
            if (solution_times[k] > ttrue[n]) {
                afinal = ae_final[k,1];
                efinal = ae_final[k,2];
                break;
            }
        }

        // Compute the likelihood of observing `a` and `e`
        a[n] ~ normal(afinal, siga[n]);
        e[n] ~ normal(efinal, sige[n]);
    }
}
"""


def gen_data(m, b, logsigQ, mua0, siga0, mue0, sige0):
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
    - m, b, logsigQ: Model parameters for tidal evolution
                  We assume a planet's Q is drawn from

                        Q ~ N(m * r + b, 10 ** logsigQ)

                  We regress on all three.

    Model
    -----
    For simplicity, let's assume that Q simply controls the (linear) rate
    at which the semi-major axis and the eccentricity get damped, so:

        a ~ N(a0 - 0.1 * Q * t, siga)
        e ~ N(e0 - 0.0025 * Q * t, sige)


    Diff eq
    -------
    We can write the evolution as a trivial differential equation for testing:

        da / dt = -0.1 * Q
        de / dt = -0.0025 * Q

    """
    # Randomize a radius and an age
    t = 9 + np.random.random()
    r = 4 + np.random.random()

    # Draw Q
    Q = normal(m * r + b, 10 ** logsigQ)

    # Draw the initial values
    a0 = normal(mua0, siga0)
    e0 = normal(mue0, sige0)

    # Fix the uncertainty on the observed variables for now
    siga = 0.01
    sige = 0.01
    sigt = 0.01
    sigr = 0.01

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
logsigQ = -1

# Priors we know
mua0 = 30.
siga0 = 0.01
mue0 = 0.75
sige0 = 0.01

# Number of planets
N = 10

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
        gen_data(m, b, logsigQ, mua0, siga0, mue0, sige0)

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
pl.show()

# Initial values
Q = np.array([normal(r[n] + 1, 0.1) for n in range(N)])
Q[Q < 1] = 1
ttrue = np.array([normal(t[n], sigt[n]) for n in range(N)])
ttrue[ttrue < 1] = 1
rtrue = np.array([normal(r[n], sigr[n]) for n in range(N)])
rtrue[rtrue < 1] = 1
a0 = np.array([normal(mua0, siga0) for n in range(N)])
a0[a0 < 1] = 1
e0 = np.array([normal(mue0, sige0) for n in range(N)])
e0[e0 < 0.001] = 0.001
init = [dict(m=1,
             b=1,
             logsigQ=-1,
             Q=Q,
             ttrue=ttrue,
             rtrue=rtrue,
             a0=a0,
             e0=e0)]

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
data['K'] = 50
data['solution_times'] = np.linspace(0, 11, 51)[1:]

# Go!
fit = StanFit(test_code, data, warmup=2000, niter=5000, n_jobs=1, init=init)

# Plot the STAN output
print(fit)
fit.plot()
pl.show()
