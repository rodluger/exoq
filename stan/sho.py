"""2D Simple harmonic oscillator example."""

import pystan
import matplotlib.pyplot as pl
import pickle
from hashlib import md5
import numpy as np
from corner import corner


def StanModel(model_code, model_name=None, **kwargs):
    """
    Override `pystan.StanModel` to allow caching of the model.

    Based on
    https://pystan.readthedocs.io/en/latest/avoiding_recompilation.html
    """
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.sm'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.sm'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except (FileNotFoundError, pickle.UnpicklingError):
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached version of the StanModel.")
    return sm


# The STAN code
sho_code = """
functions {
  real[] sho( real t,
              real[] y,
              real[] theta,
              real[] x_r,
              int[] x_i) {

      real dydt[2];
      dydt[1] = y[2];
      dydt[2] = - theta[1] * y[1];
      return dydt;
    }
}
data {
  int<lower=1> nt;

  real y[nt,2];
  real t0;
  real t[nt];
}
transformed data {
  real x_r[0];
  int x_i[0];
}
parameters {
  real y0[2];
  vector<lower=0>[2] sigma;
  real theta[1];
}
model {
  real y_hat[nt,2];
  sigma ~ cauchy(0,1);
  theta ~ normal(0,1);
  y0 ~ normal(0,1);
  y_hat = integrate_ode_rk45(sho, y0, t0, t, theta, x_r, x_i);
  for (i in 1:nt)
  y[i] ~ normal(y_hat[i], sigma);
}
"""


def gen_data(y0=[0.5, 0.5], theta=1., sigma=0.05):
    """Generate synthetic SHO data."""
    # `y0_true` is our initial data vector: (position, velocity)
    y0_true = y0

    # `theta` is our (single) parameter vector: theta = k / m
    theta_true = theta

    # Gaussian noise standard deviation
    sigma_true = sigma

    # Let's solve the ODE stupidly with Euler steps
    dt = 0.001
    t_true = np.arange(0, 10, dt)
    N_true = len(t_true)
    y_true = np.zeros((N_true, 2))
    y_true[0] = y0_true
    for i, t in enumerate(t_true[:-1]):
        dydt = np.array([y_true[i][1], -theta_true * y_true[i][0]])
        y_true[i + 1] = y_true[i] + dydt * dt

    # Downbin and exclude first half: this is the data we will regress on
    t_obs = np.array(t_true[N_true // 2::500])
    y_obs = np.array(y_true[N_true // 2::500])
    N_obs = len(t_obs)

    # Add noise
    y_obs += sigma_true * np.random.randn(N_obs, 2)

    return t_true, y_true, t_obs, y_obs


# Get some synthetic SHO data
t_true, y_true, t_obs, y_obs = gen_data()

# The STAN data
sho_dat = {'nt': len(t_obs),
           'y0': [1, 1],
           't0': 0,
           't': t_obs,
           'y': y_obs,
           }

# Instantiate the Stan model and sample from the posterior
sm = StanModel(model_code=sho_code)
fit = sm.sampling(data=sho_dat, iter=2000, chains=1, warmup=500)

# Plot the STAN output
print(fit)
fit.plot()

# Corner plot
samples_dict = fit.extract(('y0', 'sigma', 'theta'))
samples = np.hstack([samples_dict['theta'].reshape(-1, 1),
                     samples_dict['sigma'][:, 0].reshape(-1, 1),
                     samples_dict['sigma'][:, 1].reshape(-1, 1),
                     samples_dict['y0'][:, 0].reshape(-1, 1),
                     samples_dict['y0'][:, 1].reshape(-1, 1)])
labels = [r'$k/m$', r'$\sigma_x$', r'$\sigma_v$', r'$x_0$', r'$v_0$']
corner(samples, labels=labels)

# Plot some posterior samples alongside the truth
fig, ax = pl.subplots(2)
ax[0].set_ylabel('Position')
ax[1].set_ylabel('Velocity')
ax[1].set_xlabel('Time')
for i, axis in enumerate(ax):
    axis.plot(t_obs, y_obs[:, i], 'k.')
    axis.plot(t_true, y_true[:, i], 'r-')
nsamp = len(samples_dict['y0'])
for k in range(50):
    n = np.random.randint(nsamp)
    theta = samples_dict['theta'][n]
    y0 = samples_dict['y0'][n]
    t, y, _, _ = gen_data(y0=y0, theta=theta)
    ax[0].plot(t, y[:, 0], color='k', alpha=0.1, lw=1)
    ax[1].plot(t, y[:, 1], color='k', alpha=0.1, lw=1)

# Show plots
pl.show()
