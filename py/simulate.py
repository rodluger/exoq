"""Simulate a population of Kepler planets."""
import numpy as np
import matplotlib.pyplot as pl
import tidal
from scipy.stats import beta
from tqdm import tqdm
from collections import OrderedDict
np.random.seed(1234)


def sample_e(size=1):
    """From Kipping (2013). See also Hogg, Myers and Bovy (2010)."""
    a = 0.867
    b = 3.03
    return beta.rvs(a, b, size=size)


def sample_a(size=1):
    """Sample from a flat prior for a, based on zero physics."""
    inner = 0.01
    outer = 0.3
    return inner + (outer - inner) * np.random.random(size)


def sample_r(size=1):
    """
    Sample from the planet radius distribution.

    Very loosely based on Fulton et al. (2017).
    """
    mu = 2.4
    sig = 0.7
    res = mu + sig * np.random.randn(size)
    res[res < 0.1] = 0.1
    return res


def sample_M(size=1):
    """Every star is a Sun, ish."""
    res = 1 + 0.1 * np.random.randn(size)
    res[res < 0.1] = 0.1
    return res


def sample_R(M, size=1):
    """Very simple mass-radius relation for stars."""
    res = M ** (3. / 7.) + 0.1 * np.random.randn(size)
    res[res < 0.1] = 0.1
    return res


def sample_m(r, size=1):
    """Made-up radius-mass relation for planets."""
    res = r ** (1 / 0.55) + 0.1 * np.random.randn(size)
    res[res < 0.1] = 0.1
    return res


def sample_logtau(r, size=1):
    """
    Sample the tidal time lag distribution.

    We assume

        logtau ~ N(alpha - beta * r, sigma)

    """
    beta = 0.75
    alpha = 3.13  # np.log10(638 * np.exp(beta))
    sigma = 0.1
    res = alpha - beta * r + sigma * np.random.randn()
    return res


def sample_age(size=1):
    """Flat age distribution (Gyr)."""
    youngest = 1
    oldest = 10
    return youngest + (oldest - youngest) * np.random.random(size)


def generate(N=10000, plot=True):
    """Generate N systems."""
    # Draw the eccentricities and semi-major axies
    e0 = sample_e(N)
    a0 = sample_a(N)
    r = sample_r(N)
    m = sample_m(r, N)
    logtau = sample_logtau(r, N)
    M = sample_M(N)
    R = sample_R(M, N)
    t = sample_age(N)

    # Tidally evolve
    a = np.zeros(N)
    e = np.zeros(N)
    for i in tqdm(range(N)):
        a[i], e[i] = tidal.evolve(M[i], m[i], R[i], r[i], 10 ** logtau[i],
                                  1.0, t[i] * 1e9, a0[i], e0[i])

    # Add noise
    siga = 0.01 * np.ones(N)
    sige = 0.01 * np.ones(N)
    sigr = 0.01 * np.ones(N)
    sigt = 0.1 * np.ones(N)
    sigm = 0.01 * np.ones(N)
    sigM = 0.1 * np.ones(N)
    sigR = 0.1 * np.ones(N)
    a += siga * np.random.randn(N)
    e += sige * np.random.randn(N)
    r += sigr * np.random.randn(N)
    t += sigt * np.random.randn(N)
    m += sigm * np.random.randn(N)
    M += sigM * np.random.randn(N)
    R += sigR * np.random.randn(N)

    # Plot
    if plot:
        fig, ax = pl.subplots(1, 2, figsize=(8, 4))
        bins = 30
        amin = min(np.min(a0), np.min(a))
        amax = min(np.max(a0), np.max(a))
        ax[0].hist(a0, histtype='step', color='C0',
                   bins=bins, range=(amin, amax))
        ax[0].hist(a0, histtype='stepfilled', alpha=0.5, color='C0',
                   bins=bins, range=(amin, amax))
        ax[0].hist(a, histtype='step', color='C1',
                   bins=bins, range=(amin, amax))
        ax[0].hist(a, histtype='stepfilled', alpha=0.5, color='C1',
                   bins=bins, range=(amin, amax))
        ax[1].hist(e0, histtype='step', color='C0',
                   bins=bins, range=(0, 1))
        ax[1].hist(e0, histtype='stepfilled', alpha=0.5, color='C0',
                   bins=bins, range=(0, 1))
        ax[1].hist(e, histtype='step', color='C1',
                   bins=bins, range=(0, 1))
        ax[1].hist(e, histtype='stepfilled', alpha=0.5, color='C1',
                   bins=bins, range=(0, 1))
        ax[0].set_xlabel('Semi-major axis [AU]')
        ax[1].set_xlabel('Eccentricity')
        pl.show()

    # Data dictionary
    data = OrderedDict()
    data['N'] = N
    data['dat_a'] = a
    data['dat_siga'] = siga
    data['dat_e'] = e
    data['dat_sige'] = sige
    data['dat_r'] = r
    data['dat_sigr'] = sigr
    data['dat_t'] = t
    data['dat_sigt'] = sigt
    data['dat_m'] = t
    data['dat_sigm'] = sigt
    data['dat_M'] = M
    data['dat_sigM'] = sigM
    data['dat_R'] = R
    data['dat_sigR'] = sigR
    data['K'] = 50
    data['solution_times'] = np.linspace(0, 11, 51)[1:]

    init = OrderedDict()
    init['par_alpha'] = 5.00
    init['par_beta'] = 1.00
    init['par_sigma'] = 0.1
    init['par_logtau'] = np.random.uniform(-1, 3, N)
    init['par_a0'] = np.random.uniform(0.01, 0.3, N)
    init['par_e0'] = np.random.uniform(0., 0.5, N)
    init['par_t'] = np.random.uniform(1, 10, N)
    init['par_r'] = 2.4 * np.random.randn(N) + 0.1
    init['par_M'] = np.random.uniform(0.9, 1.1, N)
    init['par_R'] = np.random.uniform(0.9, 1.1, N)

    return data, [init]


if __name__ == "__main__":
    generate()
