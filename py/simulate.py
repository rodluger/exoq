"""Simulate a population of Kepler planets."""
import numpy as np
import matplotlib.pyplot as pl
import tidal
from scipy.stats import beta
from tqdm import tqdm


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
    """A made-up radius-mass relation for planets."""
    res = r ** (1 / 0.55) + 0.1 * np.random.randn(size)
    res[res < 0.1] = 0.1
    return res


def sample_tau(r, size=1):
    """
    Sample the tidal time lag distribution.

    We assume

        tau ~ N(alpha / (1 + beta * r), sigma)

    """
    beta = 0.75
    alpha = 1350.  # 638 * np.exp(beta)
    sigma = 50.
    res = alpha * np.exp(-beta * r) + sigma * np.random.randn(size)
    res[res < 1e-5] = 1e-5
    return res


def sample_age(size=1):
    """Flat age distribution."""
    youngest = 1e9
    oldest = 10e9
    return youngest + (oldest - youngest) * np.random.random(size)


def generate(N=10000):
    """Generate N systems."""
    # Draw the eccentricities and semi-major axies
    e0 = sample_e(N)
    a0 = sample_a(N)
    r = sample_r(N)
    m = sample_m(r, N)
    tau = sample_tau(r, N)
    M = sample_M(N)
    R = sample_R(M, N)
    age = sample_age(N)

    # Simple
    k2 = np.ones(N)

    # Tidally evolve
    a = np.zeros(N)
    e = np.zeros(N)
    for i in tqdm(range(N)):
        a[i], e[i] = tidal.evolve(M[i], m[i], R[i], r[i], tau[i],
                                  k2[i], age[i], a0[i], e0[i])

    # Plot
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


generate()
