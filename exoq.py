"""Tidal evolution of Kepler systems."""

import numpy as np
import matplotlib.pyplot as pl
from tqdm import tqdm
import kplr
from kplr import KOI
import emcee
import ctypes
import pickle
import corner

# C library
lib = ctypes.CDLL('tidal.so')
_Evolve = lib.Evolve
_Evolve.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                    ctypes.c_double, ctypes.c_double, ctypes.c_double,
                    ctypes.c_double,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double)]

# Constants (mks)
G = 6.67428e-11
DAYSEC = 86400.
AUM = 1.496e11
YEARSEC = DAYSEC * 365.
HRSEC = 3600.
KGM3 = 1.e3
MSUN = 1.988416e30
MJUP = 1.898e27
MNEP = 1.024e26
RSUN = 6.957e8
RJUP = 7.149e7


def N(mu, sig1, sig2=None, lo=0, hi=None):
    """Sample from a normal distribution."""
    # Check bounds
    if hi is None:
        hi = np.inf
    if lo is None:
        lo = -np.inf

    # Loop until in range
    while True:
        if sig1 is None:
            return mu
        elif sig2 is None:
            res = mu + np.abs(sig1) * np.random.randn()
        else:
            res = mu + 0.5 * (np.abs(sig1) + np.abs(sig2)) * np.random.randn()
        if (res >= lo) and (res <= hi):
            return res


def NormalPrior(koi, param, x, unit=1):
    """Return a normal prior for a parameter."""
    # Mean
    x0 = unit * getattr(koi, param)

    # +/- standard devatiation
    sig1 = getattr(koi, param + '_err1')
    sig2 = getattr(koi, param + '_err2')
    if sig1 is None:
        sig_x0 = 0.
    elif sig2 is None:
        sig_x0 = unit * np.abs(sig1)
    else:
        sig_x0 = unit * 0.5 * (np.abs(sig1) + np.abs(sig2))

    return -0.5 * (x - x0) ** 2 / sig_x0 ** 2


def GetKOIs():
    """Return a list of all KOIs with minor vetting."""
    # A little hack
    KOI._id = '{kepid}'
    client = kplr.API()

    # Get all the DR25 KOIs
    columns = ('kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition',
               'koi_period', 'koi_period_err1', 'koi_period_err2',
               'koi_impact', 'koi_impact_err1', 'koi_impact_err2',
               'koi_duration', 'koi_duration_err1', 'koi_duration_err2',
               'koi_ror', 'koi_ror_err1', 'koi_ror_err2',
               'koi_srho', 'koi_srho_err1', 'koi_srho_err2',
               'koi_score')
    params = {"select": ",".join(columns)}
    kois = [kplr.KOI(client, k) for k in
            client.ea_request("q1_q17_dr25_koi", **params)]

    # Get all the stars, and add their properties to
    # the KOIs
    columns = ('kepid', 'radius', 'radius_err1', 'radius_err2',
               'mass', 'mass_err1', 'mass_err2',
               'teff_prov')
    params = {"select": ",".join(columns)}
    all_stars = client.ea_request("q1_q17_dr25_stellar", **params)
    kepids = np.array([s['kepid'] for s in all_stars])
    for i, koi in enumerate(kois):
        koi.kepid
        ind = np.argmax(kepids == koi.kepid)
        for k, v in all_stars[ind].items():
            setattr(kois[i], 'star_' + k, v)

    # Apply some filters.
    good = []
    for i, koi in enumerate(kois):

        # False positive
        if koi.koi_pdisposition == 'FALSE POSITIVE':
            continue

        # No stellar data for this KOI
        elif koi.star_teff_prov == 'Solar':
            continue

        # Some important value is None
        elif koi.koi_impact is None:
            continue

        # Low score
        elif koi.koi_score < 0.9:
            continue

        # Bad impact parameter
        elif koi.koi_impact >= 1:
            continue

        # This target is OK
        else:
            good.append(i)

    # Make the cut
    print("Using %d of %d available KOIs." % (len(good), len(kois)))
    kois = [kois[i] for i in good]

    return kois


def TidalEvolve(M, m, R, r, tau, k2, age, a, e):
    """Tidally evolve a system forward in time."""
    a = np.array([a])
    e = np.array([e])
    _Evolve(M, m, R, r, tau, k2, age,
            np.ctypeslib.as_ctypes(a), np.ctypeslib.as_ctypes(e))

    return a[0], e[0]


def LnPrior(x, **kwargs):
    """Log prior."""
    # Parameters
    logtau, age, M, R, rR, cosi, ai, ei, w = x

    # Hard bounds
    if (logtau < -5) or (logtau > 5):
        return -np.inf
    elif (age < 1e9 * YEARSEC) or (age > 1e10 * YEARSEC):
        return -np.inf
    elif (M < 0):
        return -np.inf
    elif (R < 0):
        return -np.inf
    elif (rR < 0):
        return -np.inf
    elif (cosi < 0) or (cosi > 1):
        return -np.inf
    elif (ai < 0):
        return -np.inf
    if (ei < 0) or (ei > 1):
        return -np.inf
    elif (w < 0) or (w > 2 * np.pi):
        return -np.inf

    # TODO: Better eccentricity prior?

    return 0.


def LnLike(x, **kwargs):
    """Log likelihood."""
    # Initialize blobs
    blobs = []

    # Compute prior probability
    lnprior = LnPrior(x, **kwargs)
    if np.isinf(lnprior):
        return lnprior, blobs

    # Parameters
    logtau, age, M, R, rR, cosi, ai, ei, w = x
    r = rR * R
    koi = kwargs['koi']

    # NOTE: We set the planet mass to 1 MNEP and the k2 Love number to 0.3.
    # The variable `tau` is therefore an *effective* tidal time lag, `tau'`:
    #
    #            k2       MJ
    # tau' =    ----- * ----- * tau
    #            0.3      m
    #
    m = MNEP
    k2 = 0.3

    # Tidally evolve to get present-day eccentricity and period
    a, e = TidalEvolve(M, m, R, r, 10 ** logtau, k2, age, ai, ei)
    P = np.sqrt(4 * np.pi ** 2 * a ** 3 / (G * (M + m)))

    # Prevent star-crossing orbits
    if a * (1 - e ** 2) <= R:
        return -np.inf, blobs

    # Reject non-transiting orbits
    # Equation (7) in Winn (2010)
    b = (a * cosi / R) * (1 - e ** 2) / (1 + e * np.sin(w))
    if (b >= 1 + rR):
        return -np.inf, blobs

    # Compute the transit duration for this sample
    # Equation (14) * Equation (16) in Winn (2010)
    sini = np.sqrt(1 - cosi ** 2)
    arg = (R / a) * np.sqrt((1 + rR) ** 2 - b ** 2) / sini
    fac = np.sqrt(1 - e ** 2) / (1 + e * np.sin(w))
    T = (P / np.pi) * np.arcsin(arg) * fac

    # Update blobs
    blobs = []

    # Likelihood
    lnlike = lnprior + \
        + NormalPrior(koi, 'koi_duration', T / HRSEC) \
        + NormalPrior(koi, 'koi_period', P / DAYSEC)  \
        + NormalPrior(koi, 'star_mass', M / MSUN)     \
        + NormalPrior(koi, 'star_radius', R / RSUN)   \
        + NormalPrior(koi, 'koi_ror', rR)             \
        + NormalPrior(koi, 'koi_impact', b)

    return lnlike, blobs


def IndividualTauDistributionMCMC(koi, nwalk=100, nsteps=50000,
                                  nburn=10000, thin=10, **kwargs):
    """Compute the `tau` posterior for a given KOI."""
    # Kwargs for likelihood function
    ll_kwargs = dict(koi=koi)
    ndim = 9
    nblobs = 0

    # Get the initial state
    x0 = []
    blobs0 = []
    a = RSUN * koi.star_radius * \
        ((G * koi.koi_srho * 1.e3 * (koi.koi_period * DAYSEC) ** 2.) /
         (3. * np.pi)) ** (1. / 3.)
    M = MSUN * koi.star_mass
    R = RSUN * koi.star_radius
    RHO = M / ((4 / 3.) * np.pi * R ** 3)
    P = DAYSEC * koi.koi_period
    aRs = ((G * RHO * P ** 2.) / (3. * np.pi)) ** (1. / 3.)
    rR = koi.koi_ror
    b = koi.koi_impact
    Tcirc = np.sqrt((1 + rR) ** 2 - b ** 2) / (np.pi * aRs) * P
    tda = HRSEC * koi.koi_duration / Tcirc
    emin = np.abs((tda ** 2 - 1) / (tda ** 2 + 1))

    for k in range(nwalk):
        like = -np.inf
        while np.isinf(like):
            logtau = 2 * np.random.randn()
            age = (1 + 9 * np.random.random()) * 1e9 * YEARSEC
            M = MSUN * N(koi.star_mass, koi.star_mass_err1, koi.star_mass_err2)
            R = RSUN * N(koi.star_radius, koi.star_radius_err1,
                         koi.star_radius_err2)
            rR = N(koi.koi_ror, koi.koi_ror_err1, koi.koi_ror_err2)
            cosi = b / aRs * (1 + 0.1 * np.random.randn())
            ai = a * (1 + 0.01 * np.abs(np.random.randn()))
            ei = emin + np.abs(np.random.random()) * 0.1
            w = np.random.random() * 2 * np.pi
            x = [logtau, age, M, R, rR, cosi, ai, ei, w]
            like, blobs = LnLike(x, **ll_kwargs)
        x0.append(x)
        blobs0.append(blobs)

    # Plotting options
    labels = [r"$\log(\tau) [s]$",
              r"age [Gyr]",
              r"$M_\star$ [$M_\odot$]",
              r"$R_\star$ [$R_\odot$]",
              r"$r/R_\star$",
              r"$\cos(\i)$",
              r"$a_i$ [AU]",
              r"$e_i$",
              r"$\omega$ [deg]",
              ]

    units = [1,
             1. / (1e9 * YEARSEC),
             1. / MSUN,
             1. / RSUN,
             1.,
             1.,
             1. / AUM,
             1.,
             180. / np.pi,
             ]

    truths = [np.nan,
              np.nan,
              koi.star_mass,
              koi.star_radius,
              koi.koi_ror,
              b / aRs,
              a,
              emin,
              np.nan]

    ranges = [(-5, 5),
              (1, 10),
              1,
              1,
              1,
              (0, 1),
              1,
              (0, 1),
              (0, 360)]

    # Run MCMC
    sampler = emcee.EnsembleSampler(nwalk, ndim, LnLike,
                                    kwargs=ll_kwargs)
    for i in tqdm(sampler.sample(x0, iterations=nsteps, blobs0=blobs0,
                                 thin=thin), total=nsteps):
        pass

    # Collect the samples and blobs
    blobs = np.array(sampler.blobs).swapaxes(0, 1)
    samples = np.concatenate((sampler.chain, blobs), axis=2)

    # Do some unit conversions
    for i in range(ndim + nblobs):
        samples[:, :, i] *= units[i]

    # Plot the chains
    fig, ax = pl.subplots(int(np.ceil((ndim + nblobs) / 2)), 2,
                          figsize=(9, 7))
    ax[-1, 0].set_xlabel('Iteration')
    ax[-1, 1].set_xlabel('Iteration')
    ax = ax.flatten()
    for i in range(ndim + nblobs):
        for k in range(nwalk):
            ax[i].plot(samples[k, :, i], lw=1, alpha=0.3, zorder=-1)
        ax[i].set_ylabel(labels[i])
        ax[i].axvline(nburn // thin, color='r', lw=1, alpha=0.5)
        if i < ndim + nblobs - 1:
            ax[i].set_xticklabels([])
        ax[i].set_rasterization_zorder(0)
        ax[i].set_ylim(np.min(samples[:, nburn // thin:, i]),
                       np.max(samples[:, nburn // thin:, i]))

    # Plot the likelihood evolution if there's space
    if (ndim + nblobs) % 2:
        for k in range(nwalk):
            ax[-1].plot(sampler.lnprobability[k],
                        lw=1, alpha=0.3, zorder=-1)
        ax[-1].axvline(nburn // thin, color='r', lw=1, alpha=0.5)
        ax[-1].set_rasterization_zorder(0)
        ax[-1].set_ylim(np.min(sampler.lnprobability[:, nburn // thin:]),
                        np.max(sampler.lnprobability[:, nburn // thin:]))
        ax[-1].set_ylabel('ln(prob)')

    # Save
    fig.savefig('chains.pdf', bbox_inches='tight', dpi=800)

    # Plot the corner diagram w/ burn-in removed
    fig = corner.corner(
        samples[:, nburn // thin:, :].reshape(-1, ndim + nblobs),
        labels=labels, bins=30,
        truths=truths, range=ranges)

    # Save
    fig.savefig('corner.pdf', bbox_inches='tight')


if __name__ == '__main__':

    # Load
    try:
        kois = pickle.load(open("kois.pickle", "rb"))
    except FileNotFoundError:
        kois = GetKOIs()
        pickle.dump(kois, open("kois.pickle", "wb"))

    # Run first KOI
    IndividualTauDistributionMCMC(kois[0])
