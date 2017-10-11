import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as pl
from tqdm import tqdm
import kplr
from kplr import KOI
import emcee
import ctypes
import pickle
import corner
from numpy.ctypeslib import ndpointer, as_ctypes

# C library
lib = ctypes.CDLL('tidal.so')
_Evolve = lib.Evolve
_Evolve.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                    ctypes.c_double, ctypes.c_double, ctypes.c_double,
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

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

def N(mu, sig1, sig2 = None, lo = 0, hi = None):
    '''
    
    '''
    
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
 
def NormalPrior(koi, param, x, unit = 1):
    '''
    
    '''
    
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
    '''
    
    '''
    
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
    params={"select": ",".join(columns)}
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

def MinimumEccentricity(koi):
    '''
    
    '''
    
    # Measured duration in hours
    T = koi.koi_duration
    
    # Density in kg/m^3
    rho = koi.koi_srho * 1.e3
    
    # Semi-major axis in units of stellar radius
    aRs = ((G * rho * (koi.koi_period * DAYSEC) ** 2.) 
            / (3. * np.pi)) ** (1. / 3.)
    
    # Circular duration in hours
    Tcirc = np.sqrt((1 + koi.koi_ror) ** 2 - koi.koi_impact ** 2) \
            / (np.pi * aRs) * koi.koi_period * 24.
    
    # Minimum eccentricity
    d = T / Tcirc
    emin = np.abs((d ** 2 - 1) / (d ** 2 + 1))
    
    return emin
    
def ImpactDistribution():
    '''
    Plot the impact parameter distribution
    
    '''
    
    kois = GetKOIs()
    impacts = [k.koi_impact for k in kois]
    pl.hist(impacts, bins = 30, histtype = 'step', range = (0,1.2))
    pl.show()

def DrawTDA(koi):
    '''
    
    '''
    
    # Measured duration in seconds
    T = HRSEC * N(koi.koi_duration, koi.koi_duration_err1, koi.koi_duration_err2)

    # Density in kg/m^3
    mass = MSUN * N(koi.star_mass, koi.star_mass_err1, koi.star_mass_err2)
    radius = RSUN * N(koi.star_radius, koi.star_radius_err1, koi.star_radius_err2)
    rhos = mass / ((4 / 3.) * np.pi * radius ** 3)

    # Period in seconds
    per = DAYSEC * N(koi.koi_period, koi.koi_period_err1, koi.koi_period_err2)

    # Semi-major axis in units of stellar radius
    aRs = ((G * rhos * per ** 2.) / (3. * np.pi)) ** (1. / 3.)

    # Rp/Rs
    rprs = N(koi.koi_ror, koi.koi_ror_err1, koi.koi_ror_err2)

    # Impact parameter
    b = N(koi.koi_impact, koi.koi_impact_err1, koi.koi_impact_err2, hi = (1 + rprs))

    # Circular duration in seconds
    Tcirc = np.sqrt((1 + rprs) ** 2 - b ** 2) / (np.pi * aRs) * per

    # Duration anomaly
    TDA = T / Tcirc
    
    return TDA

def DrawEccentricityABC(tda, eps = 0.001, maxruns = 9999):
    '''
    
    '''
    
    # Rejection ABC
    diff = np.inf
    runs = 0
    while diff > eps:
        e = np.random.random()
        theta = np.random.random() * 2 * np.pi
        tda_ = np.sqrt(1 - e ** 2) / (1 + e * np.cos(theta))
        diff = np.abs(tda - tda_)
        runs += 1
        if runs > maxruns:
            return np.nan
    return e

def PopulationEccentricityDistributionABC():
    '''
    
    '''
    
    print("Getting KOI data...")
    kois = GetKOIs()
    
    print("Computing the TDA...")
    tdas = [DrawTDA(koi) for koi in kois]
    
    print("Sampling the eccentricity distribution...")
    eccs = []
    for i in tqdm(range(len(kois))):
    
        ecc = DrawEccentricityABC(tdas[i])
        if not np.isnan(ecc):
            eccs.append(ecc)
    
    # Plot!
    print(len(eccs))
    pl.hist(eccs, bins = 30)
    pl.show()

def IndividualEccentricityDistributionABC(tda, sig_tda, n = 1000, **kwargs):
    '''
    
    '''
    
    samples = []
    for i in tqdm(range(n)):
        x = DrawEccentricityABC(tda + sig_tda * np.random.randn(), **kwargs)
        if not np.isnan(x):
            samples.append(x)
    fig = pl.figure()
    pl.hist(samples, bins = 50, histtype = 'step', color = 'k')
    emin = np.abs((tda ** 2 - 1) / (tda ** 2 + 1))
    pl.axvline(emin, color = 'r', ls = '--')    
    pl.title('ABC')

def LnLikeMCMC(x, **kwargs):
    '''
    
    '''
    
    # Parameters
    e, w = x
    tda = kwargs['tda']
    sig_tda = kwargs['sig_tda']
    
    # Hard bounds
    if (e < 0) or (e > 1):
        return -np.inf
    if (w < 0) or (w > 2 * np.pi):
        return -np.inf
    
    # Transit duration anomaly
    tda_ = np.sqrt(1 - e ** 2) / (1 + e * np.cos(w))
    
    # Likelihood
    return -0.5 * (tda - tda_) ** 2 / sig_tda ** 2

def IndividualEccentricityDistributionMCMC(tda, sig_tda, nwalk = 10, ndim = 2, 
                                           nsteps = 100000, nburn = 10000, 
                                           **kwargs):
    '''
    
    '''
        
    # Initial state
    emin = np.abs((tda ** 2 - 1) / (tda ** 2 + 1))
    x0 = [[emin + np.random.random() * (1 - emin), 
           np.random.random() * 2 * np.pi] for n in range(nwalk)]
        
    # Run MCMC
    sampler = emcee.EnsembleSampler(nwalk, ndim, LnLikeMCMC, 
                                    kwargs = dict(tda = tda, 
                                                  sig_tda = sig_tda)
                                    )
    for i in tqdm(sampler.sample(x0, iterations = nsteps), total = nsteps):
        pass
  
    samples = sampler.chain[:,nburn:,0].reshape(-1)  
    fig = pl.figure()
    pl.hist(samples, bins = 50, histtype = 'step', color = 'k')
    pl.axvline(emin, color = 'r', ls = '--')
    pl.title('MCMC')

def TidalEvolve(M, m, R, r, tau, k2, age, a, e):
    '''
    
    '''
    
    a = np.array([a])
    e = np.array([e])
    _Evolve(M, m, R, r, tau, k2, age, np.ctypeslib.as_ctypes(a), np.ctypeslib.as_ctypes(e))
    
    return a[0], e[0]

def LnPriorTidal(x, **kwargs):
    '''
    
    '''
    
    # Parameters
    logtau, age, M, R, rR, b, ai, ei, w = x
    koi = kwargs['koi']
    
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
    elif (b < 0):
        return -np.inf
    elif (ai < 0):
        return -np.inf
    if (ei < 0) or (ei > 1):
        return -np.inf
    elif (w < 0) or (w > 2 * np.pi):
        return -np.inf
     
    # Reject non-transiting configurations
    # TODO: Do we need to account for the 
    # transit probability here?
    if b ** 2 >= (1 + rR) ** 2:
        return -np.inf
        
    # TODO: Better eccentricity prior?
    
    return 0.

def LnLikeTidal(x, **kwargs):
    '''
    
    '''
    
    # Initialize blobs
    blobs = []
    
    # Compute prior probability
    lnprior = LnPriorTidal(x, **kwargs)
    if np.isinf(lnprior):
        return lnprior, blobs
    
    # Parameters
    logtau, age, M, R, rR, b, ai, ei, w = x
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
        
    # Compute the transit duration for this sample
    vc = 2 * np.pi * a / P
    vsky = vc * (1 + e * np.cos(w)) / np.sqrt(1 - e ** 2)
    T = 2 * np.sqrt((R + r) ** 2 - (b * R) ** 2) / vsky
    
    # Update blobs
    blobs = []
    
    # Likelihood
    lnlike = NormalPrior(koi, 'koi_duration', T / HRSEC) \
           + NormalPrior(koi, 'koi_period', P / DAYSEC)  \
           + NormalPrior(koi, 'star_mass', M / MSUN)     \
           + NormalPrior(koi, 'star_radius', R / RSUN)   \
           + NormalPrior(koi, 'koi_ror', rR)             \
           + NormalPrior(koi, 'koi_impact', b)

    return lnprior + lnlike, blobs

def IndividualTauDistributionMCMC(koi, nwalk = 50, nsteps = 50000, nburn = 10000, 
                                  thin = 10, **kwargs):
    '''
    
    '''
    
    # Kwargs for likelihood function
    ll_kwargs = dict(koi = koi)
    ndim = 10
    nblobs = 0
    
    # Get the initial state
    x0 = []
    blobs0 = []
    a = RSUN * koi.star_radius * ((G * koi.koi_srho * 1.e3 * 
               (koi.koi_period * DAYSEC) ** 2.) / (3. * np.pi)) ** (1. / 3.)
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
            logtau = 0.25 * np.random.randn()
            age = (1 + 9 * np.random.random()) * 1e9 * YEARSEC
            M = MSUN * N(koi.star_mass, koi.star_mass_err1, koi.star_mass_err2)
            R = RSUN * N(koi.star_radius, koi.star_radius_err1, koi.star_radius_err2)
            rR = N(koi.koi_ror, koi.koi_ror_err1, koi.koi_ror_err2)
            b = N(koi.koi_impact, koi.koi_impact_err1, koi.koi_impact_err2)
            ai = a * (1 + 0.01 * np.abs(np.random.randn()))
            ei = emin + np.random.random() * 0.01
            w = np.random.random() * 2 * np.pi 
            x = [logtau, age, M, R, rR, b, rho, ai, ei, w]
            like, blobs = LnLikeTidal(x, **ll_kwargs)
        x0.append(x)
        blobs0.append(blobs)

    # Plotting options
    labels = [r"$\log(\tau) [s]$", 
              r"age [Gyr]",
              r"$M_\star$ [$M_\odot$]",
              r"$R_\star$ [$R_\odot$]",
              r"$r/R_\star$",
              r"$b$",
              r"$a_i$ [AU]", 
              r"$e_i$",
              r"$\omega$ [deg]",
              ]
              
    units = [1, 
             1. / (1e9 * YEARSEC), 
             1. / MSUN,
             1. / RSUN,
             1.,
             1. / 1000.,
             1. / AUM, 
             1., 
             180. / np.pi,
             ]
    
    truths = [np.nan,
              np.nan,
              koi.star_mass,
              koi.star_radius,
              koi.koi_ror,
              koi.koi_impact,
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
             (0,1),
             (0,360)]
    
    # Run MCMC
    sampler = emcee.EnsembleSampler(nwalk, ndim, LnLikeTidal, 
                                    kwargs = ll_kwargs)
    for i in tqdm(sampler.sample(x0, iterations = nsteps, blobs0 = blobs0, 
                                 thin = thin), total = nsteps):
        pass
    
    # Collect the samples and blobs
    blobs = np.array(sampler.blobs).swapaxes(0,1)
    samples = np.concatenate((sampler.chain, blobs), axis = 2)
    
    # Do some unit conversions    
    for i in range(ndim + nblobs):
        samples[:,:,i] *= units[i]
    
    # Plot the chains
    fig, ax = pl.subplots(int(np.ceil((ndim + nblobs) / 2)), 2, figsize = (9, 7))
    ax[-1,0].set_xlabel('Iteration')
    ax[-1,1].set_xlabel('Iteration')
    ax = ax.flatten()
    for i in range(ndim + nblobs):
        for k in range(nwalk):
            ax[i].plot(samples[k,:,i], lw = 1, alpha = 0.3, zorder = -1)
        ax[i].set_ylabel(labels[i])
        ax[i].axvline(nburn // thin, color = 'r', lw = 1, alpha = 0.5)
        if i < ndim + nblobs - 1:
            ax[i].set_xticklabels([])
        ax[i].set_rasterization_zorder(0)
        ax[i].set_ylim(np.min(samples[:,nburn // thin:,i]), np.max(samples[:,nburn // thin:,i]))
    fig.savefig('chains.pdf', bbox_inches = 'tight', dpi = 800)
    
    # Plot the corner diagram w/ burn-in removed
    fig = corner.corner(samples[:,nburn // thin:,:].reshape(-1, ndim + nblobs), 
                        labels = labels, bins = 30, 
                        truths = truths, range = ranges)
    fig.savefig('corner.pdf', bbox_inches = 'tight')

# Load
try:
    kois = pickle.load(open("kois.pickle", "rb"))
except:
    kois = GetKOIs()
    pickle.dump(kois, open("kois.pickle", "wb"))

import pdb; pdb.set_trace()

IndividualTauDistributionMCMC(kois[0])