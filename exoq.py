import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as pl
from tqdm import tqdm
import kplr
from kplr import KOI
KOI._id = '{kepid}'
client = kplr.API()

# Constants (mks)
G = 6.67428e-11
DAYSEC = 86400.
HRSEC = 3600.
KGM3 = 1.e3
MSUN = 1.988416e30
RSUN = 6.957e8

def N(mu, sig1, sig2 = None, lo = 0, hi = None):
    '''
    
    '''
    
    return mu
    
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
    
def GetKOIs():
    '''
    
    '''
    
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

def DrawEccentricity(tda, eps = 0.001, maxruns = 9999):
    '''
    
    '''
    
    # Loop until we get a physical eccentricity
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

def EccentricityDistribution():
    '''
    
    '''
    
    print("Getting KOI data...")
    kois = GetKOIs()
    
    print("Computing the TDA...")
    tdas = [DrawTDA(koi) for koi in kois]
    
    print("Sampling the eccentricity distribution...")
    eccs = []
    for i in tqdm(range(len(kois))):
    
        ecc = DrawEccentricity(tdas[i])
        if not np.isnan(ecc):
            eccs.append(ecc)
    
    # Plot!
    print(len(eccs))
    pl.hist(eccs, bins = 30)
    pl.show()

EccentricityDistribution()