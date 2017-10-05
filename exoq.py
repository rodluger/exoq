import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as pl
import kplr
from kplr import KOI
KOI._id = '{kepid}'
client = kplr.API()

# Constants (mks)
G = 6.67428e-11
DAYSEC = 86400.
HRSEC = 3600.
KGM3 = 1.e3

def N(mu, sig1, sig2 = None, positive = True):
    '''
    
    '''
    
    # Loop until positive
    while True:
    
        if sig1 is None:
            return mu
        elif sig2 is None:
            res = mu + np.abs(sig1) * np.random.randn()
        else:
            res = mu + 0.5 * (np.abs(sig1) + np.abs(sig2)) * np.random.randn()
    
        if (positive and res > 0) or not positive:
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

def ImpactDistribution():
    '''
    Plot the impact parameter distribution
    
    '''
    
    kois = GetKOIs()
    impacts = [k.koi_impact for k in kois]
    pl.hist(impacts, bins = 30, histtype = 'step', range = (0,1.2))
    pl.show()

def DrawEccentricity(koi):
    '''
    
    '''
    
    # Loop until we get a physical impact parameter
    while True:
    
        # Measured duration in seconds
        T = HRSEC * N(koi.koi_duration, koi.koi_duration_err1, koi.koi_duration_err2)
    
        # Density in kg/m^3
        rho = KGM3 * N(koi.koi_srho, koi.koi_srho_err1, koi.koi_srho_err2)
    
        # Period in seconds
        per = DAYSEC * N(koi.koi_period, koi.koi_period_err1, koi.koi_period_err2)
    
        # Semi-major axis in units of stellar radius
        aRs = ((G * rho * per ** 2.) / (3. * np.pi)) ** (1. / 3.)
    
        # Rp/Rs
        rprs = N(koi.koi_ror, koi.koi_ror_err1, koi.koi_ror_err2)
    
        # Impact parameter
        b = N(koi.koi_impact, koi.koi_impact_err1, koi.koi_impact_err2)
    
        # Circular duration in seconds
        Tcirc = np.sqrt((1 + rprs) ** 2 - b ** 2) / (np.pi * aRs) * per
    
        # Duration anomaly
        d = T / Tcirc
        
        if not np.isnan(d):
            break
    
        # Loop until we get a physical eccentricity
    while True:
        
        # Draw the mean anomaly from a uniform distribution
        theta = np.random.random() * 2 * np.pi
        
        # Solve the quadratic
        A = 1 + d ** 2 * np.cos(theta) ** 2
        B = 2 * d ** 2 * np.cos(theta)
        C = d ** 2 - 1 
        foo = B ** 2 - 4 * A * C
        e1 = (B - np.sqrt(foo)) / (2 * A)
        e2 = (B + np.sqrt(foo)) / (2 * A)
    
        if (e1 >= 0 and e1 <= 1) and (e2 >= 0 and e2 <= 1):
            if np.random.random() < 0.5:
                return e1
            else:
                return e2
        elif (e1 >= 0 and e1 <= 1):
            return e1
        elif (e2 >= 0 and e2 <= 1):
            return e2
        else:
            continue
            
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
    
kois = GetKOIs()
e = [DrawEccentricity(koi) for koi in kois]
pl.hist(e, bins = 50)
pl.show()