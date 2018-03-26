"""Download the KOIs."""

import numpy as np
import kplr
from kplr import KOI
import pickle


def GetKOIs():
    """Return a list of all KOIs with minor vetting."""
    try:
        kois = pickle.load(open("kois.pickle", "rb"))
    except (FileNotFoundError, pickle.UnpicklingError):

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

        pickle.dump(kois, open("kois.pickle", "wb"))

    return kois
