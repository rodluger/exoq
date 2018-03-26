"""Stan utils."""

import pystan
import pickle
from hashlib import md5
import numpy as np


def normal(mu, sig):
    """Return a sample from a Gaussian."""
    return mu + sig * np.random.randn()


def StanModel(model_code, model_name=None, **kwargs):
    """
    Override `pystan.StanModel` to allow caching of the model.

    Based on
    https://pystan.readthedocs.io/en/latest/avoiding_recompilation.html
    """
    code_hash = md5(str(model_code).encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cache/cached-model-{}.sm'.format(code_hash)
    else:
        cache_fn = 'cache/cached-{}-{}.sm'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except (FileNotFoundError, pickle.UnpicklingError):
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached version of the StanModel.")
    return sm


def StanFit(model_code, data,
            niter=2000, chains=1, warmup=500,
            model_name=None,
            n_jobs=-1, init='random'):
    """
    Fit a STAN model to synthetic data and cache the results.

    Based on
    https://pystan.readthedocs.io/en/latest/avoiding_recompilation.html
    """
    code_hash = md5(str(data).encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cache/cached-model-{}.sf'.format(code_hash)
    else:
        cache_fn = 'cache/cached-{}-{}.sf'.format(model_name, code_hash)

    # Instantiate the Stan model and sample from the posterior
    sm = StanModel(model_code=model_code)

    # Is the fit cached?
    try:
        fit = pickle.load(open(cache_fn, 'rb'))
    except (FileNotFoundError, pickle.UnpicklingError, ImportError):
        fit = sm.sampling(data=data, iter=niter,
                          chains=chains, warmup=warmup,
                          n_jobs=n_jobs, init=init)
        with open(cache_fn, 'wb') as f:
            pickle.dump(fit, f)
    else:
        print("Using cached version of the fit.")
    return fit
