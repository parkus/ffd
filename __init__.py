import numpy as _np
import emcee as _mc


def fit_powerlaw_dirty(e, emin):
    """
    Returns a max-likelihood power law fit of the form

    f = C

    where the data are drawn from distributions with the same power-law index
    but different lower limits.

    Parameters
    ----------
    e
    emin

    Returns
    -------
    a : power law index
    aerr : error on the above
    """
    N = len(e)
    a = N / (_np.sum(_np.log(e / emin)))
    aerr = N * a / (N - 1) / _np.sqrt(N - 2)
    return a, aerr


def get_triplet(samples):
    bounds = _np.percentile(samples, [16, 50, 84])
    negerr = bounds[1] - bounds[0]
    poserr = bounds[2] - bounds[1]
    return [bounds[1], negerr, poserr]


def loglike_powerlaw(params, e, emin, expt, N):
    a, logC = params
    if a < 0:
        return -_np.inf
    else:
        C = 10 ** logC
        a = a + 1  # supplying cumulative but want regular exponent
        lam = expt * C * emin ** (1 - a)
        return _np.sum(-lam / N + _np.log(lam) + _np.log((a - 1) / emin) - a * _np.log(e / emin))


def _get_logC_init(a_init, x, xmin, expt):
    return _np.log10(len(x) / _np.mean(expt) * _np.mean(xmin) ** a_init)


def _loglike_from_interval(x, interval):
    if x < interval[0] or x > interval[-1]:
        return -_np.inf
    else:
        return 1.0

def _prior_boilerplate(prior):
    if prior is None:
        return 0.0
    elif type(prior) is not function:
        try:
            return _loglike_from_interval(prior)
        except TypeError:
            raise ValueError('a_prior must either be a function or a list/tuple/array. See docstring.')
    else:
        return prior


def fit_powerlaw(e, emin, expt, n, nwalkers=50, nsteps=10000, return_sampler=False, a_prior=None, logC_prior=None,
                 a_init=None, logC_init=None):
    """
    Compute ML values and error for logC and a in the expression Cx**-a that describes the rate of events with values
    above x.

    Parameters
    ----------
    e : event values
    emin : detection threshold for the dataset from which event was detected
    expt : total time of observation in which event was detected
    n : number of detections in the dataset from which event was detected

    Returns
    -------
    a, logC: values and negative and positive error bars given as triplet lists, logC has units log10(d-1)
    """

    a_prior, logC_prior = map(_prior_boilerplate, (a_prior, logC_prior))

    def loglike(params):
        return a_prior(params[0]) + logC_prior(params[1]) + loglike_powerlaw(params, e, emin, expt, n)

    if a_init is None:
        a_init = fit_powerlaw_dirty(e, emin)[0]
    if logC_init is None:
        logC_init = _get_logC_init(a_init, e, emin, expt)

    ndim = 2
    pos = [[a_init, logC_init] * _np.random.normal(1, 1e-3, size=ndim) for _ in range(nwalkers)]
    sampler = _mc.EnsembleSampler(nwalkers, ndim, loglike)
    sampler.run_mcmc(pos, nsteps)

    samples = [sampler.chain[:, 50:, i].ravel() for i in range(ndim)]
    a = get_triplet(samples[0])
    logC = get_triplet(samples[1])

    if return_sampler:
        return a, logC, sampler
    else:
        return a, logC


def time_averaged_flare_power(a, logC, emin, emax):




def constrain_flare_energy_budget():
    pass
