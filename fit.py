import numpy as _np
import emcee as _mc
import utils as _ci
_ndim = 2

def fit_powerlaw_dirty(e, elim):
    """
    Returns a max-likelihood power law fit of the form

    f = C

    where the data are drawn from distributions with the same power-law index
    but different lower limits.

    Parameters
    ----------
    e
    elim

    Returns
    -------
    a : power law index
    aerr : error on the above
    """
    N = len(e)
    a = N / (_np.sum(_np.log(e / elim)))
    aerr = N * a / (N - 1) / _np.sqrt(N - 2)
    return a, aerr




def loglike_powerlaw(params, e, elim, expt, n):
    a, logC = params
    if a < 0:
        return -_np.inf
    else:
        C = 10 ** logC
        a = a + 1  # supplying cumulative but want regular exponent
        lam = expt * C * elim ** (1 - a)
        return _np.sum(-lam / n + _np.log(lam) + _np.log((a - 1) / elim) - a * _np.log(e / elim))


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


def fit_powerlaw(e, elim, expt, n, nwalkers=50, nsteps=10000, return_sampler=False, a_prior=None, logC_prior=None,
                 a_init=None, logC_init=None):
    """
    Compute ML values and error for logC and a in the expression Cx**-a that describes the rate of events with values
    above x.

    Parameters
    ----------
    e : event values
    elim : detection threshold for the dataset from which event was detected
    expt : total time of observation in which event was detected
    n : number of detections in the dataset from which event was detected

    Returns
    -------
    a, logC: values and negative and positive error bars given as triplet lists, logC has units log10(d-1)
    """

    a_prior, logC_prior = map(_prior_boilerplate, (a_prior, logC_prior))

    def loglike(params):
        return a_prior(params[0]) + logC_prior(params[1]) + loglike_powerlaw(params, e, elim, expt, n)

    if a_init is None:
        a_init = fit_powerlaw_dirty(e, elim)[0]
    if logC_init is None:
        logC_init = _get_logC_init(a_init, e, elim, expt)

    pos = [[a_init, logC_init] * _np.random.normal(1, 1e-3, size=_ndim) for _ in range(nwalkers)]
    sampler = _mc.EnsembleSampler(nwalkers, _ndim, loglike)
    sampler.run_mcmc(pos, nsteps)

    a, logC = parse_a_logC_from_sampler(sampler)
    a_triplet, logC_triplet = map(get_triplet, (a, logC))

    if return_sampler:
        return a_triplet, logC_triplet, sampler
    else:
        return a_triplet, logC_triplet


def parse_a_logC_from_sampler(sampler, burnin=50):
    return [sampler.chain[:, burnin:, i].ravel() for i in range(_ndim)]


def time_averaged_flare_power(sampler, emin, emax):
    a, logC = parse_a_logC_from_sampler(sampler)
    C = 10**logC
    average_power = a * C / (1 - a) * (1e4 ** (1 - a) - 10. ** (1 - a))
    return average_power


def constrain_flare_energy_budget(e, elim, expt, n, emin, emax, **fit_kws):
    _, _, sampler = fit_powerlaw_dirty(e, elim, expt, n, **fit_kws)

