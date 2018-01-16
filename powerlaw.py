from matplotlib import pyplot as plt
import numpy as np
import emcee
from scipy.optimize import minimize
from scipy.special import gammaln

def _loglike_from_interval(x, interval):
    if x < interval[0] or x > interval[-1]:
        return -np.inf
    else:
        return 0.0


def _prior_boilerplate(prior):
    if prior is None:
        return lambda x: 0.0
    elif not hasattr(prior, '__call__'):
        try:
            return lambda x: _loglike_from_interval(x, prior)
        except TypeError:
            raise ValueError('a_prior must either be a function or a list/tuple/array. See docstring.')
    else:
        return prior


class PowerLawFit(object):

    def __init__(self, flares, a_prior=None, logC_prior=None, nwalkers=10, nsteps=1000):
        """
        Create a PowerLawMCMC object. Intended to be created by a call to Flares.mcmc_powerlaw.
        """

        self.flares = flares
        self.a_prior = _prior_boilerplate(a_prior)
        self.logC_prior = _prior_boilerplate(logC_prior)

        # now make a function to rapidly compute the log likelihood.

        # collect info from datasets into arrays for fast power law loglike computations

        # first handle the power law portion of the likelihood
        expt, elim, n, e = [], [], [], []
        for dataset in self.flares.datasets:
            _n = dataset.n
            expt.extend([dataset.expt] * _n)
            elim.extend([dataset.elim] * _n)
            n.extend([_n] * _n)
            e.extend(dataset.e.tolist() if dataset.n > 0 else [])
        Fexpt, Felim, Fn, Fe = map(np.array, [expt, elim, n, e])
        def loglike_powerlaw(a_diff):
            return np.sum(np.log((a_diff - 1) / Felim) - a_diff * np.log(Fe / Felim))

        # next handle the (independent) poisson portion of the likelihood
        n = [dataset.n for dataset in self.flares.datasets]
        elim = [dataset.elim for dataset in self.flares.datasets]
        expt = [dataset.expt for dataset in self.flares.datasets]
        Dexpt, Delim, Dn = map(np.array, [expt, elim, n])
        def loglike_poisson(a_cum, C):
            lam = Dexpt * C * Delim ** -a_cum
            return np.sum(-lam + Dn * np.log(lam) - gammaln(Dn + 1))

        # all together now!
        def loglike(params):
            a_cum, C = params

            # always enforce a this general prior
            if a_cum <= 0:
                return -np.inf
            if C <= 0:
                return -np.inf
            return (loglike_powerlaw(a_cum + 1) + loglike_poisson(a_cum, C) + self.a_prior(a_cum)
                    + self.logC_prior(np.log10(C)))

        self.loglike = loglike

        def neglike(params):
            return -loglike(params)

        a_guess = self._quick_n_dirty_index()
        if not 0 < a_guess < 2:
            a_guess = 1.0
        elim_mean = np.mean([data.elim for data in self.flares.datasets])
        def guess_C(a_guess):
            return self.flares.n_total / self.flares.expt_total * elim_mean ** a_guess
        C_guess = guess_C(a_guess)

        result = minimize(neglike, [a_guess, C_guess], method='bfgs')
        self.ml_result = result
        if not result.success or not np.all(np.isfinite(result.x)):
            self.ml_success = False
            self.a_ml = None
            self.C_ml = None
            a_init = 1.0
        else:
            self.ml_success = True
            a, C = result.x
            self.a_ml = a
            self.C_ml = C
            a_init = a

        pos = []
        for _ in range(nwalkers):
            a = a_init + np.random.normal(0, 0.1)
            C = 10**(np.log10(C_guess(a_init)) + np.random.normal(0, 0.1))
            pos.append([a, C])
        sampler = emcee.EnsembleSampler(nwalkers, 2, loglike)
        sampler.run_mcmc(pos, nsteps)
        self.MCMCsampler = sampler

    @property
    def a(self):
        return self.MCMCsampler.flatchain[:,0]

    @property
    def C(self):
        return self.MCMCsampler.flatchain[:,1]


    def _quick_n_dirty_index(self):
        """
        Returns a quick-and-dirty max-likelihood power law fit of the form

        f ~ e**-a

        where f is the cumulative frequency of flares with energies greater than e.

        Returns
        -------
        a, aerr: floats
            max likelihood power-law index and error
        """
        e = np.concatenate([data.e for data in self.flares.datasets])
        elim = np.concatenate([[data.elim]*data.n for data in self.flares.datasets])
        N = self.flares.n_total
        a = N / (np.sum(np.log(e / elim)))
        return a

    def plotfit(self, ax=None, line_kws=None, step_kws=None):
        if self.ml_success == False:
            raise ValueError('No best fit to plot for this Fit object.')
        if ax is None:
            ax = plt.gca()
        self.flares.plot_ffd(ax=ax, **step_kws)
        emin = min(*[d.elim for d in self.flares.datasets])
        emax = max(*[np.max(d.e) for d in self.flares.datasets])
        plot(self.a_ml, self.C_ml, emin, emax, **line_kws)


def cumulative_frequency(a, C, e):
    """
    Cumulative frequency of events with energies (or other metric) greater than e for a power-law of the form

    f = C*e**-a

    to the flare energy distribution, where f is the cumulative frequency of flares with energies greater than e.
    """
    return C*e**-a

def differential_frequency(a, C, e):
    """
    Differential frequency of events with energies (or other metric) greater than e for a power-law of the form

    f = C*e**-a

    to the flare energy distribution, where f is the cumulative frequency of flares with energies greater than e.
    """
    return a*C*e**(-a-1)

def time_average(a, C, emin, emax):
    """
    Average output of events with energies in the range [emin, emax] for a power-law of the form

    f = C*e**-a

    to the flare energy distribution, where f is the cumulative frequency of flares with energies greater than e.

    If the power law is for flare equivalent durations, this amounts to the ratio of energy output in flares versus
    quiesence. If the power law is for flare energies, it is the time-averaged energy output of flares (units of power).
    """
    return a * C / (1 - a) * (emax ** (1 - a) - emin ** (1 - a))


def energy(a, C, f):
    """
    Minimum energy (or other metric given as e below) of events that occur with a frequency of f for a cumulative
    frequency distribution of the form

    f = C*e**-a.
    """
    return (f/C)**(1/-a)


def plot(a, C, emin, emax, *args, **kwargs):
    """
    Plot a power-law line of the form f = C*e**-a between emin and emax.

    Parameters
    ----------
    a : float
    C : float
    emin : float
    emax : float
    args :
        passed to matplotlib plot function
    kwargs :
        passed to matplotlib plot function

    Returns
    -------
    line : matplotlib line object
    """
    ax = kwargs.get('ax', plt.gca())
    fmin, fmax = [cumulative_frequency(a, C, e) for e in  [emin, emax]]
    return ax.plot([emin, emax], [fmin, fmax], *args, **kwargs)