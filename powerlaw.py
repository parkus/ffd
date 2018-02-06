from matplotlib import pyplot as plt
import numpy as np
import emcee
from scipy.optimize import minimize
from scipy.special import gammaln
from warnings import warn

def loglike_from_interval(x, interval):
    if x < interval[0] or x > interval[-1]:
        return -np.inf
    else:
        return 0.0


def loglike_from_hist(bins, counts):
    integral = np.sum(np.diff(bins)*counts)
    norm = counts/integral
    def loglike(x):
        if x <= bins[0]:
            return -np.inf
        if x >= bins[-1]:
            return -np.inf
        i = np.searchsorted(bins, x)
        return np.log(norm[i - 1])
    return loglike


def _prior_boilerplate(prior):
    if prior is None:
        return lambda x: 0.0
    elif not hasattr(prior, '__call__'):
        try:
            return lambda x: loglike_from_interval(x, prior)
        except TypeError:
            raise ValueError('a_prior must either be a function or a list/tuple/array. See docstring.')
    else:
        return prior


class PowerLawFit(object):

    def __init__(self, flare_dataset, a_prior=None, logC_prior=None, nwalkers=10, nsteps=1000):
        """
        Create a PowerLawMCMC object. Intended to be created by a call to Flares.mcmc_powerlaw.
        """
        if flare_dataset.n_total < 3 and a_prior is None:
            raise ValueError('At least 3 flares required to attempt a fit unless you place a prior on a. '
                             'Several more than 3 will likely be required for said fit to converge.')
        # if flare_dataset.n_total == 0:
        #     raise ValueError('No flares in the provided FlareDataset.')

        self.flare_dataset = flare_dataset
        self.a_prior = _prior_boilerplate(a_prior)
        self.logC_prior = _prior_boilerplate(logC_prior)
        self.n = flare_dataset.n_total
        if self.n == 0:
            warn('No flares in the flare dataset. Be cautious: for MCMC samples of the posterior of C and similar '
                 'derived quantities (such as flare rate), use only the linear values, not logC, and use them only to '
                 'constrain upper limits.')

        # now make a function to rapidly compute the log likelihood.

        # collect info from datasets into arrays for fast power law loglike computations

        # first handle the power law portion of the likelihood
        expt, elim, n, e = [], [], [], []
        for obs in self.flare_dataset.observations:
            _n = obs.n
            expt.extend([obs.expt] * _n)
            elim.extend([obs.elim] * _n)
            n.extend([_n] * _n)
            e.extend(obs.e.tolist() if obs.n > 0 else [])
        Fexpt, Felim, Fn, Fe = map(np.array, [expt, elim, n, e])
        def loglike_powerlaw(a_diff):
            return np.sum(np.log((a_diff - 1) / Felim) - a_diff * np.log(Fe / Felim))

        # next handle the (independent) poisson portion of the likelihood
        n = [obs.n for obs in self.flare_dataset.observations]
        elim = [obs.elim for obs in self.flare_dataset.observations]
        expt = [obs.expt for obs in self.flare_dataset.observations]
        Dexpt, Delim, Dn = map(np.array, [expt, elim, n])
        def loglike_poisson(a_cum, logC):
            # keeping lambda in log space avoids log(lam = 0) resulting from some MCMC samples where very low logC
            # values are tried such that 10**logC = 0 to computer precision
            loglam = np.log(Dexpt) + logC*np.log(10.) - a_cum*np.log(Delim)
            return np.sum(-10**loglam + Dn * loglam - gammaln(Dn + 1))

        # all together now!
        # keep C in log space. Otherwise, it seems the MCMC can't handle the narrow, highly curved distribution.
        def loglike(params):
            """
            Log likelihood of the posterior distribution for values a and logC for a power-law fit to the cumulative
            flare frequency distribution,

            (flare rate) = 10**logC * (flare energy)**-a.

            Parameters
            ----------
            params : [a, logC]
                values of the power-law parameters defined above. logC is log(C) (i.e. base-10 logarithm)

            Returns
            -------
            loglike : scalar
                ln(likelihood) of the model given the flare data
            """
            a_cum, logC = params

            # always enforce this general prior on a
            if a_cum <= 0:
                return -np.inf
            return (loglike_powerlaw(a_cum + 1) + loglike_poisson(a_cum, logC) + self.a_prior(a_cum)
                    + self.logC_prior(logC))

        self.loglike = loglike

        elim_mean = np.mean([data.elim for data in self.flare_dataset.observations])
        def guess_logC(a_guess):
            n = self.flare_dataset.n_total
            if n == 0:
                n = 1
            return np.log10(n / self.flare_dataset.expt_total * elim_mean ** a_guess)

        # the MCMC sampler seems to struggle to sample the posterior of the rate constant C well when there are no
        # flares and it is specified as logC (log10(C)). I think this is because it is basically unconstrained to -inf.
        # However, when there is a flare logC tends to be much more normally distributed than C and if C is used the
        # MCMC sampler does wacky things. So I'll treat the two separately.

        if flare_dataset.n_total > 0:

            # try to find max likelihood values
            def neglike(params):
                return -loglike(params)
            a_guess = self._quick_n_dirty_index()
            if not 0 < a_guess < 2:
                a_guess = 1.0
            logC_guess = guess_logC(a_guess)
            result = minimize(neglike, [a_guess, logC_guess], method='Nelder-Mead')
            self.ml_result = result
            if not result.success or not np.all(np.isfinite(result.x)):
                self.ml_success = False
                self.a_ml = None
                self.logC_ml = None
                a_init = 1.0
            else:
                self.ml_success = True
                a, logC = result.x
                self.a_ml = a
                self.logC_ml = logC
                a_init = a

            pos = []
            for _ in range(nwalkers):
                a = a_init + np.random.normal(0, 0.01)
                logC = guess_logC(a_init) + np.random.normal(0, 0.2)
                pos.append([a, logC])

            sampler = emcee.EnsembleSampler(nwalkers, 2, loglike)

        else: # no flares in dataset
            self.ml_success = False
            self.a_ml = None
            self.logC_ml = None
            a_init = 1.0
            pos = []

            def loglike_linear(params):
                a, C = params
                if C < 0:
                    return -np.inf
                return loglike([a, np.log10(C)])

            for _ in range(nwalkers):
                a = a_init + np.random.normal(0, 0.01)
                logC = guess_logC(a_init) + np.random.normal(0, 0.2)
                C = 10**logC
                pos.append([a, C])

            sampler = emcee.EnsembleSampler(nwalkers, 2, loglike_linear)

        pos, prob, state = sampler.run_mcmc(pos, 100) # burn in
        sampler.reset()
        sampler.run_mcmc(pos, nsteps)
        self._MCMCsampler = sampler

    @property
    def a(self):
        return self._MCMCsampler.flatchain[:,0]

    @property
    def C(self):
        if self.n == 0:
            return self._MCMCsampler.flatchain[:,1]
        else:
            return 10**self._MCMCsampler.flatchain[:,1]

    def logC(self):
        if self.n == 0:
            return np.log10(self._MCMCsampler.flatchain[:,1])
        else:
            return self._MCMCsampler.flatchain[:,1]


    def run_mcmc(self, nsteps):
        self._MCMCsampler.run_mcmc(None, nsteps)


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
        e = np.concatenate([data.e for data in self.flare_dataset.observations])
        elim = np.concatenate([[data.elim] * data.n for data in self.flare_dataset.observations])
        N = self.flare_dataset.n_total
        a = N / (np.sum(np.log(e / elim)))
        return a

    def plotfit(self, ax=None, line_kws=None, step_kws=None):
        if self.ml_success == False:
            raise ValueError('No best fit to plot for this Fit object.')
        if ax is None:
            ax = plt.gca()
        self.flare_dataset.plot_ffd(ax=ax, **step_kws)
        emin = min(*[d.elim for d in self.flare_dataset.observations])
        emax = max(*[np.max(d.e) for d in self.flare_dataset.observations])
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
    ax = kwargs.pop('ax', plt.gca())
    fmin, fmax = [cumulative_frequency(a, C, e) for e in  [emin, emax]]
    return ax.plot([emin, emax], [fmin, fmax], *args, **kwargs)[0]