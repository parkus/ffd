from matplotlib import pyplot as plt
import numpy as np
import emcee
from scipy.optimize import minimize
from scipy.special import gammaln
from warnings import warn

def loglike_from_interval(interval):
    """
    Create a function to represent a uniform likelihood across an interval, helpful for specifying a prior.

    Parameters
    ----------
    interval : [lo, hi]
        The lower and upper limits of the interval (exclusive). Can be -/+ np.inf.

    Returns
    -------
    loglike : function
        A function loglike(x) that gives 0.0 when x is in interval and -np.inf when it is not.

    """
    def loglike(x):
        if x < interval[0] or x > interval[-1]:
            return -np.inf
        else:
            return 0.0

    return loglike


def loglike_from_hist(bins, counts):
    """
    Create a function to represent a likelihood function based on histogrammed values.

    Parameters
    ----------
    bins : bin edges used in the histogram.
    counts : counts in each bin (histogram will be normalized such that integral is 1, though it shouldn't really matter)

    Returns
    -------
    loglike : function
        A function loglike(x) that returns ln(likelihood(x)) based on the value of the input histogram at x.
    """

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
    """
    Standardize input for prior distributions.

    Parameters
    ----------
    prior : None, [lo, hi] interval, or function
        user input for prior

    Returns
    -------
    loglike : function
        Standardized loglikelihood function.
    """

    if prior is None:
        return lambda x: 0.0
    elif not hasattr(prior, '__call__'):
        try:
            return loglike_from_interval(prior)
        except TypeError:
            raise ValueError('a_prior must either be a function or a list/tuple/array. See docstring.')
    else:
        return prior

#FIXME I should pick ML value of 1/a as best since this will be unbiased per crawford+ 1970
class PowerLawFit(object):

    def __init__(self, flare_dataset, a_prior=None, logC_prior=None, nwalkers=10, nsteps=1000):
        """
        Create an object representing a power-law fit to the flare frequency distribution for the flare observations
        provided in a FlareDataset object. The fit is of the form

        f = C*e**-a,

        where f is the cumulative frequency of flares with energies greater than e.
        """
        include_errors = flare_dataset.has_errors
        self.flare_dataset = flare_dataset
        self.a_prior = _prior_boilerplate(a_prior)
        self.logC_prior = _prior_boilerplate(logC_prior)
        self.n = flare_dataset.n_total

        if self.n < 3 and a_prior is None:
            raise ValueError('At least 3 flares required to attempt a fit unless you place a prior on a. '
                             'Several more than 3 will likely be required for said fit to converge.')
        if self.n == 0:
            warn('No flares in the flare dataset. Be cautious: for MCMC samples of the posterior of C and similar '
                 'derived quantities (such as flare rate), use only the linear values, not logC, and use them only to '
                 'constrain upper limits.')

        # region log likelihood function
        #
        # now make a function to rapidly compute the log likelihood. This would be much more readable if I separately
        # computed the log likelihood for each flare observation object and added them. However, it is much faster
        # computationally if I aggregate the data from each and compute likelihoods "all at once."

        # make vectors of events, exposure times, and detection limits
        expt, elim, n, e, err = [], [], [], [], []
        for obs in self.flare_dataset.observations:
            _n = obs.n
            expt.extend([obs.expt] * _n)
            elim.extend([obs.elim] * _n)
            n.extend([_n] * _n)
            e.extend(obs.e.tolist() if obs.n > 0 else [])
            if flare_dataset.has_errors:
                err.extend(obs.e_err.tolist() if obs.n > 0 else [])
        Fexpt, Felim, Fn, Fe, Ferr = map(np.array, [expt, elim, n, e, err])

        # first handle the power law portion of the likelihood
        def loglike_powerlaw(a_diff):
            return np.sum(np.log((a_diff - 1) / Felim) - a_diff * np.log(Fe / Felim))

        # next handle the poisson portion of the likelihood
        n = [obs.n for obs in self.flare_dataset.observations]
        elim = [obs.elim for obs in self.flare_dataset.observations]
        expt = [obs.expt for obs in self.flare_dataset.observations]
        expt, elim, n = map(np.array, [expt, elim, n])
        def loglike_poisson(a_cum, logC):
            # sometimes the MCMC sampler tries values for logC that are so low that 10**logC = 0 to computer precision
            # keeping lambda in log space avoids this resulting in a nan likelihood resulting from a np.log(0) operation
            loglam = np.log(expt) + logC*np.log(10.) - a_cum*np.log(elim)
            return np.sum(-np.e**loglam + n * loglam - gammaln(n + 1))

        # now handle the "nuissance" likelihood of the event energies
        def loglike_nuissance(e_vec):
                return np.sum(-0.5*np.log(2*np.pi) - np.log(Ferr) - (e_vec - Fe)**2/2/Ferr**2)

        # all together now!
        # Define likelihood as a function of a,logC so that the MCMC sampler will explore that space.
        # This avoids problems that occurred when the MCMC sampler tried to explore a,C space. The distribution in that
        # space is very narrow and highly curved, and it seems MCMC just can't accurately feel its way around it.
        #
        # However, an exception is when there are no flares to constrain the flare rate. We will get to that.
        def loglike(params, include_errors=include_errors):
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
            if include_errors:
                a_cum, logC = params[:2]
                e_vec = params[2:]
            else:
                a_cum, logC = params

            # always enforce this general prior on a
            if a_cum <= 0:
                return -np.inf

            # first the models of flare energies and rate
            result = loglike_powerlaw(a_cum + 1) + loglike_poisson(a_cum, logC)

            # next any priors
            result += self.a_prior(a_cum) + self.logC_prior(logC)

            # finally, measurement uncertainty of the event energies
            if include_errors:
                result += loglike_nuissance(e_vec)

            return result

        # save the loglike function for user access, if desired.
        self.loglike = loglike
        # endregion

        # region sample the fit posterior

        # make a function for guessing appropriate rate constants given values of a
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
        if flare_dataset.n_total > 0: # sample in a,logC space

            # try to find max-likelihood values of a, logC using numerical minimization of -loglike
            # use 1/a instead of a as a parameter to avoid bias per Crawford+ 1970
            def neglike(params):
                return -loglike(params, include_errors=False)
            a_guess = self._quick_n_dirty_index()
            if not 0 < a_guess < 2:
                a_guess = 1.0
            logC_guess = guess_logC(a_guess)
            result = minimize(neglike, [a_guess, logC_guess], method='Nelder-Mead')

            # store the max-likelihood values
            self.ml_result = result
            if not result.success or not np.all(np.isfinite(result.x)):
                self.ml_success = False
                self.a_best = None
                self.logC_best = None
                a_init = 1.0
            else:
                self.ml_success = True
                a, logC = result.x
                n = self.flare_dataset.n_total
                self.a_best = (n-1)*a/n
                self.logC_best = logC
                a_init = a

            # get ready to MCMC sample the posterior
            pos = []
            for _ in range(nwalkers):
                a = a_init + np.random.normal(0, 0.01)
                logC = guess_logC(a_init) + np.random.normal(0, 0.2)
                if include_errors:
                    e_vec = Fe + np.random.normal(0, Ferr/10., size=len(Fe))
                    pos.append([a, logC] + e_vec.tolist())
                else:
                    pos.append([a, logC])
            ndim = 2 + len(Fe) if include_errors else 2
            sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike)

        else: # no flares in dataset, sample in a,C space

            # no max-like fit is possible in this case
            self.ml_success = False
            self.a_best = None
            self.logC_best = None
            a_init = 1.0
            pos = []

            # make a loglike function that allows sampling of a,C instead of a,logC
            def loglike_linear(params):
                a, C = params
                if C < 0:
                    return -np.inf
                return loglike([a, np.log10(C)])

            # get ready to MCMC sample the posterior of a,C
            for _ in range(nwalkers):
                a = a_init + np.random.normal(0, 0.01)
                logC = guess_logC(a_init) + np.random.normal(0, 0.2)
                C = 10**logC
                pos.append([a, C])
            sampler = emcee.EnsembleSampler(nwalkers, 2, loglike_linear)

        # run some throw-away burn-in MCMC steps
        pos, prob, state = sampler.run_mcmc(pos, 100) # burn in
        sampler.reset()

        # MCMC sample either a,logC or a,C
        sampler.run_mcmc(pos, nsteps)

        # save the sampler, but don't let the user see it because the logC vs C possibility will definitely lead to confusion
        self._MCMCsampler = sampler
        # endregion

    @property
    def a(self):
        warn('You might be tempted to use the mode of expectation value of this distribution as the best-estimate for '
             'a. However, that will be a biased estimate (see Crawford+ 1970 or Maschberger+ 2009 among others.'
             'Use the  a_best property for an estimate that is less likely to be biased.')
        return self._MCMCsampler.flatchain[:,0]

    @property
    def C(self):
        if self.n == 0:
            return self._MCMCsampler.flatchain[:,1]
        else:
            return 10**self._MCMCsampler.flatchain[:,1]

    @property
    def logC(self):
        if self.n == 0:
            return np.log10(self._MCMCsampler.flatchain[:,1])
        else:
            return self._MCMCsampler.flatchain[:,1]


    def run_mcmc(self, nsteps):
        """
        Advance the MCMC sampling of the fit posterior by nsteps.
        Parameters
        ----------
        nsteps : int
            number of MCMC steps to take

        Returns
        -------

        """
        self._MCMCsampler.run_mcmc(None, nsteps)


    def _quick_n_dirty_index(self):
        """
        Returns a quick-and-dirty max-likelihood power law fit of the form

        f ~ e**-a

        where f is the cumulative frequency of flares with energies greater than e.

        Returns
        -------
        a : float
            max likelihood power-law index
        """
        e = np.concatenate([data.e for data in self.flare_dataset.observations])
        elim = np.concatenate([[data.elim] * data.n for data in self.flare_dataset.observations])
        N = self.flare_dataset.n_total
        a = N / (np.sum(np.log(e / elim)))
        a = (N-1)*a/N # corrects for bias per Crawford+ 1970
        return a

    def plotfit(self, ax=None, line_kws=None, step_kws=None):
        """
        Plot a the corrected cumulative distribution of flares and the power-law max-likelihood fit.

        Parameters
        ----------
        ax : axes on which to draw plot
        line_kws : dict of keywords to pass on to plot of best-fit line
        step_kws : dict of keywords to pass on to step plot of observed flare cumulative frequencies

        Returns
        -------

        """
        if self.ml_success == False:
            raise ValueError('No best fit to plot for this Fit object.')
        if ax is None:
            ax = plt.gca()
        self.flare_dataset.plot_ffd(ax=ax, **step_kws)
        emin = min(*[d.elim for d in self.flare_dataset.observations])
        emax = max(*[np.max(d.e) for d in self.flare_dataset.observations])
        plot(self.a_best, 10 ** self.C_ml, emin, emax, **line_kws)


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
    ax : axes object
        axes on which to plot the line
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
def random_energies(a, emin, emax, n):
    # I found it easier to just make my own than figure out the numpy power, pareto, etc. random number generators
    norm = emin**-a - emax**-a
    x_from_cdf = lambda c: ((1-c)*norm + emax**-a)**(-1/a)
    x_uniform = np.random.uniform(size=n)
    return x_from_cdf(x_uniform)