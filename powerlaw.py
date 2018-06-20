from matplotlib import pyplot as plt
import numpy as np
import emcee
from scipy.optimize import minimize
from scipy.special import gammaln
from warnings import warn
from scipy.stats import kstest




class PowerLawFit(object):

    def __init__(self, flare_dataset, a_logprior=None, logC_logprior=None,
                 scale_limits=False, nwalkers=10, nsteps=1000):
        """
        Create an object representing a power-law fit to the flare frequency distribution for the flare observations
        provided in a FlareDataset object. The fit is of the form

        f = C*e**-a,

        where f is the cumulative frequency of flares with energies greater than e.
        """
        include_errors = flare_dataset.has_errors
        self.flare_dataset = flare_dataset
        self.a_prior = _prior_boilerplate(a_logprior)
        self.logC_prior = _prior_boilerplate(logC_logprior)
        self.n = flare_dataset.n_total

        if self.n < 3 and a_logprior is None:
            raise ValueError('At least 3 flares required to attempt a fit unless you place a prior on a. '
                             'Several more than 3 will likely be required for said fit to converge.')
        if self.n == 0:
            warn('No flares in the flare dataset. Be cautious: for MCMC samples of the posterior of C and similar '
                 'derived quantities (such as flare rate), use only the linear values, not logC, and use them only to '
                 'constrain upper limits.')

        if scale_limits:
            limit_scale = 1.
            e, _, elim, _, _ = self._get_data_vecs(limit_scale, 'power')
            possible_scales = e/elim
            possible_scales = possible_scales[possible_scales > 1]
            possible_scales = iter(possible_scales)
            while True:
                a = self.index_analytic(limit_scale)
                D, p = self.KS_test(a, limit_scale=limit_scale, alternative='less')
                if p < 0.1:
                    try:
                        limit_scale = possible_scales.next()
                    except StopIteration:
                        raise ValueError("No lower limit scaling could be found for which the data are consistent "
                                         "with a power law.")
                else:
                    break
        else:
            limit_scale = None

        # region sample the fit posterior
        # the MCMC sampler seems to struggle to sample the posterior of the rate constant C well when there are no
        # flares and it is specified as logC (log10(C)). I think this is because it is basically unconstrained to -inf.
        # However, when there is a flare logC tends to be much more normally distributed than C and if C is used the
        # MCMC sampler does wacky things. So I'll treat the two separately.
        Cscale = 'log' if self.n > 0 else 'linear'
        sampler, pos, loglike = self._make_sampler(nwalkers, Cscale, limit_scale=limit_scale,
                                                   include_errors=include_errors)
        self.loglike = loglike

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


    def _make_sampler(self, nwalkers, Cscale='log', limit_scale=1, include_errors=False):
        # make vectors of events, exposure times, and detection limits
        Fe, Ferr, Felim, Fexpt, Fn = self._get_data_vecs(limit_scale, 'power')

        # first handle the power law portion of the likelihood
        def loglike_powerlaw(a_diff):
            return np.sum(np.log((a_diff - 1) / Felim) - a_diff * np.log(Fe / Felim))

        # next handle the poisson portion of the likelihood
        e, err, elim, expt, n = self._get_data_vecs(limit_scale, 'poiss')
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

        # make a function for guessing appropriate rate constants given values of a
        elim_mean = np.mean([data.elim for data in self.flare_dataset.observations])
        def guess_logC(a_guess):
            n = self.flare_dataset.n_total
            if n == 0:
                n = 1
            return np.log10(n / self.flare_dataset.expt_total * elim_mean ** a_guess)

        if Cscale == 'log': # sample in a,logC space

            # try to find max-likelihood values of a, logC using numerical minimization of -loglike
            # use 1/a instead of a as a parameter to avoid bias per Crawford+ 1970
            def neglike(params):
                return -loglike(params, include_errors=False)
            a_guess = self.index_analytic()
            if not 0 < a_guess < 2:
                a_guess = 1.0
            logC_guess = guess_logC(a_guess)
            result = minimize(neglike, [a_guess, logC_guess], method='Nelder-Mead')

            # store the max-likelihood values
            self.ml_result = result
            if not result.success or not np.all(np.isfinite(result.x)):
                self.ml_success = False
                a_init = 1.0
            else:
                self.ml_success = True
                a, logC = result.x
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
                    return -np.infs
                return loglike([a, np.log10(C)])

            # get ready to MCMC sample the posterior of a,C
            for _ in range(nwalkers):
                a = a_init + np.random.normal(0, 0.01)
                logC = guess_logC(a_init) + np.random.normal(0, 0.2)
                C = 10**logC
                pos.append([a, C])
            sampler = emcee.EnsembleSampler(nwalkers, 2, loglike_linear)

        return sampler, pos, loglike


    def index_analytic(self, limit_scale=None, _data_vecs=None):
        """
        Returns a quick-and-dirty max-likelihood power law fit of the form

        f ~ e**-a

        where f is the cumulative frequency of flares with energies greater than e.

        Returns
        -------
        a : float
            max likelihood power-law index
        """
        data_vecs = self._get_data_vecs(limit_scale, 'power') if _data_vecs is None else _data_vecs
        e, err, elim, expt, n = data_vecs
        N = len(e)
        a = N / (np.sum(np.log(e / elim)))
        a = (N-1)*a/N # corrects for bias per Crawford+ 1970
        return a


    def _get_data_vecs(self, limit_scale=None, power_or_poiss='power'):
        if limit_scale is None:
            limit_scale = 1.
        expt, e, elim, n, err = [], [], [], [], []
        for obs in self.flare_dataset.observations:
            _elim = obs.elim * limit_scale
            keep = obs.e > _elim
            _e = obs.e[keep]
            _n = len(_e)
            fac = _n if power_or_poiss == 'power' else 1
            e.append(_e)
            err.append(obs.err[keep])
            n.append(_n * fac)
            elim.append([_elim] * fac)
            expt.append([obs.expt] * fac)
        return map(np.concatenate, (e, err, elim, expt, n))


    def _combined_normfac(self, a, limit_scale=None, data_vecs=None):
        if data_vecs is None:
            data_vecs = self._get_data_vecs(self, limit_scale, 'poiss')
        e, err, elim, expt, n = data_vecs
        return a / np.sum(expt * elim**-a)


    def combined_CDF(self, e, a, limit_scale=None, data_vecs=None):
        if data_vecs is None:
            data_vecs = self._get_data_vecs(self, limit_scale, 'poiss')
        normfac = self._combined_normfac(a, limit_scale, data_vecs=data_vecs)
        _, _, elim, expt, _ = data_vecs
        keep = elim[None,:] < e[:, None]
        return normfac/a * (np.sum((expt*elim**-a)[None,:]*keep,1) + np.sum(expt[None,:]*keep,1) * e ** -a)


    def PP(self, a, limit_scale=None):
        data_vecs = self._get_data_vecs(limit_scale, 'poiss')
        e, err, elim, expt, n = data_vecs
        e = np.sort(e)
        p_analytic = self.combined_CDF(e, a, limit_scale, data_vecs)
        n = len(e)
        p_empirical = (np.arange(n) + 0.5)/n
        return p_analytic, p_empirical


    def stabilized_PP(self, a, limit_scale=None):
        return [2/np.pi*np.arcsin(np.sqrt(p)) for p in self.PP(self, a, limit_scale)]


    def KS_test(self, a, limit_scale=None, alternative='two-sided', mode='approx'):
        CDF = lambda e: self.combined_CDF(e, a, limit_scale)
        return kstest(self.flare_dataset.e, CDF, alternative=alternative, mode=mode)


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
        return prior
    else:
        pass


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