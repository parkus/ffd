from matplotlib import pyplot as plt
import numpy as np
import emcee
from scipy.optimize import minimize
from scipy.special import gammaln
from warnings import warn
from scipy.stats import kstest, pareto
from scipy.interpolate import interpn
import os
from copy import deepcopy


class PowerLawFit(object):
    """
    A PowerLawFit packages up a power-law fit to flares with the flare data and convenient tools for those fits,
    while attempting to handle some of the nuances of the fitting process. The fit is of the form

    f = C*e**-a,

    where f is the cumulative frequency of flares with energies greater than e.

    Attributes
    ----------
    flare_dataset : FlareDataset
        The flare data.
    a_logprior : function
        The prior on a, if any. Should take a single value as
        input and return ln(likelihood) of that value.
    logC_logprior : function
        As with a.
    n : int
        Total number of detected flares.
    a : array
        MCMC-sampled values of the power-law index.
    C : array
        MCMC-sampled values of the power-law rate constant.
    logC : array
        MCMC-sampled of the values of the natural log of the power-law rate constant.

    """

    def __init__(self, flare_dataset, a_logprior=None, logC_logprior=None,
                 scale_limits=False, rate_only=False, fit=True, nwalkers=100, nsteps=100):
        """
        Create an object representing a power-law fit to the flare frequency distribution for the flare observations
        provided in a FlareDataset object. The fit is of the form

        f = C*e**-a,

        where f is the cumulative frequency of flares with energies greater than e.

        Parameters
        ----------
        flare_dataset : FlareDataset
            Flare data as an ffd.FlareDataset object.
        a_logprior : function
            The prior on a, if any. Should take a single value as input
            and return ln(likelihood) of that value.
        logC_logprior : function
            As with a.
        scale_limits : False, True, or float
            If for some reason you think you've underestimated the
            detection limits, you can scale them up. Providing a
            float will scale them all by that factor. Simply
            specifying True will result in the object looking for the
            lowest constant scaling factor that results in the data
            passing a KS test for consistency with  a power law.
            Biases abound here, so be cuatious.
        rate_only : bool
            If True, the sampler will not consider the likelihood of
            the data given the sampled power law parameters.
            Instead, it will only use the priors on a and logC and the
            Poisson likelihood of observing the number of events to
            constrain the posterior.
        fit : bool
            If False, do not attempt any MCMC sampling of the fit
            when initializing the object. (Major time saver if all
            you need is to use some  of the utilities.)
        nwalkers : int
            Number of MCMC walkers to use.
        nsteps : int
            Number of MCMC steps to take upon initialization.
        """
        # store attributes
        self.flare_dataset = flare_dataset
        self.a_logprior = _prior_boilerplate(a_logprior)
        self.logC_logprior = _prior_boilerplate(logC_logprior)
        self.n = flare_dataset.n_total

        # check for sensible input
        if fit and self.n < 3 and a_logprior is None:
            raise ValueError('At least 3 flares required to attempt a fit unless you place a prior on a. '
                             'Several more than 3 will likely be required for said fit to converge.')
        if fit and self.n == 0:
            warn('No flares in the flare dataset. Be cautious: for MCMC samples of the posterior of C and similar '
                 'derived quantities (such as flare rate), use only the linear values, not logC, and use them only to '
                 'constrain upper limits.')

        if scale_limits:
            if type(scale_limits) is float:
                limit_scale = scale_limits
            else:
                # create a list of possible scaling factors based on clipping off successive least energetic events
                # in the dataset
                limit_scale = 1.
                e, elim, _, _ = self._get_data_vecs(limit_scale, 'event')
                possible_scales = e/elim
                possible_scales = possible_scales[possible_scales > 1]
                possible_scales = np.sort(possible_scales)
                possible_scales = iter(possible_scales)

                # now run through these limits until a KS test for consistency with a power law is passed
                while True:
                    try:
                        limit_scale = possible_scales.next()
                    except StopIteration:
                        raise ValueError("No lower limit scaling could be found for which the data are consistent "
                                         "with a power law.")
                    a = self.index_analytic(limit_scale)
                    D, n = self.KS_test(a, limit_scale=limit_scale, alternative='less')
                    Dcrit = KS_Dcrit(a, n, 0.25)
                    if D < Dcrit:
                        break
        else:
            limit_scale = None
        self.limit_scale = limit_scale

        # region sample the fit posterior
        # the MCMC sampler seems to struggle to sample the posterior of the rate constant C well when there are no
        # flares and it is specified as logC (log10(C)). I think this is because it is basically unconstrained to -inf.
        # However, when there is a flare logC tends to be much more normally distributed than C and if C is used the
        # MCMC sampler does wacky things. So I'll treat the two separately.
        if fit:
            Cscale = 'log' if self.n > 0 else 'linear'
            sampler, pos, loglike = self._make_sampler(nwalkers, Cscale, limit_scale=limit_scale,  rate_only=rate_only)
            self.loglike = loglike

            # run some throw-away burn-in MCMC steps
            pos, prob, state = sampler.run_mcmc(pos, 100) # burn in
            sampler.reset()

            # MCMC sample either a,logC or a,C
            sampler.run_mcmc(pos, nsteps)

            # save the sampler, but don't let the user see it because the logC vs C possibility will definitely lead to confusion
            self._MCMCsampler = sampler
        # endregion

    #region properties
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
    #endregion

    #region boilerplate
    def _make_sampler(self, nwalkers, Cscale='log', limit_scale=1, rate_only=False):
        """
        Initialize the MCMC sampler for the fit. This gets complicated since it samples C or logC according to whether
        any events have been detected.
        """

        # make vectors of events, exposure times, and detection limits
        Fe, Felim, Fexpt, Fn = self._get_data_vecs(limit_scale, 'event')

        # first handle the power law portion of the likelihood
        if rate_only:
            def loglike_powerlaw(a_diff):
                return 0.0
        else:
            def loglike_powerlaw(a_diff):
                return np.sum(np.log((a_diff - 1) / Felim) - a_diff * np.log(Fe / Felim))

        # next handle the poisson portion of the likelihood
        e, elim, expt, n = self._get_data_vecs(limit_scale, 'obs')
        def loglike_poisson(a_cum, logC):
            # sometimes the MCMC sampler tries values for logC that are so low that 10**logC = 0 to computer precision
            # keeping lambda in log space avoids this resulting in a nan likelihood resulting from a np.log(0) operation
            loglam = np.log(expt) + logC*np.log(10.) - a_cum*np.log(elim)
            return np.sum(-np.e**loglam + n * loglam - gammaln(n + 1))

        # all together now!
        # Define likelihood as a function of a,logC so that the MCMC sampler will explore that space.
        # This avoids problems that occurred when the MCMC sampler tried to explore a,C space. The distribution in that
        # space is very narrow and highly curved, and it seems MCMC just can't accurately feel its way around it.
        #
        # However, an exception is when there are no flares to constrain the flare rate. We will get to that.
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

            # first the models of flare energies and rate
            result = loglike_powerlaw(a_cum + 1) + loglike_poisson(a_cum, logC)

            # next any priors
            result += self.a_logprior(a_cum) + self.logC_logprior(logC)

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
                return -loglike(params)
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
                pos.append([a, logC])
            ndim = 2
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

        return sampler, pos, loglike


    def _get_data_vecs(self, limit_scale=None, by_obs_or_event='event'):
        """
        Boilerplate code for getting data vectors that I seem to  use over and over.

        Parameters
        ----------
        limit_scale : see init
        by_obs_or_event : 'event' or 'obs'
            If 'event', n, elim, and expt are copied for
            each event. All returned vectors are then the
            same length, such that for each event energy
            the number of flares, etc. in the observation
            from which that event was detected can quickly
            be accessed. If 'obs', n, elim, and expt are
            simply aggregated from each observation.

        Returns
        -------
        e, elim, expt, n
        """
        if limit_scale is None:
            limit_scale = 1.
        expt, e, elim, n = [], [], [], []
        for obs in self.flare_dataset.observations:
            _elim = obs.elim * limit_scale
            keep = obs.e > _elim
            _e = obs.e[keep]
            _n = len(_e)
            fac = _n if by_obs_or_event == 'event' else 1
            e.append(_e)
            n.append([_n] * fac)
            elim.append([_elim] * fac)
            expt.append([obs.expt] * fac)
        return map(np.concatenate, (e, elim, expt, n))


    def _combined_normfac(self, a, limit_scale=None, obs_data=None):

        if obs_data is None:
            obs_data = self._get_data_vecs(limit_scale, 'obs')
        e, elim, expt, n = obs_data
        return a / np.sum(expt * elim**-a)
    #endregion


    def run_mcmc(self, nsteps):
        """
        Advance the MCMC sampling of the fit posterior by nsteps.
        Parameters
        ----------
        nsteps : int
            number of MCMC steps to take

        Returns
        -------
        None, done in-place.
        """
        self._MCMCsampler.run_mcmc(None, nsteps)


    def index_analytic(self, limit_scale=None, _event_vecs=None):
        """
        Returns a max-likelihood power law fit of the form

        f ~ e**-a

        where f is the cumulative frequency of flares with energies greater than e.

        The fit is corrected for the known bias in the ML parameter for a. See Crawford+ 1970.

        Returns
        -------
        a : float
            max likelihood power-law index
        """
        event_vecs = self._get_data_vecs(limit_scale, 'event') if _event_vecs is None else _event_vecs
        e, elim, expt, n = event_vecs
        N = len(e)
        a = N / (np.sum(np.log(e / elim)))
        a = (N-1)*a/N # corrects for bias per Crawford+ 1970
        return a


    def replace_energies(self, new_energies):
        """For MC testing. This allows a new fit object to be produced where the energies of the events have been
        changed and that is all. I made it to use so I could fit simulated datasets identical to the true dataset
        in every other way then compute D for the KS test from these simulated data, ultimately so that I could
        compute good p-values for the KS test."""
        new_energies = list(new_energies)
        new_fit = deepcopy(self)
        for obs in new_fit.flare_dataset.observations:
            n = len(obs.e)
            obs.e = np.array(new_energies[:n])
            new_energies = new_energies[n:]
        return new_fit


    def combined_CDF(self, e, a, limit_scale=None, obs_data=None):
        e = np.reshape(e, [-1])
        if obs_data is None:
            obs_data = self._get_data_vecs(limit_scale, 'obs')
        normfac = self._combined_normfac(a, limit_scale, obs_data=obs_data)
        _, elim, expt, _ = obs_data
        keep = elim[None,:] < e[:, None]
        bad = e <= 0
        with np.errstate(divide='ignore', invalid='ignore'):
            result = normfac/a * (np.sum((expt*elim**-a)[None,:]*keep,1) - np.sum(expt[None,:]*keep,1) * e ** -a)
        result[bad] = 0.
        return result


    def rvs(self, a, n, limit_scale=None, exact_replica=False, obs_data=None, event_data=None):
        """
        Generate random values based on a power law describing several datasets with the same detection limits as the
        data underlying the PowerLawFit object.

        Parameters
        ----------
        a : float
            index for cumulative distribution
        n : int
            number of events to draw
        limit_scale : float, optional
            Factor by which to scale the detection limits.
        exact_replica : bool
            If True, return exactly the same number of events for each
            observation. In this case, n is ignored. If False, the
            randomly generated events can come from any of the
            observations.
        obs_data  :  list, optional
            data from the observations. Comes from
            self._get_data_vecs(False, 'obs'). This is just here so the
            user can speed things up if this method is being called
            many times.
        event_data
            data from the events. Comes from
            self._get_data_vecs(False, 'event'). This is just here so the
            user can speed things up if this method is being called
            many times.
        Returns
        -------
        rvs : array
            Randomly-generated values for event energies.
        """
        if exact_replica:
            if event_data is None:
                event_data = self._get_data_vecs(limit_scale, 'event')
            n = self.flare_dataset.n_total
            elims = event_data[1]
            rvs = random_energies(a, elims, np.inf, n)
        else:
            if obs_data is None:
                obs_data = self._get_data_vecs(limit_scale, 'obs')
            rvs_uni = np.random.uniform(0, 1, n)
            normfac = self._combined_normfac(a, limit_scale, obs_data=obs_data)
            _, elim, expt, _ = obs_data
            Plims = self.combined_CDF(elim, a, limit_scale=limit_scale, obs_data=obs_data)
            sum_over = Plims[None,:] < rvs_uni[:,None]
            rvs = ((np.sum((expt*elim**-a)[None,:]*sum_over,1) - a/normfac*rvs_uni)/np.sum(expt[None,:]*sum_over,1))**(-1./a)
        return rvs


    #region goodness of fit utilities
    def PP(self, a, e=None, limit_scale=None):
        """
        Compute percentile-percentile values comparing analytic CDF and empirical CDF. If the two distributions match
        exactly (i.e. the data exactly follow a power-law distribution), then the returned values would  describe a
        straight line with slope of 1.

        Parameters
        ----------
        a : float
            Index of cumulative power-law.
        e : array, optional
            Energies at which to compute the PP values. If None,
            the actual event energies are used.
        limit_scale : float, optional
            Factor by which to scale detection limits.

        Returns
        -------
        p_analytic, p_empirical : arrays
            Percentiles from the analytic distribution and the empirical data.
        """
        obs_data = self._get_data_vecs(limit_scale, 'obs')
        if e is None:
            e = obs_data[0]
        e = np.sort(e)
        p_analytic = self.combined_CDF(e, a, limit_scale, obs_data)
        n = len(e)
        p_empirical = (np.arange(n) + 0.5)/n
        return p_analytic, p_empirical


    def stabilized_PP(self, a, e=None, limit_scale=None):
        """
        Compute a stabilized version of the PP values, intended to be more sensitive to deviations in the tail of the
        distribution.

        This is just 2/pi * arcsin(sqrt(p)) for the values returned from PP(). See Maschberger & Kroupa 2009.

        Parameters
        ----------
        a : float
            Index of cumulative power-law.
        e : array, optional
            Energies at which to compute the PP values. If None,
            the actual event energies are used.
        limit_scale : float, optional
            Factor by which to scale detection limits.

        Returns
        -------
        p_analytic, p_empirical : arrays
            "Stabilized" percentiles from the analytic distribution and the empirical data.
        """
        return [2/np.pi*np.arcsin(np.sqrt(p)) for p in self.PP(a, e, limit_scale)]


    def goodness_of_fit(self, a, limit_scale=None, maxMCtrials=10000, rel_perr=0.3, method='stabilized KS'):
        """
        Test for power-law behavior.

        This is computationally intense because p is computed based on Monte-Carlo trials. However, only as many MC
        trials are run as is needed to achieve (approximately) the desired relative precision on the p-value.

        Parameters
        ----------
        a : float
            Index of cumulative power-law.
        limit_scale : float, optional
            Factor by which to scale detection limits.
        maxMCtrials : int
            Maximum number of Monte-Carlo trials allowed before
            process is terminated.
        rel_perr : float
            Desired relative precision in the p-value. E.g., 0.1
            means that for a p-value of 0.01, the  process will
            halt once the error on that p-value is roughly 0.001.
            Hence, you will probably be comfortable with a relatively
            large value.
        method : "stabilized KS" or "anderson-darling"
            Statistic to use for assesing fit. See Maschberger & Kroupa 2009.
            "stabilized KS" is recommended as being more sensitive
            to deviations from power-law behavior in the tail of the
            distribution.

        Returns
        -------
        p : float
            The probability that randomly-generated data from power-laws
            that is then fit with its own power law would yield a
            test statistic indicating a worse fit than the actual
            data. Low values of p mean the fit is poor (i.e. random
            data from a true power-law rarely produced fits as bad
            as the actual data).
        p_uncertainty : float
            Estimated uncertainty on p.

        """

        # get some variables
        obs_data = self._get_data_vecs(limit_scale, 'obs')
        n = self.flare_dataset.n_total

        # define function for computing  test statistic
        if method == 'stabilized KS':
            def get_stat(a, e):
                ppx, ppy = self.stabilized_PP(a, e, limit_scale=limit_scale)
                return np.max(np.abs(ppy - ppx))
        if method == 'anderson-darling':
            def get_stat(a, e):
                ppx, ppy = self.PP(a, e, limit_scale=limit_scale)
                return -len(ppy) - np.sum(2 * ppy * (np.log(ppx) + np.log(1 - ppx[::-1])))

        # compute test statistic for actual dataset
        stat = get_stat(a, None)

        # define function for computing p-value and estimated uncertainty
        def get_p():
            n = np.sum(np.asarray(stat_mc) > stat)
            m = float(len(stat_mc))
            p = n/m
            perr = np.sqrt(n)/m
            return p, perr

        # now generate random data, fit a power-law to that data, and compute the test statistic for it
        # periodically compute the p-value and uncertainty and stop once the desired precision is reached
        count = 0
        stat_mc = []
        while True:
            if count > maxMCtrials:
                break
            e_rvs = self.rvs(a, n, obs_data=obs_data, exact_replica=True)
            newfit = self.replace_energies(e_rvs)
            newfit._make_sampler(10)
            a = newfit.ml_result.x[0]
            stat_mc.append(get_stat(a, e_rvs))
            if count >= 1000 and count % 100 == 0:
                p, perr = get_p()
                with np.errstate(invalid='ignore'):
                    if perr/p < rel_perr:
                        break
            count += 1

        return get_p()


    def KS_test(self, a, limit_scale=None, alternative='two-sided', mode='approx'):
        """
        Perform a Kolmogorov-Smirnov test for power-law behavior for the dataset.

        Parameters
        ----------
        a : float
            Index of cumulative power-law.
        limit_scale : float, optional
            Factor by which to scale detection limits.
        alternative : "two-sided", "less", "greater"
            Whether to consider devations on both sides, above,
            or below a perfect 1-1 match of the theoretical and
            empirical CDF.
        mode : "approx" or "asymp"
            How to compute test statistic (see scipy.stats.kstest docs).

        Returns
        -------
        D : float
            Test statistic (max distance between theoretical and empirical
            CDFs).
        n : int
            Number of data points (i.e. number of flares). D and n are
            needed to compute a p-value for the test.
        """
        obs_data = self._get_data_vecs(limit_scale, 'obs')
        CDF = lambda e: self.combined_CDF(e, a, limit_scale, obs_data)
        e = obs_data[0]
        D = kstest(e, CDF, alternative=alternative, mode=mode)[0]
        n = len(e)
        return D, n
    #endregion


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
        return prior


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


def cumulative_frequency(a, C, e, eref=1):
    """
    Cumulative frequency of events with energies (or other metric) greater than e for a power-law of the form

    f = C*e**-a

    to the flare energy distribution, where f is the cumulative frequency of flares with energies greater than e.
    """
    return C*(e/eref)**-a


def differential_frequency(a, C, e, eref=1):
    """
    Differential frequency of events with energies (or other metric) greater than e for a power-law of the form

    f = C*e**-a

    to the flare energy distribution, where f is the cumulative frequency of flares with energies greater than e.
    """
    return a*C*(e/eref)**(-a-1)


def time_average(a, C, emin, emax, eref=1):
    """
    Average output of events with energies in the range [emin, emax] for a power-law of the form

    f = C*e**-a

    to the flare energy distribution, where f is the cumulative frequency of flares with energies greater than e.

    If the power law is for flare equivalent durations, this amounts to the ratio of energy output in flares versus
    quiesence. If the power law is for flare energies, it is the time-averaged energy output of flares (units of power).
    """
    return a * C / (1 - a) * ((emax/eref) ** (1 - a) - (emin/eref) ** (1 - a))


def energy(a, C, f, eref=1):
    """
    Minimum energy (or other metric given as e below) of events that occur with a frequency of f for a cumulative
    frequency distribution of the form

    f = C*e**-a.
    """
    return (f/C)**(1/-a)*eref


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
    eref = kwargs.pop('eref', 1)
    ax = kwargs.pop('ax', plt.gca())
    fmin, fmax = [cumulative_frequency(a, C, e, eref) for e in  [emin, emax]]
    return ax.plot([emin, emax], [fmin, fmax], *args, **kwargs)[0]


def random_energies(a, emin, emax, n):
    """
    Generate random values for event energies from a power-law distribution where probability of an event with
    energy greater than e is proportional to e**-a.

    Parameters
    ----------
    a : float
        Cumulative index of power-law.
    emin : float
        Minimum event energy.
    emax : float
        Maximum event energy.
    n : int
        Number of events to draw.

    Returns
    -------
    energies : array
        Randomly-drawn energies.
    """
    # I found it easier to just make my own than figure out the numpy power, pareto, etc. random number generators
    norm = emin**-a - emax**-a
    x_from_cdf = lambda c: ((1-c)*norm + emax**-a)**(-1/a)
    x_uniform = np.random.uniform(size=n)
    return x_from_cdf(x_uniform)


def ML_index_analytic(x, xlim):
    """
    Returns a quick-and-dirty max-likelihood power law fit of the form

    f ~ e**-a

    where f is the cumulative frequency of flares with energies greater than e.

    Returns
    -------
    a : float
        max likelihood power-law index
    """
    N = len(x)
    a = N / (np.sum(np.log(x / xlim)))
    a = (N-1)*a/N # corrects for bias per Crawford+ 1970
    return a


#region KS test stuff
_path_ks_grid = os.path.join(os.path.dirname(__file__), 'power_KS_cube.npy')
def _generate_KS_cube():
    """
    Generate a grid of D values for KS tests of power-law behavior.
    """
    a_grid = np.arange(0.2, 2, 0.05)
    n_grid = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 200]
    m = 1000
    Dcube = np.zeros([len(a_grid), len(n_grid), m], dtype='f4')
    for i, a in enumerate(a_grid):
        for j, n in enumerate(n_grid):
            D = []
            for k in range(m):
                rvs = pareto.rvs(a, size=n)
                aML = ML_index_analytic(rvs, 1.)
                cdf = lambda x: pareto.cdf(x, aML)
                D.append(kstest(rvs, cdf)[0])
            Dcube[i,j] = np.sort(D)
    np.save(_path_ks_grid, np.array((a_grid, n_grid, Dcube)))


def KS_MC(a, n_events, n_draws=10000):
    """
    Run MC trials of computing KS D values for data draw from power law with cumulative index a.
    """

    D = []
    for _ in range(n_draws):
        rvs = pareto.rvs(a, size=n_events)
        aML = ML_index_analytic(rvs, 1.)
        cdf = lambda x: pareto.cdf(x, aML)
        D.append(kstest(rvs, cdf)[0])
    return  np.sort(D)


_KS_MCMC_ary = np.load(_path_ks_grid)
_n = _KS_MCMC_ary[2].shape[-1]
_cp_grid = (np.arange(_n) + 0.5)/_n
def KS_Dcrit(a, n, p):
    """
    Compute D value associated with a p-value  of p for a dataset of n events drawn from a power law distribution with
    cumulative index a.
    """
    a_grid, n_grid, Dcube = _KS_MCMC_ary
    return interpn((a_grid, n_grid, _cp_grid), Dcube, (a, n, 1-p))

def KS_p(a, n, D, n_draws=10000):
    """
    Estimate p-value for a KS test D value D for n events drawn from  a power-law distribtuion with index a using
    n_draws MC trials.
    """
    Dmc = KS_MC(a, n, n_draws)
    i = np.searchsorted(Dmc, D)
    return 1 - float(i)/n_draws
#endregion

