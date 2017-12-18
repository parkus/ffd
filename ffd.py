import numpy as np
import emcee as emcee
import utils as utils
from math import factorial
from matplotlib import pyplot as plt
import powerlaw


class Flares(object):
    """
    A Flares object contains the energy (or other defining metric) of flares from a compendium of datasets (
    specifically FlareDataset objects).

    Attributes
    ----------
    datasets : list
        The FlareDataset objects making up Flares.
    n_total : int
        Total number of flares summed across all datasets.
    expt_total : float
        expt_total: Total exposure time summed across all datasets.,
    e : array
        Flare energies (or other defining metric) concatenated from all datasets and sorted in value.,
    expt_detectable : array
        The total exposure time in which the event stored in "e" could have been detected. (I.e. the sum of the
        exposure times of datasets where the detection limit was below e.),
    cumfreq_naive : array
        Cumulative frequency of for events >=e assuming expt_total for all.,
    cumfreq_corrected : array
        Cumulative frequency for events >=e accounting for differences in total time in which such events could have
        been detected.}
    """

    def __init__(self, datasets):
        """
        Creat a Flares object.

        Parameters
        ----------
        datasets : list
            A list of FlareDataset objects. Make sure they use a consistent metric to characterize the flares (
            energy, equivalent duration, peak flux, etc.)

        Returns
        -------
        A Flares object :)

        """
        self.datasets = datasets

        # totals
        self.n_total = sum(data.n for data in datasets)
        self.expt_total = np.sum(data.expt for data in datasets)

        # event energies concatenated from all flare datasets
        self.e = np.concatenate(data.e for data in datasets)
        self.e = np.sort(self.e)


        # exposure time in which an event of energy e *could* have been detected
        expts = np.array(data.expt for data in datasets)
        elims = np.array(data.elim for data in datasets)
        isort = np.argsort(elims)
        elims, expts = [a[isort] for a in [elims, expts]]
        cumexpts = np.cumsum(expts)
        i_lims = np.searchsorted(elims, self.e, side='right')
        self.expt_detectable = cumexpts[i_lims-1]

        # cumulative frequencies ignoring differences in detection limits and correcting for them
        cumno = np.arange(self.n_total + 1)[::-1]
        self.cumfreq_naive = cumno / self.expt_total
        self.cumfreq_corrected = cumno / self.expt_detectable

    def plot_ffd(self, *args, **kwargs):
        """
        Plot a step function of the flare frequency distribution, with the option to adjust for differences in
        detection limits.

        Parameters
        ----------
        args :
            passed to the matplotlib plot function
        ax :
            matplotlib axes object on which to draw line
        corrected : boolean
            Whether to use the naive or corrected cumulative frequency. The naive value is more common for large
            datasets and produces a drop-off of the expected power-law at low flare energies.
        kwargs :
            passed to the matplotlib plot function

        Returns
        -------
        line :
            matplotlib line object
        """
        ax = kwargs.get('ax', plt.gca())
        corrected = kwargs.get('corrected', True)
        cf = self.cumfreq_corrected if corrected else self.cumfreq_naive
        line, = ax.step(self.e, cf, where='post', **kwargs)
        return line

    def fit_powerlaw_dirty(self):
        """
        Returns a quick-and-dirty max-likelihood power law fit of the form

        f ~ e**-a

        where f is the cumulative frequency of flares with energies greater than e.

        Returns
        -------
        a, aerr: floats
            max likelihood power-law index and error
        """
        e = np.concatenate(data.e for data in self.datasets)
        elim = np.concatenate([data.elim]*data.n for data in self.datasets)
        N = self.n_total
        a = N / (np.sum(np.log(e / elim)))
        aerr = N * a / (N - 1) / np.sqrt(N - 2)
        return a, aerr

    def loglike_powerlaw(self, params):
        """
        Returns the log-likelihood of a power law fit of the form

        f = C*e**-a

        to the flare energy distribution, where f is the cumulative frequency of flares with energies greater than e
        and the rate of events follows a Poisson distribution independent of event energies. Intended for use with an
        emcee MCMC sampler.

        Parameters
        ----------
        params : list
            [a, C] in the power law function specified above.

        Returns
        -------
        loglike : float
            Log-likelihood of the fit to the flare distribution.

        """
        return np.sum(data.loglike_powerlaw(params) for data in self.datasets)

    def mcmc_powerlaw(self, nwalkers=50, nsteps=10000, a_prior=None, logC_prior=None, a_init=None, C_init=None):
        """
        Generate a PowerLawMCMC for a power law fit of the form

        f = C*e**-a

        to the flare energy distribution, where f is the cumulative frequency of flares with energies greater than e.
        This is the meat and potatoes of this whole ffd.py endeavor :)

        Parameters
        ----------
        nwalkers : int
            Number of walkers to use in the MCMC sampling.
        nsteps : int
            Number of steps to take before returning the MCMC sampler (more steps can be taken later).
        a_prior : list or function
            A function given the log-likelihood of different a values. An interval can also be specified as a 2-item
            list that will be turned into a uniform distribution between these values.
        logC_prior : list or function
            Similar ot a_prior. Log10(C) is used because the C parameter tends to be normally-distributed in log space.
        a_init : float or None
            Initial value for a to start the MCMC sampling. A small random factor will be applied. If None,
            fit_powerlaw_dirty will be used to get an a_init.
        C_init : float or None
            Similar to C_init. Note that the value is *not* in log space. If None, a rough estimate based on the
            number of events detected and a_init will be used.

        Returns
        -------
        sampler : PowerLawMCMC object
            A class containing an emcee sampler of the posterior ditribution of the fit parameters and supporting
            methods. The sampler is returned with nsteps already taken.
        """

        a_prior, logC_prior = map(_prior_boilerplate, (a_prior, logC_prior))

        def loglike(params):
            return a_prior(params[0]) + logC_prior(np.log10(params[1])) + self.loglike_powerlaw(params)

        if a_init is None:
            a_init = self.fit_powerlaw_dirty()[0]
        if C_init is None:
            elim_mean = np.mean(data.elim for data in self.datasets)
            C_init = self.n / self.expt_total * elim_mean ** a_init

        pos = [[a_init, C_init] * np.random.normal(1, 1e-3, size=2) for _ in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, 2, loglike)
        sampler.run_mcmc(pos, nsteps)

        return PowerLawMCMC(sampler, a_prior=a_prior, logC_prior=logC_prior)

    def constrain_flare_energy_budget_mcmc(self, emin, emax, **fit_kws):
        """
        A convenience function to uses MCMC sampling to constrain the time-averaged energy resulting from flares
        assuming a power law distribution fit to the data. If the equivalent duration is used as the flare metric
        (the e attribute of each FlareDataset object), then this amounts to the ratio of energy emitted by flares
        versus quiescence over time.

        Parameters
        ----------
        emin : float
            minimum flare energy (or whatever metric was used when defining the FlareDatasets) to consider. Must be >0.
        emax : float
            similar to emin. Can use np.inf.
        fit_kws :
            keyword arguments to supply to the mcmc_powerlaw method when it is called

        Returns
        -------
        p, err_neg, err_pos

        """
        fit = self.mcmc_powerlaw(**fit_kws)
        p = fit.chain('time_average', emin, emax)
        return utils.error_bars(p)




class FlareDataset(object):

    def __init__(self, detection_limit, exposure_time, flare_energies=[]):
        """
        Create a FlareDataset object.

        Parameters
        ----------
        detection_limit : float
            Minimum energy (or other flare metric) of a detectable event.
        exposure_time : float
            Total time in which flares could have been detected.
        flare_energies : array-like
            Energies (or other metric like equivalent duration, peak flux, ...) of the detected events. Use an empty
            list (default) if no events were detected but the dataset is still being included.
        """
        self.elim = detection_limit
        self.expt = exposure_time
        self.e = np.array(flare_energies)
        self.n = len(flare_energies)

    def loglike_powerlaw(self, params):
        """
        Returns the log-likelihood of a power law fit of the form

        f = C*e**-a

        to the flare energy distribution, where f is the cumulative frequency of flares with energies greater than e
        and the rate of events follows a Poisson distribution independent of event energies. Intended for use with an
        emcee MCMC sampler.

        Parameters
        ----------
        params : list
            [a, C] in the distribution given above

        Returns
        -------
        loglike : float
            log likelihood of powerlaw fit

        """
        a, C = params
        if a < 0 or C < 0:
            return -np.inf
        if C == 0:
            if self.n > 0:
                return -np.inf
            else:
                return 0.0

        lam = self.expt * C * self.elim ** (1 - a)
        if self.n == 0:
            return -lam
        else:
            a = a + 1  # supplying cumulative but want regular exponent
            n = self.n
            log = np.log
            elim = self.elim
            return -lam + n*log(lam) - log(factorial(n)) + n*log((a-1)/elim) - a*np.sum(log(self.e/elim))




class PowerLawMCMC(object):
    """
    PowerLawMCMC is just an emcee sampler with some helper functions specific to a power law fit to a flare FFD.

    Attributes
    ----------
    sampler : emcee sampler object
    a_prior : function or 0
        the function defining the prior probability distribution on a. Zero if no prior.
    logC_prior : function or 0
        as with a_prior. logC used becuase posterior tends to be normally distributed in a and log10(C).
    burnin : int
        number of steps to consider burnin
    confidence_interval : float
        when defining erro bars pick them so that the value has this probability of being within them. Default is
        0.683 for 1-sigma errors.
    upper_limit_interval : float
        as with confidence_interval, but if only an upper limit can be specified
    is_good_fit : boolean
        True if best-fit values (not upper limits) were found for a and C.
    """

    def __init__(self, mcmc_sampler, a_prior, logC_prior, burnin=50, confidence_interval=0.683,
                 upper_limit_interval=0.95):
        """
        Create a PowerLawMCMC object. Intended to be created by a call to Flares.mcmc_powerlaw.
        """
        self.sampler = mcmc_sampler
        self.a_prior = a_prior
        self.logC_prior = logC_prior
        self.burnin = burnin
        self.confidence_interval = confidence_interval
        self.upper_limit_interval = upper_limit_interval
        self.update_best_fit_values()

    #region checking upper limits
    def is_lim(self, name):
        """
        Check if a the parameter given as name is an upper limit.
        """
        return np.isnan(getattr(self, name)[0])

    def _get_is_good_fit(self):
        aislim, Cislim = map(self.is_lim, ['a', 'C'])
        return (not aislim) and (not Cislim)
    is_good_fit = property(_get_is_good_fit)

    def check_fit_is_good(self):
        """
        Check if the fit is "good," meaning that best-fit values (not upper limits) were found for a and C. Raise a
        ValueError if it isn't.
        """
        if not self.is_good_fit:
            raise ValueError('Either a or C of the PowerLawMCMC fit is just an upper limit, so cannot compute.')
    #endregion

    # region powerlaw computations
    # I thought about abstracting here to make any function in powerlaw atuomatically work, but then they
    # introspection wn't work, so I think better just to explicitly duplicate
    def _compute_from_powerlaw(self, func, *args, **kwargs):
        self.check_fit_is_good()
        return getattr(powerlaw, func)(self.a[0], self.C[0], *args, **kwargs)

    def time_average(self, emin, emax):
        """
        Average output of events with energies in the range [emin, emax] for a power-law of the form

        f = C*e**-a

        to the flare energy distribution, where f is the cumulative frequency of flares with energies greater than e.

        If the power law is for flare equivalent durations, this amounts to the ratio of energy output in flares versus
        quiesence. If the power law is for flare energies, it is the time-averaged energy output of flares (units of power).
        """
        return self._compute_from_powerlaw(emin, emax)

    def cumulative_frequency(self, emin):
        """
        Average output of events with energies in the range [emin, emax] for a power-law of the form

        f = C*e**-a

        to the flare energy distribution, where f is the cumulative frequency of flares with energies greater than e.

        If the power law is for flare equivalent durations, this amounts to the ratio of energy output in flares versus
        quiesence. If the power law is for flare energies, it is the time-averaged energy output of flares (units of power).
        """
        return self._compute_from_powerlaw(emin)

    def differential_frequency(self, e):
        """
        Differential frequency of events with energies (or other metric) greater than e for a power-law of the form

        f = C*e**-a

        to the flare energy distribution, where f is the cumulative frequency of flares with energies greater than e.
        """
        return self._compute_from_powerlaw(e)
    #endregion

    #region mcmc and fit management
    def run_mcmc(self, nsteps):
        """
        Advance the MCMC walkers by nsteps and update best fit values for a and C.
        """
        self.sampler.run_mcmc(None, nsteps)
        self.update_best_fit_values()

    def _get_fit_value(self, name):
        chain = getattr(self, name)
        return utils.error_bars(chain, self.confidence_interval, self.upper_limit)

    def update_best_fit_values(self):
        """
        Update the best-fit values for a and C.
        """
        self.a, self.C = map(self._get_fit_value, ['a', 'C'])
    #endregion

    def samples(self, name_or_function, *args, **kwargs):
        """
        Compute values from the chains of MCMC steps. This is particularly useful for examining the posterior of
        derived values, like the flare energy budget.

        Parameters
        ----------
        name_or_function : str or function
            Either a string naming the parameter return samples for ('a', 'C', or the name of afunction defined in
            powerlaw.py) or a user-defined function that takes vectors of a and C values as arguments.
        args :
            arguments to be passed along to the called function
        kwargs :
            as with args

        Returns
        -------
        samples : array
            an array of values of the function for each MCMC step
        """
        if type(name_or_function) is str:
            if name_or_function == 'a':
                return self.sampler.chain[:, self.burnin:, 0]
            if name_or_function == 'C':
                return self.sampler.chain[:, self.burnin:, 1]
            else:
                getattr(powerlaw, name_or_function)(self.chain('a'), self.chain('C'), *args, **kwargs)
        elif hasattr(name_or_function, '__call__'):
            return name_or_function(self.chain('a'), self.chain('C'), *args, **kwargs)

    def plot(self, emin, emax, *args, **kwargs):
        """
        Plot the power-law best fit line between emin and emax.

        Parameters
        ----------
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
        fmin, fmax = map(self.cumulative_frequency, [emin, emax])
        return ax.plot([emin, emax], [fmin, fmax], *args, **kwargs)




def _loglike_from_interval(x, interval):
    if x < interval[0] or x > interval[-1]:
        return -np.inf
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