import numpy as np
import emcee as emcee
from math import factorial
from matplotlib import pyplot as plt
import powerlaw
from scipy.special import gammaln


class Flares(object):
    """
    A Flares object contains the energy (or other defining metric) of flares from a compendium of datasets
    (specifically FlareDataset objects).

    Attributes
    ----------
    datasets : list
        The FlareDataset objects making up Flares.
    n_total : int
        Total number of flares summed across all datasets.
    expt_total : float
        Total exposure time summed across all datasets.
    e : array
        Flare energies (or other defining metric) concatenated from all datasets and sorted in value.
    expt_detectable : array
        The total exposure time in which the event stored in "e" could have been detected. (I.e. the sum of the
        exposure times of datasets where the detection limit was below e.)
    cumfreq_naive : array
        Cumulative frequency of for events >=e assuming expt_total for all.
    cumfreq_corrected : array
        Cumulative frequency for events >=e accounting for differences in total time in which such events could have
        been detected.
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
        self.n_total = sum([data.n for data in datasets])
        self.expt_total = np.sum([data.expt for data in datasets])

        # event energies concatenated from all flare datasets
        self.e = np.concatenate([data.e for data in datasets])
        self.e = np.sort(self.e)

        # exposure time in which an event of energy e *could* have been detected
        expts = np.array([data.expt for data in datasets])
        elims = np.array([data.elim for data in datasets])
        isort = np.argsort(elims)
        elims, expts = [a[isort] for a in [elims, expts]]
        self.expt_at_lim = np.cumsum(expts)
        self.elims, self.expts = elims, expts
        count, _ = np.histogram(self.e, np.append(elims, self.e[-1]+1))
        self.n_detected = np.cumsum(count[::-1])[::-1]

        # cumulative frequencies ignoring differences in detection limits and correcting for them
        cumno = np.arange(self.n_total)[::-1] + 1
        self.cumfreq_naive = (cumno / self.expt_total)
        i_lims = np.searchsorted(elims, self.e, side='right')
        expt_detectable = self.expt_at_lim[i_lims - 1]
        cumno_corrected = np.cumsum(1. / expt_detectable[::-1])[::-1] * self.expt_total
        self.cumfreq_corrected = cumno_corrected / self.expt_total

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
        line, = ax.step(self.e, cf, where='pre', **kwargs)
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
        e = np.concatenate([data.e for data in self.datasets])
        elim = np.concatenate([[data.elim]*data.n for data in self.datasets])
        N = self.n_total
        a = N / (np.sum(np.log(e / elim)))
        assert a > 0
        aerr = N * a / (N - 1) / np.sqrt(N - 2)
        return a, aerr

    def loglike_powerlaw(self, params, e_uplim=np.inf):
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

        # I think this is not just a matter of summing the loglikes of the individual datasets because for a poisson
        # distribution, the prob of detecting some number of events over a given timeframe cannot be decomposed into a
        # product of the probabilities over pieces of that timeframe
        #
        # but I think I can decompose into event energy intervals
        # a, C = params
        # elims, expts = self.elims, self.expt_at_lim
        # elims = np.append(elims, e_uplim)
        # n = np.append(-np.diff(self.n_detected), self.n_detected[-1]) # number actually detected in each interval of limits
        # lams = C*expts*(elims[:-1]**-a - elims[1:]**-a) # expected no of events
        # poisson_loglikes = -lams + n*np.log(lams) - gammaln(n+1)
        # poisson_loglike = np.sum(poisson_loglikes)
        #
        # # now the events also have to account for different detection threshholds, but assume the same slope...
        # # here it is just the product of the probs for each dataset, so I can sum the loglikes
        # power_loglike = np.sum([d.loglike_powerindex(params, e_uplim=e_uplim) for d in self.datasets])

        # return poisson_loglike + power_loglike

        a, C = params
        xmin = np.concatenate([[d.elim]*d.n for d in self.datasets])
        expt = np.concatenate([[d.expt]*d.n for d in self.datasets])
        N = np.concatenate([[d.n]*d.n for d in self.datasets])
        x = np.concatenate([d.e for d in self.datasets])
        a = a + 1  # supplying cumulative but want regular exponent
        lam = expt * C * xmin ** (1 - a)
        return np.sum(-(lam - gammaln(N + 1)) / N + np.log(lam) + np.log((a - 1) / xmin) - a * np.log(x / xmin))

    def mcmc_powerlaw(self, nwalkers=50, nsteps=10000, a_prior=(0,np.inf), logC_prior=None, a_init=1.0, C_init=None,
                      e_uplim=np.inf):
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
        chain : array
            MCMC chain of a,C values, given as an [nsteps,2] array. Hence a,C = chain.T.
        """

        a_prior, logC_prior = map(_prior_boilerplate, (a_prior, logC_prior))

        def loglike(params):
            a, C = params
            if C <= 0:
                return -np.inf
            if a <= 0:
                return -np.inf
            return a_prior(a) + logC_prior(np.log10(C)) + self.loglike_powerlaw(params, e_uplim)

        if C_init is None:
            elim_mean = np.mean([data.elim for data in self.datasets])
            C_init = self.n_total / self.expt_total * elim_mean ** a_init

        pos = [[a_init, C_init] * np.random.normal(1, 1e-3, size=2) for _ in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, 2, loglike)
        sampler.run_mcmc(pos, nsteps)

        return sampler.flatchain

    def mcmc_energy_budget(self, emin, emax, fmin=None, fmax=None, chain=None, **fit_kws):
        """
        A convenience function to uses MCMC sampling to constrain the time-averaged energy resulting from flares
        assuming a power law distribution fit to the data. If the equivalent duration is used as the flare metric
        (the e attribute of each FlareDataset object), then this amounts to the ratio of energy emitted by flares
        versus quiescence over time.

        This is mainly convenience code for my 2018 flare paper :)

        Parameters
        ----------
        emin : float
            minimum flare energy (or whatever metric was used when defining the FlareDatasets) to consider. Must be >=0.
        emax : float
            similar to emin. Can use np.inf.
        fmin : float
            minimum flare frequency to consider (equivalent to max energy, but note that the relationship between
            the two depends on a and C). Must be >=0. Overrides emax.
        fmax : float
            max flare frequency to consider. Can be np.inf. Overrides emin.
        chain : None
        fit_kws :
            keyword arguments to supply to the mcmc_powerlaw method when it is called

        Returns
        -------
        p, err_neg, err_pos

        """
        if chain is None:
            chain = self.mcmc_powerlaw(**fit_kws)
        a, C = chain

        if fmin is not None:
            emax = powerlaw.energy(a, C, fmin)
        if fmax is not None:
            emin = powerlaw.energy(a, C, fmax)
        return powerlaw.time_average(a, C, emin, emax)




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
        if np.any(flare_energies < detection_limit):
            raise ValueError('Detections below the detection limit don\'t make sense, but there appears to be one.')
        self.elim = detection_limit
        self.expt = exposure_time
        self.e = np.array(flare_energies)
        self.n = len(flare_energies)

    def loglike_powerlaw(self, params, e_uplim=np.inf):
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
        return self.loglike_powerlaw(params) + self.loglike_rate(params, e_uplim)

    def loglike_rate(self, params, e_uplim=np.inf):
        a, C = params
        lam = self.expt * C * (self.elim ** -a - e_uplim ** -a)
        # note log(gamma(n+1)) = log(n!)
        return -lam + self.n*np.log(lam) - gammaln(self.n+1)

    def loglike_powerindex(self, params, e_uplim=np.inf):
        if self.n == 0:
            return 0
        a, C = params
        a = a + 1 # want exponent for proper pdf not cumulative function for this
        n = self.n
        elim = self.elim
        loglike = n * np.log((a - 1) / (elim**(1-a) - e_uplim**(1-a))) - a * np.sum(np.log(self.e))

        # just a check, can be removed eventually TODO
        if e_uplim == np.inf:
            loglike2 = n * np.log((a - 1) / elim) - a * np.sum(np.log(self.e / elim))
            assert np.allclose(loglike, loglike2)
        return loglike


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