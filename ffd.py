import numpy as np
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
        cumexpts = np.cumsum(expts)
        i_lims = np.searchsorted(elims, self.e, side='right')
        self.expt_detectable = cumexpts[i_lims-1]

        # cumulative frequencies ignoring differences in detection limits and correcting for them
        cumno = np.arange(self.n_total)[::-1] + 1
        self.cumfreq_naive = (cumno / self.expt_total)
        cumno_corrected = np.cumsum(1. / self.expt_detectable[::-1])[::-1] * self.expt_total
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
        line, = ax.step(self.e, cf, where='post', **kwargs)
        return line

    def loglike_powerlaw(self, params, e_uplim):
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
        return np.sum(data.loglike_powerlaw(params, e_uplim) for data in self.datasets)

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
        self.elim = detection_limit
        self.expt = exposure_time
        self.e = np.array(flare_energies)
        self.n = len(flare_energies)

    def loglike_powerlaw(self, params, e_uplim):
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
        if a <= 0 or C < 0:
            return -np.inf
        if C == 0:
            if self.n > 0:
                return -np.inf
            else:
                return 0.0

        a = a + 1  # supplying cumulative but want regular exponent
        lam = self.expt * C * (self.elim ** (1 - a) - e_uplim ** (1 - a))
        if self.n == 0:
            return -lam
        else:
            n = self.n
            log = np.log
            elim = self.elim
            # note log(gamma(n+1)) = log(n!)
            return -lam + n*log(lam) - gammaln(n+1) + n*log((a-1)/elim) - a*np.sum(log(self.e/elim))







