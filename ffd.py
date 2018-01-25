import numpy as np
from matplotlib import pyplot as plt

class FlareDataset(object):
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

    def __init__(self, observations):
        """
        Creat a Flares object.

        Parameters
        ----------
        observations : list
            A list of FlareDataset objects. Make sure they use a consistent metric to characterize the flares (
            energy, equivalent duration, peak flux, etc.)

        Returns
        -------
        A Flares object :)

        """
        self.observations = observations

        # totals
        self.n_total = sum([data.n for data in observations])
        self.expt_total = np.sum([data.expt for data in observations])

        # event energies concatenated from all flare datasets
        self.e = np.concatenate([data.e for data in observations])
        self.e = np.sort(self.e)

        # exposure time in which an event of energy e *could* have been detected
        expts = np.array([data.expt for data in observations])
        elims = np.array([data.elim for data in observations])
        isort = np.argsort(elims)
        elims, expts = [a[isort] for a in [elims, expts]]
        self.expt_at_lim = np.cumsum(expts)
        self.elims, self.expts = elims, expts

        if self.n_total > 0:
            emax = max(np.max(elims), np.max(self.e))
            count, _ = np.histogram(self.e, np.append(elims, emax+1))
            self.n_detected = np.cumsum(count[::-1])[::-1]

            # cumulative frequencies ignoring differences in detection limits and correcting for them
            cumno = np.arange(self.n_total)[::-1] + 1
            self.cumfreq_naive = (cumno / self.expt_total)
            i_lims = np.searchsorted(elims, self.e, side='right')
            expt_detectable = self.expt_at_lim[i_lims - 1]
            cumno_corrected = np.cumsum(1. / expt_detectable[::-1])[::-1] * self.expt_total
            self.cumfreq_corrected = cumno_corrected / self.expt_total
        else:
            self.n_detected = np.array([0] * len(observations))
            self.cumfreq_naive = np.array([0] * len(observations))
            self.cumfreq_corrected = np.array([0] * len(observations))

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
        scale : 1.0
            Scale to apply to cumulative frequency (so you can use different units, for example).
        kwargs :
            passed to the matplotlib plot function

        Returns
        -------
        line :
            matplotlib line object
        """
        ax = kwargs.pop('ax', plt.gca())
        corrected = kwargs.pop('corrected', True)
        scale = kwargs.pop('scale', 1.0)
        cf = self.cumfreq_corrected if corrected else self.cumfreq_naive
        cf = cf*scale
        line, = ax.step(self.e, cf, where='pre', **kwargs)
        return line


class Observation(object):

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







