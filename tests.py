import powerlaw
import ffd
from matplotlib import pyplot as plt
import numpy as np

def simple_ffd(a_true=1., emin=1., emax=100., n_events=10, n_trials=1000):
    """Determine if the correct value of a_true is estimated."""

    a = []
    for _ in range(n_trials):
        e = powerlaw.random_energies(a_true, emin, emax, n_events)
        obs = ffd.Observation(emin, 1.0, e)
        set = ffd.FlareDataset([obs])
        fit = powerlaw.PowerLawFit(set, nwalkers=10, nsteps=1)
        a.append(fit.a_best)

    plt.figure()
    plt.xlabel('Retrieved a')
    plt.hist(a, int(np.sqrt(n_trials)))
    plt.axvline(a_true)
    plt.annotate('True Value', xy=(a_true, 0.05), xytext=(2,0), xycoords=('data', 'axes fraction'),
                 textcoords='offset points', rotation=90, ha='left', va='bottom')
    print '(Median - True)/(error on mean) = {:.2f}'.format((np.median(a) - a_true)/(np.std(a)/np.sqrt(n_trials)))
    return a