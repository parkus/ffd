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