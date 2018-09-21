import numpy as _np

class OutOfRange(Exception):
    pass


def error_bars(x, x_best=None, interval=0.683):
    """
    Get the error bars from an MCMC chain of values, x.

    Parameters
    ----------
    x : array-like
        The randomly sampled data for which to find a most
        likely value and error bars.
    x_best : float
        Value to use as the "best" value of x. If not specified,
        the median is used.
    interval : float
        The width of the confidence interval (such as 0.683 for
        1-sigma error bars).

    Returns
    -------
    x_best, err_neg, err_pos : floats
        The "best" x value (median or user specified) and the
        negative (as a negative value) and positive error bars.

    Examples
    --------
    import numpy as np
    from matplotlib import pyplot as plt
    x = np.random.normal(10., 2., size=100000)
    _ = plt.hist(x, 200) # see what the PDF it looks like
    error_bars(x)
    # should return roughly 10, -2, 2

    """

    if _np.any(_np.isnan(x)):
        raise ValueError('There are NaNs in the samples.')

    if x_best is None:
        x_best = np.median(x)

    interval_pcntl = 100 * _np.array([(1-interval)/2, (1+interval)/2])
    x_min, x_max = _np.percentile(x, interval_pcntl)

    return x_best, x_min - x_best, x_max - x_best


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