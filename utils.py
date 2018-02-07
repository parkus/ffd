import numpy as _np
from math import ceil, floor

def mode_halfsample(x, presorted=False):
    """
    Estimate the mode from which x was drawn using the halfsample method (HSM).

    Parameters
    ----------
    x : array-like
        The randomly sampled data for which to find the mode.
    presorted : boolean
        True if the data are already sorted (can save time if function is looped). Default is False.

    Returns
    -------
    mode : float
        Estimated mode of the distribution from which x was drawn.

    Notes
    -----
    Sometimes other ways of estimating the mode perform better, such as kernel density estimation, for specific
    forms of the pdf. However, this method is accurate, fairly robust, and very fast.

    Examples
    --------
    x = np.random.exponential(size=100000)
    mode_halfsample(x)
    # should return a value near 0

    References
    ----------
    Bickel & Fruhwirth 2005 https://arxiv.org/abs/math/0505419
    Robertson & Cryer 1974 Journal of the American Statistical Association 69:1012

    """
    if not presorted:
        x = _np.sort(x)
    n = len(x)
    while n > 2:
        # roughly half the number of samples
        h = int(ceil(len(x)/2))

        # find the half-sample with the shortest range
        ranges = x[h:] - x[:-h]
        i = _np.argmin(ranges)

        # make that the new data vector and do the whole thing over again
        x = x[i:i+h]
        n = len(x)
    return _np.mean(x)


class OutOfRange(Exception):
    pass


def error_bars(x, x_ml=None, interval=0.683, limit=0.95):
    """
    Find the most likely value and error bars (confidence interval) of a parameter based on random samples (as from
    an MCMC parameter search). If a most likely value with a confidence interval is not well defined, return an upper
    or lower limit as appropriate.

    Parameters
    ----------
    x : array-like
        The randomly sampled data for which to find a most likely value and error bars.
    x_ml : float
        Max-likelihood value of x. Useful when this value was found with, say, scipy.optimize.minimize and now you
        just want to get the error bars to either side of that value, even though the MCMC sampling might show a
        slightly different peak.
    interval : float
        The width of the confidence interval (such as 0.683 for 1-sigma error bars).
    limit : float
        The cumulative probability at which to set an upper or lower limit as necessary.

    Returns
    -------
    x_mode, err_neg, err_pos : floats
        The most probable x value and the negative (as a negative value) and positive error bars. If searching for a
        confidence interval resulted in a limit, then np.nan is used for x_mode and err_neg (upper limit) or err_pos
        (lower limit) and the limit value is given in the remaining position.

    Examples
    --------
    import numpy as np
    from matplotlib import pyplot as plt
    x = np.random.normal(10., 2., size=100000)
    _ = plt.hist(x, 200) # see what the PDF it looks like
    error_bars(x)
    # should return roughly 10, -2, 2

    x = np.random.gamma(1.0, 2.0, size=100000)
    _ = plt.hist(x, 200) # see what the PDF looks like
    error_bars(x)
    # should return an upper limit of roughly 6 as nan, nan, 6

    Notes
    -----
    The confidence interval is taken to be the central chunk of the PDF corresponding to the value set by `interval`.
    Error bars are then defined by the difference from the lower and upper limits of this chunk to the max-likelihood
    value. If the max-likelihood value is outside of these limits, then the PDF is interpreted as only giving an upper
    or lower limit as appropriate, and that limiting value is computed and returned instead.

    """

    if _np.any(_np.isnan(x)):
        raise ValueError('There are NaNs in the samples.')

    x = _np.sort(x)
    if x_ml is None:
        x_ml = mode_halfsample(x, presorted=True)

    interval_pcntl = 100 * _np.array([(1-interval)/2, (1+interval)/2])
    x_min, x_max = _np.percentile(x, interval_pcntl)

    if x_ml < x_min: # upper limit
        return [_np.nan, _np.nan, _np.percentile(x, 100*limit)]

    if x_ml > x_max: # lower limit
        return [_np.nan, _np.percentile(x, 100*(1-limit)), _np.nan]

    # interval is good
    return x_ml, x_min - x_ml, x_max - x_ml