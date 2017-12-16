import numpy as _np
from math import ceil


def mode_halfsample(x, presorted=False):
    """
    Estimate the mode from which x was drawn using the halfsample method (HSM).

    Parameters
    ----------
    x : array-like
        The randomly sampled data for which to find the mode.
    presorted

    Returns
    -------
    mode : float
        Estimated mode of the distribution from which x was drawn.

    Notes
    -----
    Sometimes other ways of estimating the mode perform better, such as kernel density estimation, for specific
    forms of the pdf. However, this method is accurate, fairly robust and very fast.

    Examples
    --------
    x = np.random.exponential(size=100000)
    mode_halfsample(x)

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


def error_bars(x, interval=0.683, upper_limit=0.95):
    """
    Find the most likely value and error bars (confidence interval) of a parameter based on random samples (as from
    an MCMC parameter search). If a most likely value with a confidence interval is not well defined, return an upper
    or lower limit as appropriate.

    Parameters
    ----------
    x : array-like
        The randomly sampled data for which to find a most likely value and error bars.
    interval : float
        The width of the confidence interval (such as 0.683 for 1-sigma error bars).
    upper_limit : float
        The cumulative probability at which to set the upper limit.

    Returns
    -------
    x_mode, err_neg, err_pos : floats
        The most probable x value and the negative (as a negative value) and positive error bars. If searching for a
        confidence interval resulted in a limit, then np.nan is used for x_mode and err_neg (upper limit) or err_pos
        (lower limit) and the limit value is given in the remaining position.

    Examples
    --------
    import numpy as np
    x = np.random.normal(10., 2., size=100000)
    confidence_interval(x)
    # should return roughly 10, -2, 2

    x = np.random.gamma(1.0, 2.0, size=100000)
    confidence_interval(x)
    # should return an upper limit of roughly 6 as nan, nan, 6

    """


    x = _np.sort(x)
    x_mode = mode_halfsample(x, presorted=True)

    # compute the cumulative integral based on the sorted samples
    # cdf will be a step function, jumping up at each x
    cdf = _np.arange(len(x), dtype='f8')/len(x)

    # use cdf to define an inverse cdf function
    def inverse_cdf(c_value):
        i = _np.searchsorted(cdf, c_value)
        if i == 0 or i == len(cdf):
            raise OutOfRange('CDF value beyond sampled range.')
        else:
            return x[i-1]

    # try to find each end of the confidence interval
    # if the mode of the distribution is too far from the median such that the confidence interval cannot be reached,
    # then treat as an upper or lower limit
    i_mode = _np.searchsorted(x, x_mode)
    c_mode = (cdf[i_mode-1] + cdf[i_mode])/2.0

    # try lower interval
    try:
        x_min = inverse_cdf(c_mode - interval/2.)
    except OutOfRange:
        x_lim = inverse_cdf(upper_limit)
        return _np.nan, _np.nan, x_lim

    # try upper interval
    try:
        x_max = inverse_cdf(c_mode + interval/2.)
    except OutOfRange:
        x_lim = inverse_cdf(1 - upper_limit)
        return _np.nan, x_lim, _np.nan

    # if neither encountered a limit return the interval
    return x_mode, x_min - x_mode, x_max - x_mode
