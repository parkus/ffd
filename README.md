#FFD
FFD (which stands for Flare Frequency Distribution) is a short piece of code to aid in fitting power-laws to FFDs. FFDs relate the frequency (i.e., occurrence rate) of flares to their energy (or peak flux, photometric equivalent width, etc.). In particular, this module was created to handle disparate datasets between which the flare detection limit varies. That is about the only mildly original aspect of FFD -- the rest is primarily convenience code.

The approach to the fitting is described in Appendix B of Loyd+ 2018 (The Astrophysical Journal, submitted). In essence, the number of flares detected is treated as following a Poisson distribution while the flare energies are treated as following a power law.

##  *BEWARE* An Important "Gotcha"
If there are no flares in your dataset but you are using a prior on a to constrain flare rates, be careful! In this case, C, and similar derived values like flare rates, will appear to have real lower and upper bounds in log-space, but this is not true! This is simply a consequence of the impossibility of any finite number of MCMC samples "filling out" a posterior distribution that maintains a constant value to -infinity in log space. When there are zero flares, don't to be fooled into thinking you can place lower limits on C and corresponding flare rates. Use the MCMC sampling only to specify an upper limit on these values. Specifically, do not trust the result of ffd.utils.error_bars when applied to log10(C), etc. Instead, do, e.g., np.percentile(C, 95) to find an upper limit.

I reason through this as follows. With one flare in the dataset, both a lower and upper bound on the rate constant exist. I.e., the probability of C = 0 is 0 and the probability of C = infinity is 0, and by experimenting I have found that in-between the MCMC sampler can find a distribution in log10(C) that is fairly normal. With no flares, the probability of a given C value converges to a constant-nonzero value as C -> 0 and log10(C) -> -infinity. This comes straight from the Poisson distribution. If the expected event rate is zero (i.e. C = 0), the probability of observing zero events is 1. The resulting distribution in the rate constant becomes simply the exponential distribution, for which the max-likelihood value is 0 and it only makes sense to specify an upper limit on C (or log10(C) if preferred).

## Quick Start
```
import ffd
import numpy as np

# -----------
# we will need a power-law random number generator for this
# I find the numpy and scipy powerlaw random number generators confusing,
# so let's make our own (but you can treat this as a black box :)
def power_rv(min, max, cumulative_index, n):
    a = cumulative_index
    norm = min**-a - max**-a
    # the cdf is 1 - ((x**-a - max**-a)/norm)
    x_from_cdf = lambda c: ((1-c)*norm + max**-a)**(-1/a)
    x_uniform = np.random.uniform(size=n)
    return x_from_cdf(x_uniform)


# -----------
# Example 1: Constrain the power-law fit parameters for the FFD of flares
# detected in a few different observations

# Say we are observing flares that have a true FFD described by the power law
# f = C * e**-a
# where f is the rate of flares with energy > e, C is a rate constant, and
# a is the power-law index.
a = 1.
C = 0.3

# now let's say we have a few observations, each with different exposure time
# and detection limit
expts = np.array([5, 20, 1, 2, 10, 0.5])
elims = np.array([1, 0.7, 2, 0.3, 1.5, 2])

# if you want, scale the exposure times to detect more/fewer flares to see
# how this changes the precision of the fit
expts = 1*expts

# simulate separate flare observations
observations = []
for expt, elim in zip(expts, elims):
  n_expected = expt * C * elim**-a  # expected number of flares
  n = np.random.poisson(n_expected) # simulated actual number detected
  e = power_rv(elim, np.inf, a, n)
  obs = ffd.Observation(elim, expt, e)
  observations.append(obs)

# have a look at how many flares were "detected" in each observation
# the more flares, the better we can expect to constrain a and C
# note that some of these will have no detected flares
print [obs.n for obs in observations]

# make a flare dataset
dataset = ffd.FlareDataset(observations)

# create a power-law fit object with 10 MCMC walkers and 1000 steps
fit = ffd.powerlaw.PowerLawFit(dataset, nwalkers=10, nsteps=1000)

# plot histograms of a and C to compare to the true value
plt.figure()
plt.semilogy(fit.a, fit.C, ',')
plt.plot(a, C, 'o')
plt.annotate('True Values', xy=(a,C))
plt.xlabel('a')
plt.ylabel('C')

plt.figure()
_ = plt.hist(fit.a, 100)
plt.axvline(a, color='g')
plt.xlabel('a')

plt.figure()
_ = plt.hist(np.log10(fit.C), 100)
plt.axvline(np.log10(C), color='g')
plt.xlabel('Log10(C)')

# get ML values and error bars of posterior distributions of a and C
# use log space with C because it is more normally distributed that way
error_bars(fit.a)
error_bars(fit.logC)

# -----------
# Example 2: Use a prior probability on the power-law index to constrain the
# flare rate from an observation with no detected flares

# say we know some star is similar to a group of stars that all have FFDs with
# a ~= 0.7 +/- 0.2. We can use this constraint to place an upper limit on the
# rate of flares above a given energy.

# make a prior for a that returns the log likelihood of a given a value
# (yes, I'm ignoring a constant offset)
def a_prior(a):
  return -(a - 0.7)**2/2/0.2**2

# set the exposure time. This will change the constraint on flare rate, since
# no flares in a longer amount of time means the rate must be lower
expt = 10.
elim = 1.

# do the fitting
obs = ffd.Observation(expt, elim, [])
dataset = ffd.FlareDataset([obs])
fit = ffd.powerlaw.PowerLawFit(dataset, a_prior=a_prior, nwalkers=10, nsteps=1000)

# use the MCMC samples from the fit to sample the rate of >elim flares
rate = fit.C * elim**-fit.a

# look at the distribution of rates and compute upper limit
# I'll define custom bins for the histogram to avoid really high values in the
# tail of the posterior
plt.figure()
bins = np.linspace(0, 1000./expt, 101)
_ = plt.hist(rate, bins)
plt.xlabel('Rate of >elim Flares')
np.percentile(rate, 95)
