import pandas as pd
import numpy as np

from scipy.stats import t
from scipy.stats import norm

from typing import Union

def prob_in_distribution(data, x):
  """
    Calculate the probability of a given value being in a distribution defined by the given data.

    Args:
    - data: a list or array-like object containing the data to define the distribution
    - x: a numeric value for which to calculate the probability of being in the distribution

    Returns:
    - prob: a numeric value representing the probability of x being in the distribution

    Notes:
    - This function assumes that the data follows a normal distribution.
    - The probability is calculated as a proportion of the area under the normal curve between the minimum and
      maximum values of the data.
    - If x is outside of the range of the data, the probability is 0.0.
    - If x is exactly at the mean of the data, the probability is 0.5.
    - If x is on one side of the mean, the probability is proportional to the area of the normal curve on that side.
  """
  lower_bound, upper_bound = min(data), max(data)
  
  mean, std = np.mean(data), np.std(data)

  cdf_lower = norm.cdf(lower_bound, mean, std)
  cdf_upper = 1 - norm.cdf(upper_bound, mean, std)

  if x < lower_bound or x > upper_bound:
      return 0.0
  else:
      cdf_x = norm.cdf(x, mean, std)
      if cdf_x <= 0.5:
          return 2 * (cdf_x - cdf_lower) / (1 - cdf_lower - cdf_upper)
      else:
          return 2 * (1 - cdf_x + cdf_lower) / (1 - cdf_lower - cdf_upper)

def bootstrap_simulate(
                    data: Union[np.array, pd.DataFrame] = None, 
                    n_samples: int = 1500, 
                    n_steps: int = None):
    """
    Generate an array of bootstrap samples for a given dataset.

    Args:
    - data: a numpy array or pandas dataframe containing the data to be resampled
    - n_samples: an integer representing the number of bootstrap samples to generate
    - n_steps: an integer representing the number of steps in the random walk simulation; if None, defaults to the
      length of the input data

    Returns:
    - bootstrap_samples: a numpy array of shape (n_samples, n_steps) containing the bootstrap samples

    Notes:
    - This function uses the bootstrap method to resample the input data with replacement.
    - For each bootstrap sample, a random walk is simulated based on the resampled data.
    - The random walk starts at the mean of the resampled data, and each step is a random draw from a standard normal
      distribution.
    - The variance of the random walk is adjusted to match the variance of the resampled data.
    - The resulting random walk is saved as one of the bootstrap samples.
    - The function returns an array of shape (n_samples, n_steps), where each row represents a bootstrap sample.

    Example Usage:
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> bootstrap_simulate(data, n_samples=1000, n_steps=10)
    """
    # Initialize array to hold bootstrap samples
    bootstrap_samples = np.empty((n_samples, n_steps))
    
    # Loop over number of bootstrap samples
    for i in range(n_samples):
        # Resample data with replacement
        bootstrap_data = np.random.choice(data, size=len(data))#, replace=True)
        
        # Simulate random walk based on bootstrap data
        walk = np.cumsum(np.random.randn(n_steps))
        walk -= walk[0]
        walk *= bootstrap_data.std() / walk.std()
        walk += bootstrap_data.mean()

        # Smooth the simulated random walk using a moving average filter
        smoother = 2  # the amount of smoothing on either side
        
        # Pad the beginning and end of the input array
        pad_size = smoother
        padded_walk = np.pad(walk, (pad_size, pad_size), mode='edge')
        
        # Apply the smoothing filter
        walk_smoothed = np.convolve(padded_walk, np.ones(2*smoother+1)/(2*smoother+1), mode='valid')

        # Save random walk as one of the bootstrap samples
        bootstrap_samples[i] = walk_smoothed
    
    return bootstrap_samples
    
def bootstrap_p_value(
                    control: Union[np.array, pd.DataFrame] = None, 
                    treatment: Union[np.array, pd.DataFrame] = None, 
                    simulations: np.array = None, 
                    alpha: float = 0.05,
                    mape: float = None):
    """
    Calculate the p-value for a difference in means between 
    a control group and a treatment group using the bootstrap method.

    Args:
    - control: a numpy array or pandas dataframe containing data from the control group
    - treatment: a numpy array or pandas dataframe containing data from the treatment group
    - simulations: a numpy array of bootstrap samples generated using the `bootstrap_simulate` function
    - alpha: a float representing the significance level for the hypothesis test; defaults to 0.05
    - mape: a float representing the maximum allowable percent error for the bootstrapped means; if None, no error
      adjustment is made

    Returns:
    - p_value: a list containing a single float representing the calculated p-value
    - confidence_interval: a list containing two floats representing the lower and upper bounds of the confidence
      interval for the difference in means
    - bootstrapped_means: a numpy array containing the bootstrapped means used to calculate the p-value and confidence
      interval

    Notes:
    - This function assumes that the data in both control and treatment groups are normally distributed.
    - The p-value is calculated using a two-sided hypothesis test.
    - The null hypothesis is that the means of the two groups are equal.
    - The alternative hypothesis is that the means of the two groups are not equal.
    - The p-value is calculated as the proportion of bootstrapped means that are more extreme than the observed
      difference in means between the control and treatment groups.
    - A confidence interval is also calculated for the difference in means, based on the bootstrapped means.
    - If the `mape` argument is provided, the bootstrapped means are adjusted to account for potential error in the
      bootstrap sampling process.
    - The `prob_in_distribution` function is used to calculate the p-value based on the bootstrapped means.

    Example Usage:
    >>> control_data = np.array([1, 2, 3, 4, 5])
    >>> treatment_data = np.array([2, 3, 4, 5, 6])
    >>> simulations = bootstrap_simulate(control_data, n_samples=1000, n_steps=10)
    >>> bootstrap_p_value(control_data, treatment_data, simulations, alpha=0.1, mape=0.1)
    """

    # Calculate the mean of the data
    mean = np.mean(control)
    mean_treatment = np.mean(treatment)
        
    bootstrapped_means = np.empty(len(simulations))
    
    for i in range(len(simulations)):
        bootstrapped_means[i] = simulations[i].mean()
    
    bootstrapped_means_min = bootstrapped_means - (mean * mape)
    bootstrapped_means_max = bootstrapped_means + (mean * mape)
    bootstrapped_means = np.concatenate([bootstrapped_means_min, bootstrapped_means, bootstrapped_means_max])
    
    lower, upper = np.percentile(bootstrapped_means, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    
    p_value = prob_in_distribution(bootstrapped_means, mean_treatment)

    return [p_value], [lower, upper], bootstrapped_means