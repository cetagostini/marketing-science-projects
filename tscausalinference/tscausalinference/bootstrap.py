import pandas as pd
import numpy as np

from scipy.stats import norm, boxcox
import statsmodels.api as sm

from typing import Union

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def min_max_scale(data, min_range, max_range):
    """
    The function min_max_scale takes in an array of data, a minimum value for the range, and a maximum value for the range. It returns the data transformed into the specified range.

    Parameters:
    -----------
      data: array-like, shape (n_samples,)
      The data to be transformed.
      min_range: scalar or array-like, shape (n_features,)
      The minimum value of the range to transform the data.
      max_range: scalar or array-like, shape (n_features,)
      The maximum value of the range to transform the data.
    
    Returns:
    --------
      scaled_data: array-like, shape (n_samples,)
      The transformed data within the specified range.
    
    Examples:
    ---------
    >>> data = [1, 2, 3, 4, 5]
    >>> min_range = 0
    >>> max_range = 10
    >>> scaled_data = min_max_scale(data, min_range, max_range)
    >>> print(scaled_data)
    >>> Output: [0. 2. 4. 6. 10.]
    """

    data_min, data_max = data.min(), data.max()
    return (data - data_min) * (max_range - min_range) / (data_max - data_min) + min_range

def structural_bootstrap(data, num_bootstraps, block_length):
  """
  The function structural_bootstrap performs a structural bootstrap on a time series data. It takes in a time series data, the number of bootstraps to perform, and the block length for the decomposition. It returns an array of bootstrapped time series data.

  Parameters:
  -----------
    data: The time series data to perform the bootstrap on.
    num_bootstraps (int): The number of bootstraps to perform.
    block_length (int): The length of the blocks to decompose the data into.
  
  Returns:
  --------
    bootstrapped_data: An array of bootstrapped time series data.

  """
  new_series = data.copy()
  decomposition = sm.tsa.seasonal_decompose(new_series, period=block_length)

  trend = decomposition.trend
  seasonal = decomposition.seasonal
  remainder = decomposition.resid

  # Remove missing values in trend and remainder
  trend = trend[np.isfinite(trend)]
  remainder = remainder[np.isfinite(remainder)]

  n = len(trend)
  num_blocks = n - block_length + 1
  remainder_blocks = np.array([remainder[i:i+block_length] for i in range(num_blocks)])

  bootstrapped_data = np.empty((num_bootstraps, n))
  for i in range(num_bootstraps):
    bootstrap_indices = np.random.randint(num_blocks, size=(n // block_length) + 2)
    bootstrap_blocks = remainder_blocks[bootstrap_indices]

    bootstrap_remainder = np.concatenate(bootstrap_blocks)[:n]
    bootstrap_data = trend + seasonal[block_length // 2 : -(block_length // 2)] + bootstrap_remainder

    bootstrapped_data[i] = bootstrap_data

  return bootstrapped_data

def random_walk_bootstrap(bootstrap_samples, n_samples, n_steps, variable):
  """
  Performs a random walk bootstrap on the given variable using the bootstrap samples.

  Args:
      bootstrap_samples: array-like, shape (n_samples, n_steps)
          The bootstrap samples to use for resampling the data.
      n_samples: int
          The number of samples to generate.
      n_steps: int
          The number of steps to simulate for each sample.
      variable: array-like, shape (n,)
          The variable to simulate the random walk on.

  Returns:
      bootstrap_samples: array-like, shape (n_samples, n_steps)
          An array of bootstrapped time series data.
  """
  # Resample data with replacement
  bootstrap_data = np.random.choice(variable, size=(n_samples, n_steps))

  # Simulate random walk based on bootstrap data
  walk = np.cumsum(np.random.randn(n_samples, n_steps), axis=1)
  walk -= walk[:, 0][:, np.newaxis]

  std_ratio = bootstrap_data.std(axis=1)[:, np.newaxis] / walk.std(axis=1)[:, np.newaxis]
  walk *= std_ratio
  walk += bootstrap_data.mean(axis=1)[:, np.newaxis]

  # Smooth the simulated random walks using an exponential moving average filter
  smoother = 2  # the amount of smoothing on either side
  alpha = 1 / (smoother + 1)
  walk_smoothed = pd.DataFrame(walk).ewm(alpha=alpha).mean().values

  bootstrap_samples = walk_smoothed.copy()

  return bootstrap_samples

def prior_bootstrap(bootstrap_samples, n_samples, n_steps, variable, mape):
  """
  Performs a prior bootstrap on the given variable using the bootstrap samples.

  Args:
      bootstrap_samples: array-like, shape (n_samples, n_steps)
          The bootstrap samples to use for resampling the data.
      n_samples: int
          The number of samples to generate.
      n_steps: int
          The number of steps to simulate for each sample.
      variable: array-like, shape (n,)
          The variable to simulate the random walk on.
      mape: float
          The maximum percentage deviation allowed for the prior.

  Returns:
      bootstrap_samples: array-like, shape (n_samples, n_steps)
          An array of bootstrapped time series data.
  """
  min_range = variable.min() * (1 - mape)
  max_range = variable.max() * (mape + 1)

  variable = np.array(variable)

  # Resample data with replacement
  bootstrap_data = np.random.choice(variable, size=(n_samples, n_steps))

  # Simulate random walk based on bootstrap data
  walk = np.cumsum(np.random.normal(loc=0, scale=bootstrap_data.std(axis=1)[:, np.newaxis], size=(n_samples, n_steps)), axis=1)
  walk += bootstrap_data.mean(axis=1)[:, np.newaxis]

  walk = min_max_scale(walk, min_range, max_range)
  walk = variable - (np.mean(variable) - np.mean(walk, axis=1)[:, np.newaxis])

  bootstrap_samples = walk.copy()

  return bootstrap_samples

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
                    variable: Union[np.array, pd.DataFrame] = None, 
                    n_samples: int = 1500, 
                    n_steps: int = None,
                    mape: float = None,
                    prio = False,
                    method = 'BRW'):
    """
    Performs a bootstrap simulation on the given variable and returns an array of bootstrapped time series data.

    Args:
        variable: array-like, shape (n,)
            The variable to perform the bootstrap simulation on.
        n_samples: int, default=1500
            The number of bootstraps to generate.
        n_steps: int, default=None
            The number of steps to simulate for each sample.
        mape: float, default=None
            The maximum percentage deviation allowed for the prior.
        prio: bool, default=False
            Whether to use a prior bootstrap or not.
        method: {'BRW', 'SB'}, default='BRW'
            The method to use for the bootstrap simulation.

    Returns:
        bootstrap_samples: array-like, shape (n_samples, n_steps)
            An array of bootstrapped time series data.
    """
    # Initialize array to hold bootstrap samples
    bootstrap_samples = np.empty((n_samples, n_steps))

    if prio:
      if method == 'BRW':
        bootstrap_samples = prior_bootstrap(bootstrap_samples, n_samples, n_steps, variable, mape)
      elif method == 'SB':
        bootstrap_samples = structural_bootstrap(data = variable, num_bootstraps = n_samples, block_length = 7)
    else:
      bootstrap_samples = random_walk_bootstrap(bootstrap_samples, n_samples, n_steps, variable)
    
    return bootstrap_samples
    
def bootstrap_p_value(
                    control: Union[np.array, pd.DataFrame] = None, 
                    treatment: Union[np.array, pd.DataFrame] = None, 
                    simulations: np.array = None, 
                    alpha: float = 0.05
                    ):
    """
    Calculates the p-value of the difference between the means of the control and treatment groups using a bootstrap test.

    Args:
        control: array-like, shape (n,)
            The control group data.
        treatment: array-like, shape (m,)
            The treatment group data.
        simulations: array-like, shape (n_samples, n_steps)
            An array of bootstrapped time series data.
        alpha: float, default=0.05
            The level of significance for the hypothesis test.

    Returns:
        p_value: list of float
            The p-value of the hypothesis test.
        confidence_interval: list of float
            The confidence interval of the hypothesis test.
        bootstrapped_means: array-like, shape (n_samples,)
            An array of the means of the bootstrapped time series data.
        norm_simulations: array-like, shape (n_samples, n_steps)
            An array of the normalized bootstrapped time series data.

    Raises:
        ValueError: If the control and treatment groups have different lengths.
    """

    # Calculate the mean of the data
    mean = np.mean(control)
    mean_treatment = np.mean(treatment)
    mean_simulations = np.mean(simulations)
    diff = mean_simulations - mean
        
    bootstrapped_means = np.empty(len(simulations))
    norm_simulations = simulations.copy()
    
    for i in range(len(norm_simulations)):
        norm_simulations[i] -= diff
        bootstrapped_means[i] = norm_simulations[i].mean()
    
    lower, upper = np.percentile(bootstrapped_means, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    
    p_value = prob_in_distribution(bootstrapped_means, mean_treatment)

    return [p_value], [lower, upper], bootstrapped_means, norm_simulations