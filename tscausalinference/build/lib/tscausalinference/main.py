from typing import List, Union

from pandas import DataFrame
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tabulate import tabulate

from tscausalinference.synth_regression import synth_analysis
from tscausalinference.bootstrap import bootstrap_simulate, bootstrap_p_value
from tscausalinference.load_synth_data import create_synth_dataframe
from tscausalinference.sensitivity_regression import sensitivity_analysis
from tscausalinference.plots import plot_intervention, plot_simulations, seasonal_decompose
from tscausalinference.summaries import summary, summary_intervention

sns.set_theme()
sns.set_context("paper")

pd.options.mode.chained_assignment = None 

class tscausalinference:
    """
    Performs time-series causal inference analysis using synthetic control and bootstrap simulations.

    Args:
        data: A numpy array or pandas DataFrame containing the time-series data.
        intervention: A list of integers, strings or pandas Timestamps representing the start and end dates of the intervention period.
        regressors: A list of strings representing the names of columns in the data that are used as regressors in the model.
        alpha: A float representing the significance level for hypothesis testing.
        seasonality: A boolean indicating whether to include seasonality in the model.
        n_samples: An integer representing the number of bootstrap samples to simulate.
        cross_validation_steps: An integer representing the number of cross-validation steps to use in the synthetic control analysis.
        model_params: A dictionary containing additional parameters to pass to the model.

    Attributes:
        data: A pandas DataFrame containing the pre-intervention and post-intervention time-series data, as well as the predicted values and confidence intervals.
        pre_int_metrics: A dictionary containing the pre-intervention metrics calculated during the synthetic control analysis.
        int_metrics: A dictionary containing the intervention metrics calculated during the synthetic control analysis.
        string_filter: A string representing the filter used to select the intervention period in the data.
        simulations: A numpy array containing the bootstrap simulations.
        stadisticts: A dictionary containing the test statistics and p-values for the intervention effect.
        stats_ranges: A dictionary containing the confidence intervals for the test statistics.
        samples_means: A dictionary containing the mean values for the control and treatment samples.

    Methods:
        plot_intervention(past_window=5, back_window=25, figsize=(15, 10)):
            Plots the pre-intervention and post-intervention time-series data, as well as the predicted values and confidence intervals.
    
    Example:
        >>> from tscausalinference import tscausalinference as tsci
        >>> import pandas as pd
        >>> # Load data
        >>> df = pd.read_csv('mydata.csv')
        >>> intervention = ['2022-07-04', '2022-07-19']
        >>> data = tsci(data = df, intervention = intervention)
        >>> data.plot_intervention() 
    """

    def __init__(self,
        data: Union[np.array, DataFrame],
        intervention: Union[List[int], List[str], List[pd.Timestamp]],
        regressors: list = [],
        alpha: float = 0.05,
        seasonality: bool = True,
        n_samples: int = 1500,
        cross_validation_steps: int = 5,
        model_params: dict = {}
        ):

        self.data = data
        self.intervention = intervention
        self.alpha = alpha
        self.seasonality = seasonality
        self.regressors = regressors
        self.n_samples = n_samples
        self.cross_validation_steps = cross_validation_steps
        self.model_params = model_params

        self.data, self.pre_int_metrics, self.int_metrics = synth_analysis(
            df = data, 
            regressors = regressors, 
            intervention = intervention, 
            cross_validation_steps = cross_validation_steps,
            alpha = alpha,
            model_params = model_params
            )
        self.string_filter = "ds >= '{}' & ds <= '{}'".format(intervention[0], intervention[1])
        
        self.simulations = bootstrap_simulate(
                data = self.data.query(self.string_filter).yhat, 
                n_samples = n_samples, 
                n_steps = len(self.data.query(self.string_filter).index)
                )
        
        self.stadisticts, self.stats_ranges, self.samples_means = bootstrap_p_value(control = self.data.query(self.string_filter).yhat, 
                                                                                    treatment = self.data.query(self.string_filter).y, 
                                                                                    simulations = self.simulations,
                                                                                    mape = abs(round(self.pre_int_metrics[2][1],6))/100
                                                                                    )
 
    def plot(self, 
              method: str = 'intervention',
              past_window: int = 5, 
              back_window: int = 25, 
              figsize: tuple = (25, 10),
              simulation_number: int = 10):
        """
        """
        if method not in ['intervention','simulations','decomposition']:
            error = "Your method should be defined as one of these -> ('intervention','simulations','decomposition') "
            raise TypeError(error)
        
        if method == 'intervention':
            plot_intervention(data = self.data, past_window = past_window, back_window = back_window, figsize = figsize, intervention = self.intervention)
        elif method == 'simulations':
            plot_simulations(data = self.data, past_window = past_window, back_window = back_window, figsize = figsize, simulation_number = simulation_number,
                             intervention = self.intervention, simulations = self.simulations, stadisticts = self.stadisticts, 
                             stats_ranges = self.stats_ranges, samples_means = self.samples_means)
        elif method == 'decomposition':
            seasonal_decompose(data = self.data, intervention = self.intervention, figsize = figsize)
        
    
    def summarization(self, statistical_significance = 0.05, method = 'general'):
        """
        """
        if method not in ['general','detailed']:
            error = "Your method should be defined as one of these -> ('general','detailed') "
            raise TypeError(error)

        if method == 'general':
            summary(data = self.data, statistical_significance = statistical_significance, 
            stadisticts = self.stadisticts, pre_int_metrics = self.pre_int_metrics, 
            int_metrics = self.int_metrics, intervention = self.intervention, n_samples = self.n_samples)
        elif method == 'detailed':
            summary_intervention(data = self.data, intervention = self.intervention, int_metrics = self.int_metrics) 

class synth_dataframe:
    """
    Creates a synthetic dataframe with time series components.

    Args:
    - n (int): Number of periods to generate.
    - trend (float): Slope of the trend component.
    - seasonality (int): Period of the seasonality component.
    - simulated_effect (float): Multiplicative effect of the treatment.
    - eff_n (int): Number of periods the treatment lasts.
    - noise_power (float): Scale of the noise component.
    - regressor (int): Number of regressors to include in the dataframe.

    Methods:
    - DataFrame: Returns a pandas dataframe with columns 'ds', 'y' and the regressors created.

    Example:
    >>> synth = synth_dataframe(n=365, trend=0.1, seasonality=7, simulated_effect=1.2, eff_n=30, noise_power=0.1, regressor=3)
    >>> df = synth.DataFrame()
    """

    def __init__(self,
                n: int = 365, 
                trend: float = 0.1, 
                seasonality: int = 7, 
                simulated_effect: float = 0.15, 
                eff_n: int = 15, 
                noise_power: float = 0.15, 
                regressors: int = 2):
        self.n = n
        self.seasonality = seasonality
        self.trend = trend
        self.simulated_effect = simulated_effect
        self.eff_n = eff_n
        self.noise_power = noise_power
        self.regressors = regressors

        self.df = create_synth_dataframe(n = self.n, 
                           trend = self.trend, 
                           seasonality = self.seasonality, 
                           simulated_effect = self.simulated_effect, 
                           eff_n = self.eff_n, 
                           noise_power = self.noise_power, 
                           regressor = self.regressors)
    
    def DataFrame(self):
        return self.df

class sensitivity:
    def __init__(self,
                df: DataFrame = pd.DataFrame(), 
                training_period = None, 
                cross_validation_steps: int = 5, 
                alpha: float = 0.05, 
                model_params: dict = {}, 
                regressors: list = []):
    
        self.df = df
        self.training_period = training_period
        self.cross_validation_steps = cross_validation_steps
        self.alpha = alpha
        self.model_params = model_params
        self.regressors = regressors

        self.data, self.training, self.test = sensitivity_analysis(
            df = df, 
            regressors = regressors, 
            training_period = training_period, 
            cross_validation_steps = cross_validation_steps,
            alpha = alpha,
            model_params = model_params
            )