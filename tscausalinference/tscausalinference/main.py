from typing import List, Union

from pandas import DataFrame
import pandas as pd

import numpy as np
import seaborn as sns

from tscausalinference.synth_regression import synth_analysis
from tscausalinference.bootstrap import bootstrap_simulate, bootstrap_p_value
from tscausalinference.load_synth_data import create_synth_dataframe
from tscausalinference.sensitivity_regression import sensitivity_analysis
from tscausalinference.plots import plot_intervention, plot_simulations, seasonal_decompose, sensitivity_curve, plot_training, plot_diagnostic
from tscausalinference.summaries import summary, summary_intervention, summary_incrementality
from tscausalinference.evaluators import mde_area

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sns.set_theme()
sns.set_context("paper")

pd.options.mode.chained_assignment = None 

class tscausalinference:
    """
    A class to perform time series causal inference using Synthetic Controls Methodology.

    Parameters:
    data (np.array or DataFrame): The time series data.
    intervention (list): The intervention period in the format ['start_date', 'end_date'].
    regressors (list, optional): A list of column names to be used as regressors in the model. Defaults to empty list.
    alpha (float, optional): The level of significance. Defaults to 0.05.
    seasonality (bool, optional): Whether or not to include seasonality in the model. Defaults to True.
    n_samples (int, optional): Number of bootstrapping samples. Defaults to 1500.
    cross_validation_steps (int, optional): Number of cross-validation steps. Defaults to 5.
    model_params (dict, optional): Additional parameters for the Prophet model. Defaults to empty dictionary.

    Returns:
        None
    """

    def __init__(self,
        data: Union[np.array, DataFrame],
        intervention: Union[List[int], List[str], List[pd.Timestamp]],
        regressors: list = [],
        alpha: float = 0.05,
        seasonality: bool = True,
        n_samples: int = 1500,
        cross_validation_steps: int = 5,
        model_params: dict = {},
        model_type = 'gam',
        autocorrelation = False
        ):

        self.data = data
        self.intervention = intervention
        self.alpha = alpha
        self.seasonality = seasonality
        self.regressors = regressors
        self.n_samples = n_samples
        self.cross_validation_steps = cross_validation_steps
        self.model_params = model_params
        self.model_type = model_type
        self.autocorrelation = autocorrelation
        
        self.string_filter = "ds >= '{}' & ds <= '{}'".format(intervention[0], intervention[1])
        
    def run(self, prior = False, method = 'BRW'):
        """
        Runs the causal analysis with the specified configuration.

        Args:
            prior (bool, optional): Whether to use a priori information from seasonality and trend in the analysis. Defaults to False.
            method (string, optional): Available only when prior is TRUE, Bootstrap Random Walk (BRW) or Structural Bootstrap (BS)

        Returns:
            self: The tscausalinference object, updated with the results of the causal analysis.      
        """
        self.data, self.pre_int_metrics, self.int_metrics, self.hyper_parameters = synth_analysis(
            df = self.data, 
            regressors = self.regressors, 
            intervention = self.intervention, 
            cross_validation_steps = self.cross_validation_steps,
            alpha = self.alpha,
            model_params = self.model_params,
            model_type = self.model_type,
            autocorrelation = self.autocorrelation
            )
    
        self.simulations = bootstrap_simulate(
            variable = self.data.query(self.string_filter).yhat, 
            n_samples = self.n_samples, 
            n_steps = len(self.data.query(self.string_filter).index),
            mape = abs(round(self.pre_int_metrics[2][1],6)) / 100,
            prio = prior,
            method = method
            )
        
        self.stadisticts, self.stats_ranges, self.samples_means, self.norm_simulations = bootstrap_p_value(control = self.data.query(self.string_filter).yhat, 
                                                                                    treatment = self.data.query(self.string_filter).y, 
                                                                                    simulations = self.simulations
                                                                                    )

        return self
    
    def plot(self, 
              method: str = 'intervention',
              past_window: int = 5, 
              back_window: int = 25, 
              figsize: tuple = (25, 10),
              simulation_number: int = 10):
        """
        Plots the time series data.

        Parameters:
            method (str, optional): The method used to plot the data. Available options are 'intervention', 'simulations', and 'decomposition'. Defaults to 'intervention'.
            past_window (int, optional): Number of past periods to include in the plot. Defaults to 5.
            back_window (int, optional): Number of future periods to include in the plot. Defaults to 25.
            figsize (tuple, optional): The figure size. Defaults to (25, 10).
            simulation_number (int, optional): The number of simulations to include in the plot if the method is 'simulations'. Defaults to 10.

        Returns:
            None
        """
        if method not in ['intervention','simulations','decomposition']:
            error = "Your method should be defined as one of these -> ('intervention','simulations','decomposition') "
            raise TypeError(error)
        
        if method == 'intervention':
            plot_intervention(data = self.data, past_window = past_window, back_window = back_window, figsize = figsize, intervention = self.intervention)
        elif method == 'simulations':
            plot_simulations(data = self.data, past_window = past_window, back_window = back_window, figsize = figsize, simulation_number = simulation_number,
                             intervention = self.intervention, simulations = self.norm_simulations, stadisticts = self.stadisticts, 
                             stats_ranges = self.stats_ranges, samples_means = self.samples_means)
        elif method == 'decomposition':
            seasonal_decompose(data = self.data, intervention = self.intervention, figsize = figsize)
    
    def model_parameters(self):
        return self.hyper_parameters
    
    def summarization(self,  method = 'general', statistical_significance = 0.05, interrupted_variable = None, window = 30):
        """
        Generates a summary report for the time series causal inference analysis.

        Parameters:
            statistical_significance (float, optional): The level of significance for testing statistical significance. Defaults to 0.05.
            method (str, optional): The method used to generate the summary report. Available options are 'general' and 'detailed'. Defaults to 'general'.

        Returns:
            None
        """
        if method not in ['general','detailed','incremental']:
            error = "Your method should be defined as one of these -> ('general','detailed','incremental') "
            raise TypeError(error)

        if method == 'general':
            summary(data = self.data, statistical_significance = statistical_significance, 
            stadisticts = self.stadisticts, pre_int_metrics = self.pre_int_metrics, 
            int_metrics = self.int_metrics, intervention = self.intervention, n_samples = self.n_samples, ci_int = self.stats_ranges)
        elif method == 'detailed':
            summary_intervention(data = self.data, intervention = self.intervention, 
                                 int_metrics = self.int_metrics, stadisticts = self.stats_ranges)
        elif method == 'incremental':
            summary_incrementality(variable = interrupted_variable, intervention = self.intervention, int_metrics = self.int_metrics, window = window) 

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
    """
    The `sensitivity` class helps analyze the sensitivity of a model by performing
    a sensitivity analysis with a given dataset and model type. It supports cross-validation,
    hyperparameter optimization, and visualization of results.

    Attributes:
        df (pd.DataFrame): The input dataset.
        test_period: The period used for testing.
        cross_validation_steps (int): The number of cross-validation steps.
        alpha (float): The significance level for hypothesis testing.
        model_params (dict): Model parameters to be used in the analysis.
        regressors (list): The list of regressors to be used in the analysis.
        verbose (bool): Whether to print log messages or not.
        n_samples (int): Number of samples to be used in the analysis.
        autocorrelation (bool): Whether to consider autocorrelation in the analysis.
        model_type (str): The type of model to use in the analysis GAM or Bayesian ('gam' is the default).

    Example:
        >>> import pandas as pd
        >>> data = pd.read_csv("data.csv")
        >>> sens = sensitivity(df=data, test_period=10, alpha=0.05)
        >>> sens.run()
        >>> sens.plot(method="sensitivity")    
    """
    def __init__(self,
                df: DataFrame = pd.DataFrame(), 
                test_period = None, 
                cross_validation_steps: int = 5, 
                alpha: float = 0.05, 
                model_params: dict = {}, 
                regressors: list = [],
                verbose: bool = False,
                n_samples = 1000,
                autocorrelation = False,
                model_type='gam'):

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Parameter `df` must be an instance of pandas.DataFrame")
        
        if not isinstance(test_period, list):
            raise ValueError("Parameter `test_period` must be a list")

        self.df = df
        self.test_period = test_period
        self.cross_validation_steps = cross_validation_steps
        self.alpha = alpha
        self.model_params = model_params
        self.regressors = regressors
        self.model_type = model_type
        self.autocorrelation = autocorrelation
        self.verbose = verbose
        self.n_samples = n_samples

    def run(self, prior = False, method = 'BRW'):
        """
        Runs the sensitivity analysis with the specified configuration.

        Args:
            prio (bool, optional): Whether to use a priori information from seasonality and trend in the analysis. Defaults to False.

        Returns:
            self: The sensitivity object, updated with the results of the sensitivity analysis.

        Example:
            >>> sens = sensitivity(df=data, test_period=10, alpha=0.05)
            >>> sens.run()        
        """
        self.data, self.analysis, self.hyper_parameters = sensitivity_analysis(df = self.df, 
                         test_period = self.test_period, 
                         cross_validation_steps = self.cross_validation_steps, 
                         alpha = self.alpha, 
                         model_params = self.model_params, 
                         regressors= self.regressors,
                         verbose = self.verbose,
                         n_samples = self.n_samples,
                         prio = prior,
                         autocorrelation = self.autocorrelation,
                         model_type = self.model_type,
                         method = method)
        return self
    
    def data_analysis(self):
        """
        Retrieves the results of the sensitivity analysis.

        Returns:
            pd.DataFrame: The results of the sensitivity analysis, including p-values and other statistics.

        Example:
            >>> sens = sensitivity(df=data, test_period=10, alpha=0.05)
            >>> sens.run()
            >>> analysis = sens.data_analysis()    
        """
        return self.analysis
    
    def plot(self, method = 'sensitivity', figsize=(25, 10), past_window = 25, back_window = 10):
        """
        Plots the results of the sensitivity analysis using the specified method.

        Args:
            method (str, optional): The plotting method to use ('sensitivity', 'training', or 'diagnostic'). Defaults to 'sensitivity'.
            figsize (tuple, optional): The size of the plot. Defaults to (25, 10).
            past_window (int, optional): The number of past data points to display in the training plot. Defaults to 25.
            back_window (int, optional): The number of back data points to display in the training plot. Defaults to 10.

        Raises:
            TypeError: If an invalid method is provided.

        Example:
            >>> sens = sensitivity(df=data, test_period=10, alpha=0.05)
            >>> sens.run()
            >>> sens.plot(method="sensitivity")        
        """
        
        if method not in ['sensitivity','training', 'diagnostic']:
            error = "Your method should be defined as one of these -> ('sensitivity','training', 'diagnostic')"
            raise TypeError(error)
        
        if method == 'sensitivity':
            area = mde_area(y = self.analysis.pvalue.values, x = self.analysis.index)
            return sensitivity_curve(arr1 = self.analysis.index, arr2 = self.analysis.pvalue.values, area = area, figsize = figsize)
        
        elif method == 'training':
            return plot_training(data = self.data, past_window = past_window, back_window = back_window, figsize = figsize, intervention = self.test_period)
        
        elif method == 'diagnostic':
            plot_diagnostic(data = self.data, figsize = figsize, intervention = self.test_period)
        
    
    def model_best_parameters(self):
        """
        Retrieves the best hyperparameters found during the sensitivity analysis.

        Returns:
            dict: A dictionary containing the best hyperparameters for the model.

        Example:
            >>> sens = sensitivity(df=data, test_period=10, alpha=0.05)
            >>> sens.run()
            >>> best_params = sens.model_best_parameters()
        """
        return self.hyper_parameters