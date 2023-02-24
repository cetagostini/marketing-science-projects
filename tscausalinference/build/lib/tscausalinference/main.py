from typing import Any, Dict, List, Optional, Tuple, Union

from pandas import DataFrame
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

from tscausalinference.synth_regression import synth_analysis
from tscausalinference.bootstrap import bootstrap_simulate, bootstrap_p_value

sns.set_theme()
sns.set_context("paper")

class tscausalinference:
    """
    Time series causal inference using structural causal models (SCM) and a difference-in-differences (DiD) approach.

    Parameters
    ----------
    data : Union[np.array, DataFrame]
        A NumPy array or Pandas DataFrame containing the time series data.
    intervention : Union[List[int], List[str], List[pd.Timestamp]]
        A list of two elements indicating the start and end dates of the intervention.
    regressors : list, optional
        A list of column names representing regressors in the SCM. Default is an empty list.
    alpha : float, optional
        The level of significance for the hypothesis test. Default is 0.05.
    past_window : int, optional
        The number of days to show in the plot before the start of the intervention. Default is 5.
    back_window : int, optional
        The number of days to show in the plot after the end of the intervention. Default is 25.
    seasonality : bool, optional
        If True, the model includes a seasonal component. Default is True.
    n_samples : int, optional
        The number of bootstrap samples to simulate. Default is 1500.
    cross_validation_steps : int, optional
        The number of cross-validation steps used for training. Default is 5.

    Methods
    -------
    plot_intervention()
        Plots the time series data before and after the intervention with confidence intervals.
    plot_simulations(simulation_number: int = 10)
        Plots a specified number of bootstrap simulations.
    """

    def __init__(self,
        data: Union[np.array, DataFrame] = None,
        intervention: Union[List[int], List[str], List[pd.Timestamp]] = None,
        regressors: list = [],
        alpha: float = 0.05,
        past_window: int = 5,
        back_window: int = 25,
        seasonality: bool = True,
        n_samples: int = 1500,
        cross_validation_steps: int = 5
        ):

        self.data = data
        self.intervention = intervention
        self.alpha = alpha
        self.past_window = past_window
        self.back_window = back_window
        self.seasonality = seasonality
        self.regressors = regressors
        self.n_samples = n_samples
        self.cross_validation_steps = cross_validation_steps

        self.data = synth_analysis(df = data, 
                    regressors = regressors, 
                    intervention = intervention, 
                    seasonality = seasonality,
                    cross_validation_steps = cross_validation_steps
                    )
        self.string_filter = "ds >= '{}' & ds <= '{}'".format(intervention[0],intervention[1])
        
        self.simulations = bootstrap_simulate(
                data = self.data.query(self.string_filter).yhat, 
                n_samples = n_samples, 
                n_steps = len(self.data.query(self.string_filter).index)
                )
        
        self.stadisticts, self.stats_ranges, self.samples_means = bootstrap_p_value(control = self.data.query(self.string_filter).yhat, treatment = self.data.query(self.string_filter).y, simulations = self.simulations, center = False)

    
    def plot_intervention(self):
        """
        The function plot_intervention is a method of the tscausalinference class. It generates a plot showing the predicted and actual values of the target variable around the intervention period, as well as the cumulative effect of the intervention.

        Parameters
        ----------
            No parameters are required.
        
        Raises
        ------
            No raises are defined.
        
        Returns
        -------
            No returns are defined, as the method simply generates a plot.
        """
        data = self.data
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

        lineplot = sns.lineplot(x = 'ds', y = 'yhat', color = 'r', alpha=0.5, linestyle='--', ci=95,
                            err_kws={'linestyle': '--', 'hatch': '///', 'fc': 'none'}, ax=axes[0],
                    data = data[(data.ds >= pd.to_datetime(self.intervention[0]) - pd.Timedelta(days=self.back_window))&(data.ds <= pd.to_datetime(self.intervention[1]) + pd.Timedelta(days=self.past_window))]
                    )

        sns.lineplot(x = 'ds', y = 'y', ax=axes[0], color = 'b',
                    data = data[(data.ds >= pd.to_datetime(self.intervention[0]) - pd.Timedelta(days=self.back_window))&(data.ds <= pd.to_datetime(self.intervention[1]) + pd.Timedelta(days=self.past_window))]
                    )

        lineplot.axvline(pd.to_datetime(self.intervention[0]), color='r', linestyle='--',alpha=.5)
        lineplot.axvline(pd.to_datetime(self.intervention[1]), color='r', linestyle='--',alpha=.5)

        lineplot.fill_between(data[(data.ds >= pd.to_datetime(self.intervention[0]) - pd.Timedelta(days=self.back_window))&(data.ds <= pd.to_datetime(self.intervention[1]) + pd.Timedelta(days=self.past_window))]['ds'], 
                        data[(data.ds >= pd.to_datetime(self.intervention[0]) - pd.Timedelta(days=self.back_window))&(data.ds <= pd.to_datetime(self.intervention[1]) + pd.Timedelta(days=self.past_window))]['yhat_lower'],
                        data[(data.ds >= pd.to_datetime(self.intervention[0]) - pd.Timedelta(days=self.back_window))&(data.ds <= pd.to_datetime(self.intervention[1]) + pd.Timedelta(days=self.past_window))]['yhat_upper'], 
                        color='r', alpha=.1)

        lineplot.legend(['Prediction', 'Real'])

        cumplot = sns.lineplot(x = 'ds', y = 'cummulitive_effect', color = 'g',
             data = data[(data.ds >= pd.to_datetime(self.intervention[0]) - pd.Timedelta(days=self.back_window))&(data.ds <= pd.to_datetime(self.intervention[1]) + pd.Timedelta(days=self.past_window))],
             ax=axes[1]
             )

        sns.lineplot(x = 'ds', y = 'cummulitive_effect', color = 'b', linestyle='--',
                    data = data[(data.ds >= self.intervention[0]) & (data.ds <= self.intervention[1])],
                    ax=axes[1]
                    )

        sns.lineplot(x = 'ds', y = 'cummulitive_effect', color = 'g',
                    data = data[data.ds > self.intervention[1]],
                    ax=axes[1]
                    )

        cumplot.axvline(pd.to_datetime(self.intervention[0]), color='r', linestyle='--')
        cumplot.axvline(pd.to_datetime(self.intervention[1]), color='r', linestyle='--')

        cumplot.axvspan(pd.to_datetime(self.intervention[0]), pd.to_datetime(self.intervention[1]), alpha=0.07, color='r')

        plt.show()

    def plot_simulations(self, simulation_number: int = 10):
        """
        The plot_simulations() method of tscausalinference class plots the distribution of the mean difference between the treatment and control group based on the bootstrap simulations generated in bootstrap_simulate(). The plot includes a histogram and a box plot of the simulated means.

        Parameters
        ----------
            Simulation_number (int): The number of simulations to plot. Default is 10.
        
        Raises
        ------
            ValueError: if simulation_number is greater than the number of simulations generated.
        """
        data = self.data

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
        for i in range(simulation_number):#range(len(samples[0])):
            sns.lineplot(x=data[(data.ds >= pd.to_datetime(self.intervention[0]))&(data.ds <= pd.to_datetime(self.intervention[1]))].ds, y=self.simulations[i], 
                        linewidth=0.5, alpha=0.45,
                        color = 'orange', legend = False, ax=axes[0])

        sns.lineplot(x = 'ds', y = 'yhat', color = 'b',
                    err_kws={'linestyle': '--', 'hatch': '///', 'fc': 'none'}, ax=axes[0],
                    data = data[(data.ds <= pd.to_datetime(self.intervention[0]))],
                    linewidth=1, label='Training')

        sns.lineplot(x = 'ds', y = 'yhat', color = 'b',
                    err_kws={'linestyle': '--', 'hatch': '///', 'fc': 'none'},
                    data = data[(data.ds >= pd.to_datetime(self.intervention[0]))&(data.ds <= pd.to_datetime(self.intervention[1]))],
                    linewidth=1, label='Forcast', ls='--', ax=axes[0])

        sns.lineplot(x = 'ds', y = 'y', color = 'g',
                    err_kws={'linestyle': '--', 'hatch': '///', 'fc': 'none'},
                    data = data[(data.ds >= pd.to_datetime(self.intervention[0]))&(data.ds <= pd.to_datetime(self.intervention[1]))],
                    label='Real', ax=axes[0])
        
        # add a custom legend entry for the yellow line
        custom_legend = plt.Line2D([], [], color='orange', label='simulations')

        # create a new legend with your custom entry and add it to the plot
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(custom_legend)
        axes[0].legend(handles=handles, loc='upper left')

        # Add the mean value to the right corner
        plt.text(1.05, 0.95, f'P-Value: {self.stadisticts[0]:.2f}', ha='left', va='center', transform=plt.gca().transAxes)
        plt.text(1.05, 0.80, f'P-Effect: {self.stadisticts[1]:.2f}', ha='left', va='center', transform=plt.gca().transAxes)

        sns.histplot(self.samples_means, kde=True, ax=axes[1])

        # Add title and labels
        plt.title('Histogram of Bootstrapped Means')
        plt.xlabel('Bootstrapped Mean Value')
        plt.ylabel('Frequency')

        plt.axvline(self.stats_ranges[0], color='r', linestyle='--',alpha=.5,)
        plt.axvline(self.stats_ranges[1], color='r', linestyle='--',alpha=.5,)

        # Add the real mean as a scatter point
        plt.scatter(data[(data.ds >= pd.to_datetime(self.intervention[0]))&(data.ds <= pd.to_datetime(self.intervention[1]))].y.mean(), 0, color='orange', s=600)

        # Show the plot
        sns.despine()