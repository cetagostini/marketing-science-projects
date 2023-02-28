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

sns.set_theme()
sns.set_context("paper")

pd.options.mode.chained_assignment = None 

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
    summary()
        Plots an overview about the results.
    summary_intervention()
        Plots an overview of the results during intervention period.
    seasonal_decompose()
        Plots the seasonal decomposition of the time series.
    """

    def __init__(self,
        data: Union[np.array, DataFrame],
        intervention: Union[List[int], List[str], List[pd.Timestamp]],
        regressors: list = [],
        alpha: float = 0.05,
        past_window: int = 5,
        back_window: int = 25,
        seasonality: bool = True,
        n_samples: int = 1500,
        cross_validation_steps: int = 5,
        seasonality_mode: str = 'additive'
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
        self.seasonality_mode = seasonality_mode

        self.data, self.pre_int_metrics, self.int_metrics = synth_analysis(
            df = data, 
            regressors = regressors, 
            intervention = intervention, 
            seasonality = seasonality,
            cross_validation_steps = cross_validation_steps,
            alpha = alpha,
            model_mode = seasonality_mode
            )
        self.string_filter = "ds >= '{}' & ds <= '{}'".format(intervention[0],intervention[1])
        
        self.simulations = bootstrap_simulate(
                data = self.data.query(self.string_filter).yhat, 
                n_samples = n_samples, 
                n_steps = len(self.data.query(self.string_filter).index)
                )
        
        self.stadisticts, self.stats_ranges, self.samples_means = bootstrap_p_value(control = self.data.query(self.string_filter).yhat, treatment = self.data.query(self.string_filter).y, simulations = self.simulations)
 
    def plot_intervention(self):
        """
        The function plot_intervention is a method of the tscausalinference class. It generates a plot showing the predicted and actual values of the target variable around the intervention period, as well as the cumulative effect of the intervention.

        Parameters
        ----------
            No parameters are required.
        
        Raises
        -------
            No raises are defined.
        
        Returns
        --------
            No returns are defined, as the method simply generates a plot.
        """
        data = self.data.copy()
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
        The plot_simulations() method of tscausalinference class plots the distribution of the mean difference between the treatment and control group based on the bootstrap simulations generated in bootstrap_simulate(). 
        The plot includes a histogram and a box plot of the simulated means.

        Parameters
        ----------
            Simulation_number (int): The number of simulations to plot. Default is 10.
        
        Raises
        -------
            ValueError: if simulation_number is greater than the number of simulations generated.
        """
        data = self.data.copy()

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
        for i in range(simulation_number):#range(len(samples[0])):
            sns.lineplot(x=data[(data.ds >= pd.to_datetime(self.intervention[0]))&(data.ds <= pd.to_datetime(self.intervention[1]))].ds, y=self.simulations[i], 
                        linewidth=0.5, alpha=0.45,
                        color = 'orange', legend = False, ax=axes[0])

        sns.lineplot(x = 'ds', y = 'y', color = 'b',
                    ax=axes[0],
                    data = data[(data.ds >= pd.to_datetime(self.intervention[0]) - pd.Timedelta(days=self.back_window))],
                    linewidth=1, label='Training')

        sns.lineplot(x = 'ds', y = 'yhat', color = 'b',
                    err_kws={'linestyle': '--', 'hatch': '///', 'fc': 'none'},
                    data = data[(data.ds >= pd.to_datetime(self.intervention[0]))&(data.ds <= pd.to_datetime(self.intervention[1]))],
                    linewidth=1, label='Forcast', ls='--', ax=axes[0])

        sns.lineplot(x = 'ds', y = 'y', color = 'g',
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
    
    def summary(self):
        """
        Parameters
        ----------
            No parameters are required.
        
        Raises
        -------
            No raises are defined.
        
        Returns
        --------
            No returns are defined, as the method simply generates a overview.
        """
        data = self.data.copy()
        
        data['ds'] = pd.to_datetime(data['ds'])

        std_res = self.pre_int_metrics[3][1]

        if std_res < 0.2:
            mde = 15
            noise = 'low'
        elif (std_res > 0.2) & (std_res <= 0.6):
            mde = 21
            noise = 'medium'
        elif std_res > 0.6:
            mde = 25
            noise = 'high'
        
        if self.stadisticts[0] <= 0.01:
            summary = """
    Considerations
    --------------
    a) The standard deviation of the residuals is {}. This means that the noise in your data is {}.
    b) Based on this information, in order for your effect to be detectable, it should be greater than {}%.

    Summary
    -------
    During the intervention period, the response variable had an average value of approximately {}. 
    By contrast, in the absence of an intervention, we would have expected an average response of {}. 
    The 95% confidence interval of this counterfactual prediction is {} to {}.

    The usual error of your model is {}%, while the difference during the intervention period is {}%. 
    During the intervention, the error increase {}% ({} percentage points), suggesting some factor is 
    impacting the quality of the model, and the differences are significant.

    The probability of obtaining this effect by chance is very small 
    (after {} simulations, bootstrap probability p = {}). 
    This means that the causal effect can be considered statistically significant.
            """
        else:
            summary = """
    Considerations
    --------------
    a) The standard deviation of the residuals is {}. This means that the noise in your data is {}.
    b) Based on this information, in order for your effect to be detectable, it should be greater than {}%.

    Summary
    -------
    During the intervention period, the response variable had an average value of approximately {}. 
    By contrast, in the absence of an intervention, we would have expected an average response of {}. 
    The 95% confidence interval of this counterfactual prediction is {} to {}.

    The usual error of your model is {}%, while the difference during the intervention period is {}%. 
    During the intervention, the error increase {}% ({} percentage points), suggesting that the model can explain well what should happen,
    and that the differences are not significant.

    The probability of obtaining this effect by chance is not small 
    (after {} simulations, bootstrap probability p = {}). 
    This means that the causal effect cannot be considered statistically significant.
            """

        print(
            summary.format(
                round(std_res,5),
                noise,
                mde,
                round(data[(data.ds >= pd.to_datetime(self.intervention[0])) & (data.ds <= pd.to_datetime(self.intervention[1]))].y.mean(),2),
                round(data[(data.ds >= pd.to_datetime(self.intervention[0])) & (data.ds <= pd.to_datetime(self.intervention[1]))].yhat.mean(),2),
                round(data[(data.ds >= self.intervention[0]) & (data.ds <= self.intervention[1])].yhat_lower.mean(),2), 
                round(data[(data.ds >= self.intervention[0]) & (data.ds <= self.intervention[1])].yhat_upper.mean(),2),
                round(self.pre_int_metrics[2][1],2),
                round(self.int_metrics[3][1],2),
                round(self.int_metrics[3][1]/self.pre_int_metrics[2][1],2),
                round(self.int_metrics[3][1] - self.pre_int_metrics[2][1],2),
                self.n_samples,
                round(self.stadisticts[0],5)
            )
        )
    
    def summary_intervention(self):
        """
        Parameters
        ----------
            No parameters are required.
        
        Raises
        -------
            No raises are defined.
        
        Returns
        --------
            No returns are defined, as the method simply generates a overview.
        """
        data = self.data

        strings_info = """
+-----------------------+-----------+
        intervention metrics
+-----------------------+-----------+
{}
+-----------------------+-----------+
      {}
        """

        print(
            strings_info.format(
                tabulate(self.int_metrics, 
                headers=['Metric', 'Value'], 
                tablefmt='pipe'),
                'CI 95%: [{}, {}]'.format(
                    round(data[(data.ds >= self.intervention[0]) & (data.ds <= self.intervention[1])].yhat_lower.sum(),2), 
                    round(data[(data.ds >= self.intervention[0]) & (data.ds <= self.intervention[1])].yhat_upper.sum()),2)
            ).strip()
        )

    def seasonal_decompose(self):
        data = self.data.copy()
        data['ds'] = pd.to_datetime(data['ds'])
        data.set_index('ds', inplace = True)

        data = data[(data.index < self.intervention[0])].copy()

        cols = list(set(data.columns.to_list()) - set(['point_effects', 'yhat', 'yhat_lower', 'yhat_upper', 'cummulitive_y', 'cummulitive_yhat', 'cummulitive_effect', 'cummulitive_yhat_lower', 'cummulitive_yhat_upper']))

        fig, axes = plt.subplots(nrows=len(cols), ncols=1, figsize=(18, 12))

        for number, name in enumerate(cols):
            sns.lineplot(x = data.index, y = data[name], color = 'b', ax=axes[number], linewidth=1, label = name)
        
        # Show the plot
        sns.despine()

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