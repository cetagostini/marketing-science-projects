
from pandas import DataFrame
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
sns.set_context("paper")

pd.options.mode.chained_assignment = None 


def plot_intervention(data, past_window: int = 5, back_window: int = 25, figsize=(15, 10), intervention = None):
        """
        Plots the pre-intervention and post-intervention time-series data, as well as the predicted values and confidence intervals.

        Args:
            past_window: An integer representing the number of days to include after the end of the intervention period.
            back_window: An integer representing the number of days to include before the start of the intervention period.
            figsize: A tuple representing the figure size in inches.

        Returns:
            None

        Raises:
            None
        """
        data = data.copy()
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)

        lineplot = sns.lineplot(x = 'ds', y = 'yhat', color = 'r', alpha=0.5, linestyle='--', ci=95,
                            err_kws={'linestyle': '--', 'hatch': '///', 'fc': 'none'}, ax=axes[0],
                    data = data[(data.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=back_window))&(data.ds <= pd.to_datetime(intervention[1]) + pd.Timedelta(days=past_window))]
                    )

        sns.lineplot(x = 'ds', y = 'y', ax=axes[0], color = 'b',
                    data = data[(data.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=back_window))&(data.ds <= pd.to_datetime(intervention[1]) + pd.Timedelta(days=past_window))]
                    )

        lineplot.axvline(pd.to_datetime(intervention[0]), color='r', linestyle='--',alpha=.5)
        lineplot.axvline(pd.to_datetime(intervention[1]), color='r', linestyle='--',alpha=.5)

        lineplot.fill_between(data[(data.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=back_window))&(data.ds <= pd.to_datetime(intervention[1]) + pd.Timedelta(days=past_window))]['ds'], 
                        data[(data.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=back_window))&(data.ds <= pd.to_datetime(intervention[1]) + pd.Timedelta(days=past_window))]['yhat_lower'],
                        data[(data.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=back_window))&(data.ds <= pd.to_datetime(intervention[1]) + pd.Timedelta(days=past_window))]['yhat_upper'], 
                        color='r', alpha=.1)

        lineplot.legend(['Prediction', 'Real'])

        cumplot = sns.lineplot(x = 'ds', y = 'point_effects', color = 'g',
             data = data[(data.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=back_window))&(data.ds <= pd.to_datetime(intervention[1]) + pd.Timedelta(days=past_window))],
             ax=axes[1]
             )

        sns.lineplot(x = 'ds', y = 'point_effects', color = 'b', linestyle='--',
                    data = data[(data.ds >= intervention[0]) & (data.ds <= intervention[1])],
                    ax=axes[1]
                    )

        sns.lineplot(x = 'ds', y = 'point_effects', color = 'g',
                    data = data[(data.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=back_window))&(data.ds <= pd.to_datetime(intervention[1]) + pd.Timedelta(days=past_window))],
                    ax=axes[1]
                    )

        cumplot.axvline(pd.to_datetime(intervention[0]), color='r', linestyle='--')
        cumplot.axvline(pd.to_datetime(intervention[1]), color='r', linestyle='--')

        cumplot.axvspan(pd.to_datetime(intervention[0]), pd.to_datetime(intervention[1]), alpha=0.07, color='r')

        plt.show()

def plot_simulations(data, simulation_number: int = 10, past_window: int = 5, back_window: int = 25, figsize=(18, 5), 
                     intervention = None, simulations = None, stadisticts = None, stats_ranges = None, samples_means = None):
        """
        The plot_simulations() method of tscausalinference class plots the distribution of the mean difference between the treatment and control group based on the bootstrap simulations generated in bootstrap_simulate(). 
        The plot includes a histogram and a box plot of the simulated means.

        Args:
            Simulation_number (int): The number of simulations to plot. Default is 10.
            past_window: An integer representing the number of days to include after the end of the intervention period.
            back_window: An integer representing the number of days to include before the start of the intervention period.
            figsize: A tuple representing the figure size in inches.
        
        Raises:
            ValueError: if simulation_number is greater than the number of simulations generated.
        """
        data = data.copy()

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        for i in range(simulation_number):#range(len(samples[0])):
            sns.lineplot(x=data[(data.ds >= pd.to_datetime(intervention[0]))&(data.ds <= pd.to_datetime(intervention[1]))].ds, y=simulations[i], 
                        linewidth=0.5, alpha=0.45,
                        color = 'orange', legend = False, ax=axes[0])

        sns.lineplot(x = 'ds', y = 'y', color = 'b',
                    ax=axes[0],
                    data = data[(data.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=back_window))&(data.ds <= pd.to_datetime(intervention[1]) + pd.Timedelta(days=past_window))],
                    linewidth=1, label='Training')

        sns.lineplot(x = 'ds', y = 'yhat', color = 'b',
                    err_kws={'linestyle': '--', 'hatch': '///', 'fc': 'none'},
                    data = data[(data.ds >= pd.to_datetime(intervention[0]))&(data.ds <= pd.to_datetime(intervention[1]))],
                    linewidth=1, label='Forcast', ls='--', ax=axes[0])

        sns.lineplot(x = 'ds', y = 'y', color = 'g',
                    data = data[(data.ds >= pd.to_datetime(intervention[0]))&(data.ds <= pd.to_datetime(intervention[1]))],
                    label='Real', ax=axes[0])
        
        # add a custom legend entry for the yellow line
        custom_legend = plt.Line2D([], [], color='orange', label='simulations')

        # create a new legend with your custom entry and add it to the plot
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(custom_legend)
        axes[0].legend(handles=handles, loc='upper left')

        # Add the mean value to the right corner
        plt.text(1.05, 0.95, f'P Value: {stadisticts[0]:.2f}', ha='left', va='center', transform=plt.gca().transAxes)
        #plt.text(1.05, 0.80, f'P Center: {stadisticts[1]:.2f}', ha='left', va='center', transform=plt.gca().transAxes)
        #plt.text(1.05, 0.75, f'Prob. NonEffect: {stadisticts[2]:.2f}', ha='left', va='center', transform=plt.gca().transAxes)

        sns.histplot(samples_means, kde=True, ax=axes[1])

        # Add title and labels
        plt.title('Histogram of Bootstrapped Means')
        plt.xlabel('Bootstrapped Mean Value')
        plt.ylabel('Frequency')

        plt.axvline(stats_ranges[0], color='r', linestyle='--',alpha=.5,)
        plt.axvline(stats_ranges[1], color='r', linestyle='--',alpha=.5,)

        # Add the real mean as a scatter point
        plt.scatter(data[(data.ds >= pd.to_datetime(intervention[0]))&(data.ds <= pd.to_datetime(intervention[1]))].y.mean(), 0, color='orange', s=600)

        # Show the plot
        sns.despine()

def seasonal_decompose(data, intervention, figsize=(18, 12)):
    data = data.copy()
    data['ds'] = pd.to_datetime(data['ds'])
    data.set_index('ds', inplace = True)

    data = data[(data.index < intervention[0])].copy()

    cols = list(set(data.columns.to_list()) - set(['point_effects', 'yhat', 'yhat_lower', 'yhat_upper', 'cummulitive_y', 'cummulitive_yhat', 'cummulitive_effect', 'cummulitive_yhat_lower', 'cummulitive_yhat_upper']))

    fig, axes = plt.subplots(nrows=len(cols), ncols=1, figsize=figsize)

    for number, name in enumerate(cols):
        sns.lineplot(x = data.index, y = data[name], color = 'b', ax=axes[number], linewidth=1, label = name)
    
    # Show the plot
    sns.despine()