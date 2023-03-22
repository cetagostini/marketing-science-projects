
from pandas import DataFrame
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline
from sklearn.metrics import r2_score

sns.set_theme()
sns.set_context("paper")

pd.options.mode.chained_assignment = None 


def plot_intervention(data, past_window: int = 5, back_window: int = 25, figsize=(15, 10), intervention = None):
        """
        Plots the effect of an intervention over time.

        Args:
        data (DataFrame): a pandas DataFrame with columns 'ds', 'y', 'yhat', 'yhat_upper', 'yhat_lower', and 'point_effects'.
        past_window (int): an integer with the number of days to include before the start of the intervention period. Default is 5.
        back_window (int): an integer with the number of days to include after the end of the intervention period. Default is 25.
        figsize (tuple): a tuple with the size of the figure to create. Default is (15, 10).
        intervention (list): a list with two strings representing the start and end date of the intervention in 'yyyy-mm-dd' format.

        Returns:
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

        # Add horizontal lines for percentiles
        perc_75 = np.percentile(data[data.ds < pd.to_datetime(intervention[0])]['point_effects'], 75)
        perc_25 = np.percentile(data[data.ds < pd.to_datetime(intervention[0])]['point_effects'], 25)
        median = np.percentile(data[data.ds < pd.to_datetime(intervention[0])]['point_effects'], 50)

        cumplot.axhline(y=perc_75, linestyle='--', color='grey', alpha=0.5)
        cumplot.axhline(y=perc_25, linestyle='--', color='grey', alpha=0.5)
        cumplot.axhline(y=median, linestyle='--', color='grey', alpha=0.5)

        plt.show()

def plot_simulations(data, simulation_number: int = 10, past_window: int = 5, back_window: int = 25, figsize=(18, 5), 
                     intervention = None, simulations = None, stadisticts = None, stats_ranges = None, samples_means = None):
        """
        Plots a graph that shows simulations and prediction variance. Also plots a histogram of bootstrapped means.

        Args:
        data (pd.DataFrame): The dataset containing the time series
        simulation_number (int): The number of simulations to plot
        past_window (int): The number of days before the intervention period to plot the training set
        back_window (int): The number of days after the intervention period to plot the training set
        figsize (tuple): The size of the figure
        intervention (tuple): The start and end date of the intervention period
        simulations (ndarray): The array containing the simulations
        stadisticts (list): A list containing statistics values
        stats_ranges (list): A list containing the minimum and maximum values to plot in the histogram
        samples_means (ndarray): An array containing the bootstrapped means

        Returns:
        None
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
    """
    Decompose a time series into seasonal and trend components, and visualize the results.

    Args:
        data (pd.DataFrame): A time series to decompose. Must contain a column 'ds' with datetime values.
        intervention (tuple): A tuple with two datetime values representing the start and end of an intervention period.
        figsize (tuple, optional): The size of the figure to display the subplots. Defaults to (18, 12).

    Returns:
        None

    Raises:
        ValueError: If the 'ds' column is missing from the input DataFrame.

    Example:
        seasonal_decompose(my_data, ('2020-01-01', '2020-03-31'), figsize=(20, 10))
    """
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

def sensitivity_curve(arr1 = None, arr2 = None, area = None, figsize=(25, 8)):
    # create the plot
    fig, ax = plt.subplots(figsize = figsize)

    # plot the markers
    sns.scatterplot(x=arr1, y=arr2, ax=ax, s=100, color='grey', label='P-Value')

    # interpolate the line using a spline
    x_new = np.linspace(arr1.min(), arr1.max(), 300)
    spl = make_interp_spline(arr1, arr2, k=3)  # k=3 for cubic spline
    y_smooth = spl(x_new)

    # plot the interpolated line
    ax.plot(x_new, y_smooth, color='violet', linewidth=3, label='Effect')

    bbox_props = dict(boxstyle='round,pad=0.5', fc='white', ec='gray', lw=1) # set the box properties
    ax.text(1.0, 0.4, f'Sensitivity: {area:.2f}', fontsize=14, bbox=bbox_props) # add the text box annotation to the plot

    # add axis labels and a title
    ax.set_xlabel('Estimated Effect', fontsize=14)
    ax.set_ylabel('P-Value', fontsize=14)
    ax.set_title('Minimum Detectable Effect over Time Series', fontsize=18)

    # add a legend
    ax.legend(fontsize=12)

    # add x-axis and y-axis ticks and tick labels
    ax.tick_params(axis='both', labelsize=12)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))

    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    idx = np.argmax(arr2 < 0.05)
    ax.axvline(x=arr1[idx], linestyle='--', color='gray')

    plt.show()

def plot_training(data, past_window: int = 5, back_window: int = 25, figsize=(15, 10), intervention = None):
        data = data.copy()
        fig, axes = plt.subplots(figsize=figsize)

        lineplot = sns.lineplot(x = 'ds', y = 'yhat', color = 'r', alpha=0.5, linestyle='--', ci=95,
                            err_kws={'linestyle': '--', 'hatch': '///', 'fc': 'none'}, ax=axes,
                    data = data[(data.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=back_window))&(data.ds <= pd.to_datetime(intervention[1]) + pd.Timedelta(days=past_window))]
                    )

        sns.lineplot(x = 'ds', y = 'y', ax=axes, color = 'b',
                    data = data[(data.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=back_window))&(data.ds <= pd.to_datetime(intervention[1]) + pd.Timedelta(days=past_window))]
                    )

        lineplot.axvline(pd.to_datetime(intervention[0]), color='r', linestyle='--',alpha=.5)
        lineplot.axvline(pd.to_datetime(intervention[1]), color='r', linestyle='--',alpha=.5)

        lineplot.fill_between(data[(data.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=back_window))&(data.ds <= pd.to_datetime(intervention[1]) + pd.Timedelta(days=past_window))]['ds'], 
                        data[(data.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=back_window))&(data.ds <= pd.to_datetime(intervention[1]) + pd.Timedelta(days=past_window))]['yhat_lower'],
                        data[(data.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=back_window))&(data.ds <= pd.to_datetime(intervention[1]) + pd.Timedelta(days=past_window))]['yhat_upper'], 
                        color='r', alpha=.1)
        
        # Add horizontal lines for percentiles
        perc_75 = np.percentile(data['y'], 75)
        perc_25 = np.percentile(data['y'], 25)
        median = np.percentile(data['y'], 50)

        axes.axhline(y=perc_75, linestyle='--', color='grey', alpha=0.5)
        axes.axhline(y=perc_25, linestyle='--', color='grey', alpha=0.5)
        axes.axhline(y=median, linestyle='--', color='grey', alpha=0.5)

        lineplot.legend(['Prediction', 'Real'])

        axes.set_xlabel('Ds')
        axes.set_ylabel('Y')
        axes.set_title('Model fit in training and test period')

        plt.show()

def plot_diagnostic(data = DataFrame(), figsize = (25, 10), intervention = []):
    data = data.copy()
    training = data[data.ds <= (pd.to_datetime(intervention[0]) - pd.Timedelta(days=1))].copy()
    test = data[data.ds > pd.to_datetime(intervention[1])].copy()

    bbox_props = dict(boxstyle='round,pad=0.5', fc='white', ec='gray', lw=1) # set the box properties
    
    r_train = r2_score(training['y'], training['yhat'])
    r_test = r2_score(test['y'], test['yhat'])

    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = figsize, sharex=False, sharey=False)
    fig.suptitle('Model Diagnostics')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    scatter_train = sns.scatterplot(data=training, x="yhat", y="y", ax=axes[0][0])
    scatter_train.set_title('Model fit in Training')

    axes[0][0].text(1.0, 0.4, f'R2: {r_train:.2f}', fontsize = 14, bbox = bbox_props)

    scatter_test = sns.scatterplot(data=test, x="yhat", y="y", ax=axes[0][1])
    scatter_test.set_title('Model fit in Test')

    axes[0][1].text(1.0, 0.4, f'R2: {r_test:.2f}', fontsize = 14, bbox = bbox_props)

    hist_train = sns.histplot(data = training, x="residuals", ax=axes[1][0])
    hist_train.set_title('Model residuals in Training')

    hist_test = sns.histplot(data = test, x="residuals", ax=axes[1][1])
    hist_test.set_title('Model residuals in Test')
    
    # Show the plot
    sns.despine()