import pandas as pd
import numpy as np

from tabulate import tabulate

def summary(data = None, statistical_significance = 0.05, 
            stadisticts = None, pre_int_metrics = None, 
            int_metrics = None, intervention = None, n_samples = None,
            ci_int = None):
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
    data = data.copy()
    
    data['ds'] = pd.to_datetime(data['ds'])

    std_res = pre_int_metrics[3][1]

    if std_res < 0.2:
        mde = 15
        noise = 'low'
    elif (std_res > 0.2) & (std_res <= 0.6):
        mde = 21
        noise = 'medium'
    elif std_res > 0.6:
        mde = 25
        noise = 'high'
    
    if stadisticts[0] <= statistical_significance:
        summary = """
Summary
-------
During the intervention period, the response variable had an average value of approximately {real_mean}. 
By contrast, in the absence of an intervention, we would have expected an average response of {pred_mean}. 
The 95% confidence interval of this counterfactual prediction is {ci_lower} to {ci_upper}.

The usual error of your model is between -{general_mape}% & to {general_mape}% , during the intervention period was {real_mape}%. 
suggesting some factor is impacting the quality of the model, and the differences are significant.

The probability of obtaining this effect by chance is very small 
(after {n_simulations} simulations, bootstrap probability p = {pvalue}). 
This means that the causal effect can be considered statistically significant.
        """
    else:
        summary = """
Summary
-------
During the intervention period, the response variable had an average value of approximately {real_mean}. 
By contrast, in the absence of an intervention, we would have expected an average response of {pred_mean}. 
The 95% confidence interval of this counterfactual prediction is {ci_lower} to {ci_upper}.

The usual error of your model is between -{general_mape}% to {general_mape}% , during the intervention period was {real_mape}%. 
suggesting that the model can explain well what should happen, and that the differences are not significant.

The probability of obtaining this effect by chance is not small 
(after {n_simulations} simulations, bootstrap probability p = {pvalue}). 
This means that the causal effect cannot be considered statistically significant.
        """

    print(
        summary.format(
            real_mean = round(data[(data.ds >= pd.to_datetime(intervention[0])) & (data.ds <= pd.to_datetime(intervention[1]))].y.mean(),2),
            pred_mean = round(data[(data.ds >= pd.to_datetime(intervention[0])) & (data.ds <= pd.to_datetime(intervention[1]))].yhat.mean(),2),
            ci_lower = round(ci_int[0], 2), 
            ci_upper= round(ci_int[1], 2),
            general_mape = abs(round(pre_int_metrics[2][1],2)),
            real_mape = abs(round(int_metrics[4][1],2)),
            n_simulations = n_samples,
            pvalue = round(stadisticts[0],5)
        )
    )
    
def summary_intervention(data, intervention = None, int_metrics = None, stadisticts = None):
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
    data = data.copy()

    # Create two dates as strings
    date_str1 = intervention[0]
    date_str2 = intervention[1]

    # Convert the strings to datetime objects
    date1 = pd.to_datetime(date_str1)
    date2 = pd.to_datetime(date_str2)

    # Calculate the difference in days
    days_diff = int((date2 - date1).days) + 1

    lower = stadisticts[0] * days_diff
    upper = stadisticts[1] * days_diff

    strings_info = """
+-----------------------+------------+
 / / / / intervention metrics / / / /
+-----------------------+------------+
{}
+-----------------------+------------+
    {}
    """

    print(
        strings_info.format(
            tabulate(int_metrics, 
            headers=['Metric', 'Value'], 
            tablefmt='pipe'),
            'CI 95%: [{}, {}]\n    CCI 95%: [{}, {}]'.format(
                round(stadisticts[0], 0), 
                round(stadisticts[1], 0),
                round(lower, 0),
                round(upper, 0),
            )
        ).strip()
    )

def summary_incrementality(variable = None, intervention = None, int_metrics = None, window = 90):
    
    intervention_dates = [pd.to_datetime(intervention[0]), pd.to_datetime(intervention[1])]

    # Filter the variable results to the last 90 days before intervention[0]
    variable_filt = variable[(intervention_dates[0] - np.timedelta64(window, 'D')):intervention_dates[0]]

    # Calculate the mean and median of the last 90 days of the variable
    var_mean_90 = np.mean(variable_filt)

    # Calculate the increment on the variable mean during the intervention period
    var_intervention = np.mean(variable[intervention_dates[0]:intervention_dates[1]])
    var_mean_diff = var_intervention - var_mean_90

    # Estimate the additional unit added on int_metrics by each unit added on the variable during intervention
    incremental_unit = (var_mean_diff * (intervention_dates[1] - intervention_dates[0]).days) / abs(round(int_metrics[2][1], 0))
    
    # Calculate the percentage increase in the intervention period
    percentage_increase = (var_mean_diff / var_mean_90) * 100

    str_window = str(window)

    table_info = [[f'Last {str_window} days Mean', var_mean_90],
                    ['Intervention Mean',var_intervention], ['Increase (%)',percentage_increase], 
                    ['Diff. Units',var_mean_diff],['Incremental Unit Value',incremental_unit]
                    ]

    # Format the print statement
    print(f"""\
summary
-------
Each extra unit on ´y´ represents {incremental_unit:.2f} units on your variable.

+-----------------------+------------+
 / / / /   DETAILED OVERVIEW   / / / /
+-----------------------+------------+
""")

    print(
        tabulate(table_info, 
        headers=['METRIC', 'VALUE'], 
        tablefmt='pipe')
        )