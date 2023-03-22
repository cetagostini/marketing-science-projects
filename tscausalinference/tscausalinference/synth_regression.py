from prophet.utilities import regressor_coefficients

from tabulate import tabulate
from sklearn.metrics import r2_score, mean_absolute_error

import pandas as pd
from pandas import DataFrame

import numpy as np

import itertools

import logging

from tscausalinference.evaluators import mape
from tscausalinference.regression import prophet_regression

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

logging.getLogger('fbprophet').setLevel(logging.ERROR)

pd.options.mode.chained_assignment = None 

def synth_analysis(df: DataFrame = None, 
                    regressors: list = [], 
                    intervention: list = None, 
                    cross_validation_steps: int = 5,
                    alpha: float = 0.05,
                    model_params: dict = {},
                    verbose = True,
                    model_type = 'gam',
                    autocorrelation = False
                    ):
    """
    Fits a Prophet model and computes performance metrics for a given input DataFrame. The function is designed to work with
    time series data that has a "ds" column with datetime values and a "y" column with numerical values. Optionally, the
    function can also take a list of regressors as input.

    Args:
        df: A pandas DataFrame with time series data.
        regressors: A list of strings indicating the names of columns in `df` that should be considered as regressors.
        intervention: A list containing two strings representing the start date and end date of an intervention period to be
            analyzed. The dates should be specified in the format "YYYY-MM-DD".
        cross_validation_steps: An integer representing the number of steps to use for cross-validation.
        alpha: A float between 0 and 1 representing the confidence interval to use for predictions.
        model_params: A dictionary containing the parameters to use for the Prophet model.

    Returns:
        data: A pandas DataFrame with columns 'ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper' and additional columns for each
            regressor specified in `regressors`. The DataFrame contains predictions made by the Prophet model and actual
            values from the input DataFrame.
        pre_int_metrics: A list of metrics computed for the period before the intervention. Each metric is represented as a
            list containing two elements: a string with the metric name and a float with its value.
        int_metrics: A list of metrics computed for the intervention period. Each metric is represented as a list containing
            two elements: a string with the metric name and a float with its value.

    Raises:
        ValueError: If `df` is not a pandas DataFrame or if it does not contain columns named 'ds' and 'y', or if it is
            empty.
        TypeError: If `intervention` is None or if `model_params` is empty or if one of the parameters in `model_params`
            is not of type int, float, or str.

    Example:
        To fit a Prophet model on an example dataframe `df`, run:

        >>> data, pre_int_metrics, int_metrics = synth_analysis(df=df,
                                                                regressors=['holiday', 'temperature'],
                                                                intervention=['2021-01-01', '2021-02-01'],
                                                                cross_validation_steps=3,
                                                                alpha=0.1,
                                                                model_params={'changepoint_range': [0.8, 0.9],
                                                                              'yearly_seasonality': True,
                                                                              'weekly_seasonality': True})
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    elif not set(['ds', 'y']).issubset(df.columns):
        raise ValueError("df must contain columns named 'ds' and 'y'")
    elif df.empty:
        raise TypeError("df is empty")

    if isinstance(intervention, type(None)):
      intervention_error = 'Define your intervention as a list with the start_date and end_date of your intervention'
      raise TypeError(intervention_error)
    
    if len(model_params) < 1:
        model_parameters = {
                'changepoint_range': 0.85,
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'holidays': None,
                'seasonality_mode': 'additive',
                'changepoint_prior_scale': 0.05,
                'interval_width': 1 - alpha}
    else:
        model_parameters = model_params.copy()
    
    pre_intervention = [df.ds.min(), (pd.to_datetime(intervention[0]) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')]
    
    data, parameters = prophet_regression(
                                df = df, 
                                intervention = intervention, 
                                cross_validation_steps = cross_validation_steps, 
                                alpha = alpha, 
                                model_params = model_parameters, 
                                regressors = regressors,
                                verbose = verbose,
                                model_type = model_type,
                                autocorrelation = autocorrelation)

    data['cummulitive_y'] = data['y'].cumsum()
    data['cummulitive_yhat'] = data['yhat'].cumsum()

    data['point_effects'] = data['y'] - data['yhat']
    data['cummulitive_effect'] = data['point_effects'].cumsum()

    data['cummulitive_yhat_lower'] = data['yhat_lower'].cumsum()
    data['cummulitive_yhat_upper'] = data['yhat_upper'].cumsum()
    
    pre_int_metrics = [
    ['r2', r2_score(y_pred = data[(data.ds >= pd.to_datetime(pre_intervention[0]))&(data.ds <= pd.to_datetime(pre_intervention[1]))&(data.y > 0)].yhat, y_true = data[(data.ds >= pd.to_datetime(pre_intervention[0]))&(data.ds <= pd.to_datetime(pre_intervention[1]))&(data.y > 0)].y)],
    ['MAE', mean_absolute_error(y_pred = data[(data.ds >= pd.to_datetime(pre_intervention[0]))&(data.ds <= pd.to_datetime(pre_intervention[1]))&(data.y > 0)].yhat, y_true = data[(data.ds >= pd.to_datetime(pre_intervention[0]))&(data.ds <= pd.to_datetime(pre_intervention[1]))&(data.y > 0)].y)],
    ['MAPE (%)', mape(y_pred = data[(data.ds >= pd.to_datetime(pre_intervention[0]))&(data.ds <= pd.to_datetime(pre_intervention[1]))&(data.y > 0)].yhat, y_true = data[(data.ds >= pd.to_datetime(pre_intervention[0]))&(data.ds <= pd.to_datetime(pre_intervention[1]))&(data.y > 0)].y)],
    ['Noise (Std)', np.std(data[(data.ds >= pd.to_datetime(pre_intervention[0]))&(data.ds <= pd.to_datetime(pre_intervention[1]))&(data.y > 0)].y - data[(data.ds >= pd.to_datetime(pre_intervention[0]))&(data.ds <= pd.to_datetime(pre_intervention[1]))&(data.y > 0)].yhat)]
    ]

    strings_info = """
+------------------------+
 Pre intervention metrics
+------------------------+
{}
    """
    if verbose:
        print(
            strings_info.format(
                tabulate(pre_int_metrics, 
                headers=['Metric', 'Value'], 
                tablefmt='pipe')
            ).strip()
        )

    int_metrics = [
    ['Actual cumulative', data[(data.ds >= intervention[0]) & (data.ds <= intervention[1])].y.sum()],
    ['Predicted cumulative:', data[(data.ds >= intervention[0]) & (data.ds <= intervention[1])].yhat.sum()],
    ['Difference', data[(data.ds >= intervention[0]) & (data.ds <= intervention[1])].point_effects.sum()],
    ['Change (%)', (data[(data.ds >= intervention[0]) & (data.ds <= intervention[1])].y.sum() / data[(data.ds >= intervention[0]) & (data.ds <= intervention[1])].yhat.sum() -1)*100]
    ]
    
    return data, pre_int_metrics, int_metrics, parameters