from tabulate import tabulate
from sklearn.metrics import r2_score, mean_absolute_error

import pandas as pd
from pandas import DataFrame

import numpy as np
import logging

from tscausalinference.evaluators import mape
from tscausalinference.regression import prophet_regression
from tscausalinference.bootstrap import bootstrap_simulate, bootstrap_p_value
from tscausalinference.synth_regression import synth_analysis

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

logging.getLogger('fbprophet').setLevel(logging.ERROR)

pd.options.mode.chained_assignment = None 

def training_model(df: DataFrame = pd.DataFrame(), 
                         training_period = None, 
                         cross_validation_steps: int = 5, 
                         alpha: float = 0.05, 
                         model_params: dict = {}, 
                         regressors: list = [],
                         verbose: bool = True,
                         model_type = 'gam'):

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    elif not set(['ds', 'y']).issubset(df.columns):
        raise ValueError("df must contain columns named 'ds' and 'y'")
    elif df.empty:
        raise TypeError("df is empty")

    if isinstance(training_period, type(None)):
      training_period_error = 'Define your intervention as a list with the start_date and end_date of your TRAINING PERIOD'
      raise TypeError(training_period_error)
    
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
        
        print('Default parameters grid: \n{}',format(model_parameters))
    else:
        model_parameters = model_params.copy()
    
    start_date = (pd.to_datetime(training_period[0])).strftime('%Y-%m-%d')
    end_date = (pd.to_datetime(training_period[1])).strftime('%Y-%m-%d')
    
    data, paremeters = prophet_regression(
            df = df, 
            intervention = training_period, 
            cross_validation_steps = cross_validation_steps, 
            alpha = alpha, 
            model_params = model_parameters, 
            regressors = regressors,
            verbose = verbose,
            model_type = model_type)

    data['residuals'] = data['y'] - data['yhat']
    
    training_results = [
    ['r2', r2_score(y_pred = data[(data.ds >= pd.to_datetime(start_date))&(data.ds <= pd.to_datetime(end_date))&(data.y > 0)].yhat, y_true = data[(data.ds >= pd.to_datetime(start_date))&(data.ds <= pd.to_datetime(end_date))&(data.y > 0)].y)],
    ['MAE', mean_absolute_error(y_pred = data[(data.ds >= pd.to_datetime(start_date))&(data.ds <= pd.to_datetime(end_date))&(data.y > 0)].yhat, y_true = data[(data.ds >= pd.to_datetime(start_date))&(data.ds <= pd.to_datetime(end_date))&(data.y > 0)].y)],
    ['MAPE (%)', mape(y_pred = data[(data.ds >= pd.to_datetime(start_date))&(data.ds <= pd.to_datetime(end_date))&(data.y > 0)].yhat, y_true = data[(data.ds >= pd.to_datetime(start_date))&(data.ds <= pd.to_datetime(end_date))&(data.y > 0)].y)],
    ['Noise (Std)', np.std(data[(data.ds >= pd.to_datetime(start_date))&(data.ds <= pd.to_datetime(end_date))&(data.y > 0)].y - data[(data.ds >= pd.to_datetime(start_date))&(data.ds <= pd.to_datetime(end_date))&(data.y > 0)].yhat)]
    ]

    strings_info = """
+------------------------+
     TRAINING METRICS
+------------------------+
{}
    """

    if verbose:
        print(
            strings_info.format(
                tabulate(training_results, 
                headers=['Metric', 'Value'], 
                tablefmt='pipe')
            ).strip()
        )

    test_results = [
    ['r2', r2_score(y_pred = data[(data.ds > pd.to_datetime(end_date))&(data.y > 0)].yhat, y_true = data[(data.ds > pd.to_datetime(end_date))&(data.y > 0)].y)],
    ['MAE', mean_absolute_error(y_pred = data[(data.ds > pd.to_datetime(end_date))&(data.y > 0)].yhat, y_true = data[(data.ds > pd.to_datetime(end_date))&(data.y > 0)].y)],
    ['MAPE (%)', mape(y_pred = data[(data.ds > pd.to_datetime(end_date))&(data.y > 0)].yhat, y_true = data[(data.ds > pd.to_datetime(end_date))&(data.y > 0)].y)],
    ['Noise (Std)', np.std(data[(data.ds > pd.to_datetime(end_date))&(data.y > 0)].y - data[(data.ds > pd.to_datetime(end_date))&(data.y > 0)].yhat)]
    ]

    strings_info = """
+------------------------+
     TEST METRICS
+------------------------+
{}
    """
    if verbose:
        print(
            strings_info.format(
                tabulate(test_results, 
                headers=['Metric', 'Value'], 
                tablefmt='pipe')
            ).strip()
        )
    
    return data, training_results, test_results, paremeters

def sensitivity_analysis(df: DataFrame = pd.DataFrame(), 
                         test_period = None, 
                         cross_validation_steps: int = 5, 
                         alpha: float = 0.05, 
                         model_params: dict = {}, 
                         regressors: list = [],
                         verbose: bool = False,
                         n_samples = 1000,
                         model_type = 'gam'):
        
        df_temp = df.copy()
        
        data, training, test, model_parameters = training_model(
            df = df_temp, 
            regressors = regressors, 
            training_period = test_period, 
            cross_validation_steps = cross_validation_steps,
            alpha = alpha,
            model_params = model_params,
            verbose = verbose,
            model_type = model_type
            )
        
        effects = np.linspace(1.0, 2.0, 30)
        e_dataframe = pd.DataFrame()

        for effect in effects:
            temp_test = data.copy()

            # create a boolean mask for the rows in the test period
            test_mask = (temp_test['ds'] >= pd.to_datetime(test_period[0])) & (temp_test['ds'] <= pd.to_datetime(test_period[1]))
            
            #mask last 90 days
            rolling_mask = (temp_test['ds'] >= (pd.to_datetime(test_period[0]) - pd.Timedelta(days = 89))) & (temp_test['ds'] < pd.to_datetime(test_period[0]))
            
            # multiply 'y' by 1.1 in the test period
            temp_test.loc[test_mask, 'y'] *= effect
            
            simulations = bootstrap_simulate(
                    data = temp_test[test_mask].yhat, 
                    n_samples = n_samples, 
                    n_steps = len(temp_test[test_mask].index)
                    )
            
            stadisticts, stats_ranges, samples_means = bootstrap_p_value(control = temp_test[test_mask].yhat, 
                                                                                        treatment = temp_test[test_mask].y, 
                                                                                        simulations = simulations,
                                                                                        mape = abs(round(test[2][1],6))/100
                                                                                        )
            
            results_df = pd.DataFrame({'injected_effect': [round(effect, 2)],
                           'model': [model_parameters],
                           'pvalue': [stadisticts[0]],
                           'train': [training], 
                           'test': [test],
                           'ci_lower': [temp_test.yhat_lower.sum()],
                           'ci_upper': [temp_test.yhat_upper.sum()],
                           'y_intervention': [temp_test[test_mask].y.sum()],
                           'y_intervention_mean': [temp_test[test_mask].y.mean()],
                           'y_last90days_mean': [temp_test[rolling_mask].y.mean()],
                           'y_historical_mean': [temp_test[temp_test['ds'] < pd.to_datetime(test_period[0])].y.mean()]})
            
            e_dataframe = pd.concat([e_dataframe, results_df])

        return e_dataframe.set_index('injected_effect'), model_parameters