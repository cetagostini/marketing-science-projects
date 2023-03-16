from prophet import Prophet

from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.utilities import regressor_coefficients

from tabulate import tabulate
from sklearn.metrics import r2_score, mean_absolute_error

import pandas as pd
from pandas import DataFrame

import numpy as np

import itertools

import logging

from tscausalinference.evaluators import mape

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

logging.getLogger('fbprophet').setLevel(logging.ERROR)

pd.options.mode.chained_assignment = None 

def sensitivity_analysis(df: DataFrame = pd.DataFrame(), 
                         training_period = None, 
                         cross_validation_steps: int = 5, 
                         alpha: float = 0.05, 
                         model_params: dict = {}, 
                         regressors: list = []):

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
                'mcmc_samples': 1000,
                'interval_width': 1 - alpha}
        
        print('Default parameters grid: \n{}',format(model_parameters))
    else:
        model_parameters = model_params.copy()
    
    start_date = (pd.to_datetime(training_period[0])).strftime('%Y-%m-%d')
    end_date = (pd.to_datetime(training_period[1])).strftime('%Y-%m-%d')

    training_dataframe = df[(df.ds >= pd.to_datetime(start_date))&(df.ds <= pd.to_datetime(end_date))&(df.y > 0)].fillna(0).copy()
    training_dataframe['ds'] = pd.to_datetime(training_dataframe['ds'])

    test_dataset = df[(df.ds >= pd.to_datetime(start_date)) & (df.y > 0)].fillna(0).copy()
    test_dataset['ds'] = pd.to_datetime(test_dataset['ds'])

    prediction_period = len(test_dataset[(test_dataset.ds > pd.to_datetime(end_date))&(test_dataset.ds <= test_dataset.ds.max())].index)

    print('Training period: {} to {}'.format(training_period[0], training_period[1]))
    print('Test period: {} to {}\n'.format((pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), test_dataset.ds.max()))
    print('Prediction horizon: {} days'.format(prediction_period))
    
    condition_int = isinstance(model_parameters[list(model_parameters.keys())[0]], float)
    condition_float = isinstance(model_parameters[list(model_parameters.keys())[0]], int)
    condition_str = isinstance(model_parameters[list(model_parameters.keys())[0]], str)
    
    if not (condition_int)|(condition_float)|(condition_str):
        print('Grid Search Cross-Validation mode:\n')
        if isinstance(model_parameters[list(model_parameters.keys())[0]], list):
            # Generate all combinations of parameters
            all_params = [dict(zip(model_parameters.keys(), v)) for v in itertools.product(*model_parameters.values())]
            rmses = []  # Store the RMSEs for each params here

            # Use cross validation to evaluate all parameters
            for params in all_params:
                m = Prophet(**params).fit(training_dataframe)  # Fit model with given params
                df_cv = cross_validation(m, '{} days'.format(cross_validation_steps), disable_tqdm=False , parallel="processes")
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmses.append(df_p['rmse'].mean())

            # Find the best parameters
            tuning_results = pd.DataFrame(all_params)
            tuning_results['rmse'] = rmses
            # Python
            best_params = all_params[np.argmin(rmses)]
            print(best_params)
            
            prophet = Prophet(**best_params)
        else:
            raise TypeError("Your parameters on the Grid are not list type")
    
    else:
        model_parameters.update({'interval_width': 1 - alpha})
        print('Custom parameters grid: \n{}',format(model_parameters))
        prophet = Prophet(**model_parameters)

    for regressor in regressors:
            prophet.add_regressor(name = regressor)
    
    prophet.fit(training_dataframe)

    prophet_predict = prophet.predict(test_dataset)

    df_cv = cross_validation(prophet, horizon = '{} days'.format(cross_validation_steps), disable_tqdm=False)
    df_p = performance_metrics(df_cv)

    model_mape_mean = df_p.mape.mean()

    df['ds'] = pd.to_datetime(df['ds'])

    data = pd.merge(
        prophet_predict[['ds','yhat', 'yhat_lower', 'yhat_upper', 'trend']+list(prophet.seasonalities.keys())], 
        df[["ds", "y"]], how='left', on='ds'
        )

    data['yhat'] = data['yhat'].astype(float)
    data['y'] = data['y'].astype(float)

    data['residuals'] = data['y'] - data['yhat']

    # print response
    print(f'\nCross-validation MAPE: {model_mape_mean:.2%}')
    print('\nSeasons detected: {}'.format(list(prophet.seasonalities.keys())))
    if len(regressors) >= 1:
        regressor_df = regressor_coefficients(prophet)
        # format table
        table = []
        for index, row in regressor_df.iterrows():
            table.append([index, *list(row.values)])

        print(tabulate(table, headers=['Regressor', 'Regressor Mode', 'Center', 'Coef. Lower', 'Coef', 'Coef. Upper'], tablefmt='grid'))
    
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

    print(
        strings_info.format(
            tabulate(test_results, 
            headers=['Metric', 'Value'], 
            tablefmt='pipe')
        ).strip()
    )
    
    return data, training_results, test_results