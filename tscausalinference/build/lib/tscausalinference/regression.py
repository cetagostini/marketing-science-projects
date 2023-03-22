from prophet import Prophet

from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.utilities import regressor_coefficients

from tabulate import tabulate

import pandas as pd
from pandas import DataFrame

import numpy as np

import itertools

from tscausalinference.evaluators import mape

def prophet_regression(df: DataFrame = pd.DataFrame(), 
                         intervention = None, 
                         cross_validation_steps: int = 5, 
                         alpha: float = 0.05, 
                         model_params: dict = {}, 
                         regressors: list = [],
                         verbose = True,
                         model_type = 'gam',
                         autocorrelation = False):

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
        
        print('Default parameters grid: \n{}',format(model_parameters))
    else:
        model_parameters = model_params.copy()
    
    df['y_lag7'] = df.y.shift(7)
    df['y_lag14'] = df.y.shift(14)
    
    if autocorrelation:
        regressors.append('y_lag7')
        regressors.append('y_lag14')
    
    pre_intervention = [df.ds.min(), (pd.to_datetime(intervention[0]) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')]
    post_intervention = [(pd.to_datetime(intervention[1]) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), df.ds.max()]

    training_dataframe = df[(df.ds >= pd.to_datetime(pre_intervention[0]))&(df.ds <= pd.to_datetime(pre_intervention[1]))&(df.y > 0)].fillna(0).copy()
    training_dataframe['ds'] = pd.to_datetime(training_dataframe['ds'])

    test_dataset = df[(df.ds > pd.to_datetime(pre_intervention[0]))&(df.ds <= pd.to_datetime(post_intervention[1]))&(df.y > 0)].fillna(0).copy()
    test_dataset['ds'] = pd.to_datetime(test_dataset['ds'])

    prediction_period = len(test_dataset[(test_dataset.ds >= pd.to_datetime(intervention[0]))&(test_dataset.ds <= pd.to_datetime(intervention[1]))].index)

    print('Training period: {} to {}'.format(pre_intervention[0], pre_intervention[1]))
    print('Test period: {} to {}\n'.format(intervention[0], intervention[1]))
    print('Post period: {} to {}\n'.format(post_intervention[0], post_intervention[1]))
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
            print('Total parameters combinations: {}'.format(len(all_params)))

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
            
            model_parameters = best_params.copy()
            prophet = Prophet(**model_parameters)
        else:
            raise TypeError("Your parameters on the Grid are not list type")
    
    else:
        model_parameters.update({'interval_width': 1 - alpha})
        print('Custom parameters grid: \n{}'.format(model_parameters))
        if model_type == 'bayesian':
            model_parameters.update({'mcmc_samples': 1000})
            prophet = Prophet(**model_parameters)
            
        else:
            prophet = Prophet(**model_parameters)

    for regressor in regressors:
            prophet.add_regressor(name = regressor)
    
    prophet.fit(training_dataframe)

    prophet_predict = prophet.predict(test_dataset)

    df_cv = cross_validation(prophet, horizon = '{} days'.format(cross_validation_steps),disable_tqdm=False)
    df_p = performance_metrics(df_cv)

    model_mape_mean = df_p.mape.mean()

    df['ds'] = pd.to_datetime(df['ds'])

    data = pd.merge(
        prophet_predict[['ds','yhat', 'yhat_lower', 'yhat_upper', 'trend']+list(prophet.seasonalities.keys())], 
        df[["ds", "y"]], how='left', on='ds'
        )

    data['yhat'] = data['yhat'].astype(float)
    data['y'] = data['y'].astype(float)

    # print response
    if verbose:
        print(f'\nCross-validation MAPE: {model_mape_mean:.2%}')
        print('\nSeasons detected: {}'.format(list(prophet.seasonalities.keys())))
    
    if len(regressors) >= 1:
        regressor_df = regressor_coefficients(prophet)
        # format table
        table = []
        for index, row in regressor_df.iterrows():
            table.append([index, *list(row.values)])
        if verbose:
            print(tabulate(table, headers=['Regressor', 'Regressor Mode', 'Center', 'Coef. Lower', 'Coef', 'Coef. Upper'], tablefmt='grid'))    

    return data, model_parameters
