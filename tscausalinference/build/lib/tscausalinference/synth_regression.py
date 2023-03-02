from prophet import Prophet

from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.utilities import regressor_coefficients

from tabulate import tabulate
from sklearn.metrics import r2_score, mean_absolute_error

import pandas as pd
from pandas import DataFrame

import numpy as np

import logging

from tscausalinference.evaluators import mape

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

logging.getLogger('fbprophet').setLevel(logging.ERROR)

pd.options.mode.chained_assignment = None 

def synth_analysis(df: DataFrame = None, 
                    regressors: list = [], 
                    intervention: list = None, 
                    seasonality: bool = True,
                    cross_validation_steps: int = 5,
                    alpha: float = 0.05,
                    model_mode: str = 'additive'
                    ):
    """
    A function to analyze a dataset using the Prophet library and provide evaluation metrics, response metrics and 
    intervention metrics. 

    Args:
    ------
    df : DataFrame, default None
        A pandas DataFrame with two columns 'ds' and 'y', containing the date and the corresponding target variable.
    regressors : list, default []
        List of column names in df to be used as regressors for the Prophet model.
    intervention : list, default None
        A list with two values, containing the start and end date for the intervention period, as a string in the 
        format 'YYYY-MM-DD'.
    seasonality : bool, default True
        Whether or not to include seasonality in the Prophet model.
    cross_validation_steps : int, default 5
        The number of days to use for cross-validation.
    
    Returns:
    -------
    data : DataFrame
    prints :
        - Cross-validation MAPE
        - Regressor coefficients (if any)
        - Pre-intervention response metrics (r2, MAE, and MAPE)
        - Intervention metrics (actual and predicted cumulative, and difference)

    Raises:
    -------
    ValueError: 
        If df is not a DataFrame, or does not contain columns named 'ds' and 'y'.
        If intervention is not a list with two values.
        If intervention period is not found in df.
    TypeError: 
        If df is None.
    KeyError:
        If any regressor in the list is not a column in df.

    Examples:
    --------
    >>> synth_analysis(df, ['regressor_1', 'regressor_2'], ['2021-01-01', '2021-02-01'], False, 10)
    Training period: 2019-01-01 to 2021-01-01
    Test period: 2021-01-01 to 2021-02-01

    Cross-validation MAPE: 5.83%
    +---------------+------------------+
    | Regressor     |   Coefficient    |
    +===============+==================+
    | regressor_1   |          0.293   |
    +---------------+------------------+
    | regressor_2   |         -0.0133  |
    +---------------+------------------+

    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    elif not set(['ds', 'y']).issubset(df.columns):
        raise ValueError("df must contain columns named 'ds' and 'y'")
    elif df.empty:
        raise TypeError("df is empty")

    if isinstance(intervention, type(None)):
      intervention_error = 'Define your intervention as a list with the start_date and end_date of your intervention'
      raise intervention_error
    
    pre_intervention = [df.ds.min(),(pd.to_datetime(intervention[0]) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')]
    post_intervention = [(pd.to_datetime(intervention[0]) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),df.ds.max()]

    training_dataframe = df[(df.ds > pd.to_datetime(pre_intervention[0]))&(df.ds <= pd.to_datetime(pre_intervention[1]))&(df.y > 0)].fillna(0).copy()
    training_dataframe['ds'] = pd.to_datetime(training_dataframe['ds'])

    test_dataset = df[(df.ds > pd.to_datetime(pre_intervention[0]))&(df.ds <= pd.to_datetime(post_intervention[1]))&(df.y > 0)].fillna(0).copy()
    test_dataset['ds'] = pd.to_datetime(test_dataset['ds'])

    prediction_period = len(test_dataset[(test_dataset.ds > pd.to_datetime(intervention[0]))&(test_dataset.ds <= pd.to_datetime(intervention[1]))].index)

    print('Training period: {} to {}'.format(pre_intervention[0], pre_intervention[1]))
    print('Test period: {} to {}\n'.format(intervention[0], intervention[1]))
    print('Prediction horizon: {} days'.format(prediction_period))

    if seasonality:
        prophet = Prophet(daily_seasonality=True, 
                    weekly_seasonality=True, 
                    changepoint_prior_scale = 0.05, 
                    changepoint_range = 0.85,
                    interval_width= 1 - alpha,
                    seasonality_mode = model_mode)
    else:
        prophet = Prophet(
                        changepoint_prior_scale = 0.05, 
                        changepoint_range = 0.85,
                        interval_width= 1 - alpha,
                        seasonality_mode = model_mode)

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

    data['cummulitive_y'] = data['y'].cumsum()
    data['cummulitive_yhat'] = data['yhat'].cumsum()

    data['point_effects'] = data['y'] - data['yhat']
    data['cummulitive_effect'] = data['point_effects'].cumsum()

    data['cummulitive_yhat_lower'] = data['yhat_lower'].cumsum()
    data['cummulitive_yhat_upper'] = data['yhat_upper'].cumsum()

    # print response
    print(f'\nCross-validation MAPE: {model_mape_mean:.2%}')
    if len(regressors) >= 1:
        regressor_df = regressor_coefficients(prophet)
        # format table
        table = []
        for index, row in regressor_df.iterrows():
            table.append([index, *list(row.values)])

        print(tabulate(table, headers=['Regressor', 'Regressor Mode', 'Center', 'Coef. Lower', 'Coef', 'Coef. Upper'], tablefmt='grid'))

    print('\nSeasons detected: {}'.format(list(prophet.seasonalities.keys())))
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
    
    return data, pre_int_metrics, int_metrics