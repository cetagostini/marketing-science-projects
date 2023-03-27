import pandas as pd
import numpy as np
import datetime

def weekly_season(intervention = [], df = pd.DataFrame()):
    start_date = datetime.datetime.strptime(intervention[0], '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(intervention[1], '%Y-%m-%d').date()

    prophet_predict = df.copy()
    prophet_predict['weekday_name'] = pd.to_datetime(prophet_predict['ds']).dt.strftime("%A")

    condition = (prophet_predict.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=60))&(prophet_predict.ds < pd.to_datetime(intervention[0]))

    units = prophet_predict[condition].groupby('weekday_name').median().weekly
    trends = prophet_predict[condition].groupby('weekday_name').mean().trend

    day_percentages = abs(units / units.abs().sum() - 1)
    trend_percentages = abs(trends / trends.abs().sum())

    df_pctg = pd.merge(pd.DataFrame(day_percentages, index=day_percentages.index).reset_index(),
                        pd.DataFrame(trend_percentages, index=trend_percentages.index).reset_index(),
                        on='weekday_name')

    period = (end_date - start_date).days + 1

    dates = pd.date_range(intervention[0], periods = period, freq = 'D')

    df = pd.DataFrame({'date': dates})
    df['weekday_name'] = pd.to_datetime(df['date']).dt.strftime('%A')

    df = pd.merge(df, df_pctg, on='weekday_name')

    seasonality = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))].sort_values('date')['weekly'].values
    trend = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))].sort_values('date')['trend'].values
    
    return np.array(seasonality), np.array(trend)

def yearly_season(intervention = [], df = pd.DataFrame()):
    start_date = datetime.datetime.strptime(intervention[0], '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(intervention[1], '%Y-%m-%d').date()

    prophet_predict = df.copy()
    prophet_predict['year_name'] = pd.to_datetime(prophet_predict['ds']).dt.strftime("%Y")

    condition = (prophet_predict.ds >= pd.to_datetime(intervention[0]) - pd.Timedelta(days=60))&(prophet_predict.ds < pd.to_datetime(intervention[0]))

    units = prophet_predict[condition].groupby('year_name').median().yearly
    trends = prophet_predict[condition].groupby('year_name').mean().trend

    day_percentages = abs(units / units.abs().sum() - 1)
    trend_percentages = abs(trends / trends.abs().sum())

    df_pctg = pd.merge(pd.DataFrame(day_percentages, index=day_percentages.index).reset_index(),
                        pd.DataFrame(trend_percentages, index=trend_percentages.index).reset_index(),
                        on='year_name')
    
    period = (end_date - start_date).days + 1
    
    dates = pd.date_range(intervention[0], periods = period, freq = 'D')
    

    df = pd.DataFrame({'date': dates})
    df['year_name'] = pd.to_datetime(df['date']).dt.strftime('%Y')
    
    df = pd.merge(df, df_pctg, on='year_name')

    seasonality = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))].sort_values('date')['yearly'].values
    trend = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))].sort_values('date')['trend'].values
    
    return np.array(seasonality), np.array(trend)