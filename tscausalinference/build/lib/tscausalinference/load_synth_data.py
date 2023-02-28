import pandas as pd
import numpy as np

def create_synth_dataframe(n: int = 365, 
                           trend: float = 0.1, 
                           seasonality: int = 7, 
                           simulated_effect: float = 0.15, 
                           eff_n: int = 15, 
                           noise_power: float = 0.2, 
                           regressor: int = 2) -> pd.DataFrame:
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

    Returns:
    - df (pd.DataFrame): A pandas dataframe with columns 'ds', 'y' and the regressors created.

    Example:
    - create_synth_dataframe(n=365, trend=0.1, seasonality=7, simulated_effect=1.2, eff_n=30, noise_power=0.1, regressor=3)
    """

    # Create a time index
    control_index = pd.date_range('2022-01-01', periods=n, freq='D')
    # Create a time index
    treatment_index = pd.date_range(str(control_index.max()- pd.Timedelta(days=eff_n-1)), periods=eff_n, freq='D')

    # Create the second time series
    trend_component = np.arange(start=1, stop = n+1)* trend
    seasonality_component = np.cos(np.arange(start=1, stop = n+1) * 2 * np.pi / seasonality)
    data_control = trend_component + seasonality_component + np.random.normal(scale=noise_power, size=n)

    # Create the first time series
    data_treatment = data_control[-len(treatment_index):] * simulated_effect

    df = pd.merge(
        pd.DataFrame({'control':data_control, 'ds':control_index}),
        pd.DataFrame({'treatment':data_treatment, 'ds':treatment_index}),
        on = 'ds',
        how = 'left'
        ).fillna(0)

    df['y'] = df.control  + df.treatment

    # Create regressors
    for i in range(regressor):
        trend_component = np.arange(start=1, stop = n+1) * trend//2
        seasonality_component = np.cos(np.arange(start=1, stop = n+1) * 2 * np.pi / seasonality//int(np.random.randint(1,11)))

        regressor_data = trend_component + seasonality_component + np.random.normal(scale=noise_power, size=len(df.index))
        
        df[f'regressor{i+1}'] = regressor_data

    cols = ['ds', 'y'] + [f'regressor{i+1}' for i in range(regressor)]
    df['ds'] = pd.to_datetime(df.ds)

    # Print min and max date
    print(f"Min date: {df['ds'].min()}\nMax date: {df['ds'].max()}")

    # Print day where effect was injected
    print(f"Day where effect was injected: {df[df['treatment'] != 0]['ds'].min()}")

    #Effect injected
    print(f"Power of the effect: {simulated_effect*100}%")

    return df[cols]
