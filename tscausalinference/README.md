# tscausalinference

`tscausalinference` is a Python library for performing causal inference analysis over time series data. It uses the counterfactual methodology on top of the Prophet time-series forecasting library, with the help of Bootstrap simulations method for statistical significance testing and manage uncertainty.

## How it works

Causal inference is a family of statistical methods used to determine the cause of changes in one variable if the changes occur in a different variable. The tscausalinference library creates synthetic control groups (forecast response) to determine the impact of a real treatment group (actual response). By defining these two groups, the library calculates the counterfactual result (difference between the groups) and determines its statistical significance using the same A/B testing methodology.

The Prophet model is used to generate control data by making predictions about what would have happened in the absence of the intervention. This control data represents a counterfactual scenario, where the intervention did not occur, and allows us to compare the actual outcomes to what would have happened if the intervention had not been implemented. Bootstrap simulations are performed to estimate the sampling distribution of the effect and to test its statistical significance.

The library works as follows:

1. Build a Prophet model.
2. Generate control data.
3. Perform Bootstrap simulations.
4. Calculate p-values using the Bootstrap simulations.

## Why Prophet?

Prophet is a time-series forecasting library in Python that uses statistical models to make predictions about future values based on past trends and regressors. It takes into account seasonal trends, holiday effects, and other factors that can affect the outcome variable. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well. Additionally, Prophet is a simple and scalable framework that is well-documented and supported by its own community.

## Why Bootstrap?

Bootstrap is a statistical procedure that involves resampling a single dataset to create many simulated datasets. This process allows us to calculate standard errors, construct confidence intervals, and perform hypothesis testing for numerous types of sample statistics. Bootstrap is a useful method for estimating the effect of an intervention because it can help us detect significant changes in the mean or variance of a time series. One of the main challenges in time series analysis is that we often have a limited amount of data, especially when studying the effects of a specific intervention. Bootstrap is a non-parametric method that does not require any assumptions about the underlying distribution of the data, making it a flexible method that can be applied in a wide range of situations.

## Installation

tscausalinference can be installed using pip:

```python
!pip install tscausalinference
```

## Example Usage

The `tscausalinference` function takes the following arguments:

- `data`: the time series data as a Pandas DataFrame
- `intervention`: the time period of the intervention as a tuple of start and end dates
- `regressors`: optional list of regressors to be included in the Prophet model
- `seasonality`: boolean indicating whether to include seasonality in the Prophet model
- `cross_validation_steps`: number of steps to use in cross-validation for Prophet model tuning

```python
import tscausalinference as tsci
import pandas as pd

# Load data
df = pd.read_csv('mydata.csv')

data = tscausalinference(data = df, 
                    intervention = intervention, 
                    regressors=[],
                    seasonality = True,
                    cross_validation_steps = 6)

data.plot_intervention()
```

License
This project is licensed under the MIT License - see the LICENSE file for details.