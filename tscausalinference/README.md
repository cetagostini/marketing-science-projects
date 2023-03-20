# tscausalinference

`tscausalinference` is a Python library for performing causal inference analysis over time series data. It uses the counterfactual methodology on top of the Prophet time-series forecasting library, with the help of Bootstrap simulations method for statistical significance testing and to manage uncertainty.

## How it works

Causal inference is a family of statistical methods used to determine the cause of changes in one variable if the changes occur in a different variable. The tscausalinference library creates synthetic control groups (forecast response) to determine the impact of a real treatment group (actual response). By defining these two groups, the library calculates the counterfactual result (difference between the groups) and determines its statistical significance.

The Prophet model is used to generate control data by making predictions about what would have happened in the absence of the intervention. This control data represents a counterfactual scenario, where the intervention did not occur, and allows us to compare the actual outcomes to what would have happened if the intervention had not been implemented. 

To manage the uncertainty of the prediction and to test the statistical significance of the results, the `tscausalinference` library performs Monte Carlo simulations through the Bootstrap method. This method involves resampling a single dataset to create many simulated datasets. The library creates a set of alternative random walks from the synthetic control data, and builds a distribution of each mean from the complete period of the simulation. 

The real mean of the period is then compared against this distribution, and we calculate how extreme this mean value is based on the distribution of **what should happen** using the cumulative distribution function (CDF). This helps us to determine the statistical significance of the effect.

The library works as follows:

1. Build a Prophet model to create a synthetic control group.
2. Perform Monte Carlo simulations using the Bootstrap method to create a set of alternative random walks from the synthetic control data.
3. Build a distribution of each mean from the complete period of the simulation.
4. Compare the real mean of the period against this distribution using the CDF to determine the statistical significance of the effect.

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
- `seasonality_mode`: optional string to be included in the Prophet model, default 'additive'

```python
from tscausalinference import tscausalinference as tsci
import pandas as pd

# Load data
df = pd.read_csv('mydata.csv')
intervention = ['2022-07-04', '2022-07-19']

data = tsci(data = df, intervention = intervention)

data.plot()
```
![plot_intervention Method](https://github.com/carlangastr/marketing-science-projects/blob/main/tscausalinference/introduction_notebooks/plots/output.png)

### Create your own data
```python
from tscausalinference import synth_dataframe

synth = synth_dataframe() #Create fake data using our custom function.
df = synth.DataFrame()
```
```md
Min date: 2022-01-01 00:00:00
Max date: 2022-12-31 00:00:00
Day where effect was injected: 2022-12-17 00:00:00
Power of the effect: 30.0%
```
**Expected Schema**
```md
<class 'pandas.core.frame.DataFrame'>
Int64Index: 365 entries, 0 to 364
Data columns (total 4 columns):
 #   Column      Non-Null Count  Dtype         
---  ------      --------------  -----         
 0   ds          365 non-null    datetime64[ns]
 1   y           365 non-null    float64       
 2   regressor1  365 non-null    float64       
 3   regressor2  365 non-null    float64       
dtypes: datetime64[ns](1), float64(3)
memory usage: 14.3 KB
```
### Checking seasonal decomposition
```python
data = tsci(data = df, intervention = intervention)

data.plot(method = 'decompose')
```
![seasonal_decompose Method](https://github.com/carlangastr/marketing-science-projects/blob/main/tscausalinference/introduction_notebooks/plots/seasonal_decompose.png)

### Check the sensibility of your time series
You can validate before run your experiment how big should be the effect in order to be catch it.

```python
from tscausalinference import sensitivity

pre_check = sensitivity(
    df=df,
    test_period=['2023-12-25', '2024-01-05'],
    cross_validation_steps=10,
    alpha=0.05,
    model_params={'changepoint_range': [0.85,0.50],
                  'changepoint_prior_scale': [0.05, 0.50]},
    regressors=['regressor2', 'regressor1'],
    verbose=True,
    n_samples=1000)
```
![sensitivity Method](https://github.com/carlangastr/marketing-science-projects/blob/main/tscausalinference/introduction_notebooks/plots/sensitivity.png)

### Checking p-value, simulations & distributions
```python
data = tsci(data = df, intervention = intervention)

data.plot(method = 'simulations')
```
![plot_simulations Method](https://github.com/carlangastr/marketing-science-projects/blob/main/tscausalinference/introduction_notebooks/plots/pvalue.png)

### Customizing model
```python
data = tsci(data = df, intervention = intervention, regressors = ['a','b'],
                        alpha = 0.03, n_samples = 10000, cross_validation_steps = 15
                        )

data.summarization()
```
```md
    Considerations
    --------------
    a) The standard deviation of the residuals is 0.12762. This means that the noise in your data is low.
    b) Based on this information, in order for your effect to be detectable, it should be greater than 15%.

    Summary
    -------
    During the intervention period, the response variable had an average value of approximately 21.78. 
    By contrast, in the absence of an intervention, we would have expected an average response of 19.15. 
    The 95% confidence interval of this counterfactual prediction is 18.9 to 19.39.

    The usual error of your model is 2.23%, while the difference during the intervention period is 13.77%. 
    During the intervention, the error increase 6.18% (11.54 percentage points), suggesting some factor is impacting the quality of the model, and the differences are significant.

    The probability of obtaining this effect by chance is very small 
    (after 10000 simulations, bootstrap probability p = 0.002). 
    This means that the causal effect can be considered statistically significant.
```

## Extra documentation
1. Check out [Pypi](https://pypi.org/project/tscausalinference) for more information.
2. Check out [Introduction Notebooks](https://github.com/carlangastr/marketing-science-projects/blob/main/tscausalinference/introduction_notebooks/basic.ipynb) to see an example of usage.
3. Check on [Google colab](https://colab.research.google.com/drive/1OeF0OLlu0d9oM0xWiJSIEnxN-eWyFskq#scrollTo=YTX8Z4ZOwpyw) 🔥.

## Inspirational articles:
1. [Bootstrap random walks](https://reader.elsevier.com/reader/sd/pii/S0304414915300247?token=0E54369709F75136F10874CA9318FB348A6B9ED117081D7607994EDB862C09E8F95AE336C38CD97AD7A2C50FF14A8708&originRegion=eu-west-1&originCreation=20230224195555)
2. [Public Libs](https://github.com/lytics/impact)
3. [A Nonparametric approach for multiple change point analysis](https://arxiv.org/pdf/1306.4933.pdf)
4. [Causal Impact on python](https://www.youtube.com/watch?v=GTgZfCltMm8&t=272s)
5. [Causal Inference Using Bayesian Structural Time-Series Models](https://towardsdatascience.com/causal-inference-using-bayesian-structural-time-series-models-ab1a3da45cd0)
6. [Wikipedia](https://en.wikipedia.org/wiki/Random_walk)


## License
This project is licensed under the MIT License - see the LICENSE file for details.
