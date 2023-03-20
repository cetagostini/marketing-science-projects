import pandas as pd

from tabulate import tabulate

def summary(data = None, statistical_significance = 0.05, 
            stadisticts = None, pre_int_metrics = None, 
            int_metrics = None, intervention = None, n_samples = None):
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
Considerations
--------------
a) The standard deviation of the residuals is {}. This means that the noise in your data is {}.
b) Based on this information, in order for your effect to be detectable, it should be greater than {}%.

Summary
-------
During the intervention period, the response variable had an average value of approximately {}. 
By contrast, in the absence of an intervention, we would have expected an average response of {}. 
The 95% confidence interval of this counterfactual prediction is {} to {}.

The usual error of your model is {}%, while the difference during the intervention period is {}%. 
During the intervention, the error increase {}% ({} percentage points), suggesting some factor is 
impacting the quality of the model, and the differences are significant.

The probability of obtaining this effect by chance is very small 
(after {} simulations, bootstrap probability p = {}). 
This means that the causal effect can be considered statistically significant.
        """
    else:
        summary = """
Considerations
--------------
a) The standard deviation of the residuals is {}. This means that the noise in your data is {}.
b) Based on this information, in order for your effect to be detectable, it should be greater than {}%.

Summary
-------
During the intervention period, the response variable had an average value of approximately {}. 
By contrast, in the absence of an intervention, we would have expected an average response of {}. 
The 95% confidence interval of this counterfactual prediction is {} to {}.

The usual error of your model is between ±{}%, while the difference during the intervention period is {}%. 
During the intervention, the error increase {}% ({} percentage points), suggesting that the model can explain well what should happen,
and that the differences are not significant.

The probability of obtaining this effect by chance is not small 
(after {} simulations, bootstrap probability p = {}). 
This means that the causal effect cannot be considered statistically significant.
        """

    print(
        summary.format(
            round(std_res,5),
            noise,
            mde,
            round(data[(data.ds >= pd.to_datetime(intervention[0])) & (data.ds <= pd.to_datetime(intervention[1]))].y.mean(),2),
            round(data[(data.ds >= pd.to_datetime(intervention[0])) & (data.ds <= pd.to_datetime(intervention[1]))].yhat.mean(),2),
            round(data[(data.ds >= intervention[0]) & (data.ds <= intervention[1])].yhat_lower.mean(),2), 
            round(data[(data.ds >= intervention[0]) & (data.ds <= intervention[1])].yhat_upper.mean(),2),
            abs(round(pre_int_metrics[2][1],2)),
            abs(round(int_metrics[3][1],2)),
            (1 - round(abs(round(int_metrics[3][1], 2)) / abs(round(pre_int_metrics[2][1], 2)), 2)) * 100,
            round(round(abs(int_metrics[3][1]),2) - abs(round(pre_int_metrics[2][1],2)),2),
            n_samples,
            round(stadisticts[0],5)
        )
    )
    
def summary_intervention(data, intervention = None, int_metrics = None):
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
    lower = data[(data.ds >= intervention[0]) & (data.ds <= intervention[1])].yhat_lower.sum()
    upper = data[(data.ds >= intervention[0]) & (data.ds <= intervention[1])].yhat_upper.sum()

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
            'CI 95%: [{}, {}]'.format(
                round(lower, 0), 
                round(upper, 0)
            )
        ).strip()
    )