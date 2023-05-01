import numpy as np
import pandas as pd
import scipy
import collections
import h5py
from itertools import groupby


import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from statsmodels.stats.proportion import proportion_confint



def groupsize_analysis(events_gs):
    """
    Conducts a group size analysis on a dataset of events involving collective action.

    Parameters:
    -----------
    events_gs : pandas DataFrame
        A DataFrame with columns 'S' (group size), 'N' (collective action binary variable), 
        and any additional variables describing the events.

    Returns:
    --------
    GS : list of int
        The unique group sizes in the dataset.
    gs_thresholds : list of float
        The threshold tmperature needed to initiate collective action for each group size.
    gs_ci_lo : list of float
        The lower confidence interval for the threshold for each group size.
    gs_ci_hi : list of float
        The upper confidence interval for the threshold for each group size.
    gs_logits : list of float
        The logistic regression results for each group size.
    gs_rcs : list of pandas DataFrame
        The response curve (proportion of events involving collective action as a function of group size) 
        for each group size.
    gs_logit_full : statsmodels LogitResult
        The full logistic regression model with group size and temperature as predictors.

    Notes:
    ------
    The function removes any events with missing values for the collective action variable.
    The function calls two helper functions: find_threshold_from_events and response_curve_from_events.
    """

    # logistic regression model with group size as a varaible
    events = events_gs[~np.isnan(events_gs['collective_binary'])]
    X = events[['S','N']]
    y = events['collective_binary']
    gs_logit_full = logistic_regression(X, y)


    # analysis per group size
    GS = sorted(events_gs['N'].unique())
    gs_thresholds = []
    gs_ci_hi = []
    gs_ci_lo = []
    gs_logits = []
    gs_rcs = []

    for igs, gs in enumerate(GS):

        print('gs = ' + str(gs))

        events_i = events[events['N']==gs]

        th, cil, cih, logit = find_threshold_from_events(events_i)

        gs_rcs.append(response_curve_from_events(events_i))
        gs_thresholds.append(th)
        gs_ci_lo.append(cil)
        gs_ci_hi.append(cih)
        gs_logits.append(logit)

    return GS, gs_thresholds, gs_ci_lo, gs_ci_hi, gs_logits, gs_rcs, gs_logit_full


def find_threshold_ci(x, y, B=1000, alpha=0.05):
    """
    Calculates the confidence interval for the threshold value using bootstrapping.

    Parameters:
    -----------
        x (array-like): The independent variable values.
        y (array-like): The dependent binary variable values.
        B (int, optional): The number of bootstrap samples to generate. Defaults is set to 1000.
        alpha (float, optional): The significance level for the confidence interval. Defaults to 0.05.

    Returns:
    --------
        tuple: A tuple containing the lower and upper bounds of the confidence interval for the threshold value.
    """
    
    n = len(x) 
    thresholds = np.zeros((B,))
    
    for b in range(B):
        # rarely, sampling causes singularity, so just sample again
        while True:
            try:
                samples = np.random.choice(n, size=n, replace = True)
                thresholds[b] = find_threshold(x[samples], y[samples])
                break
            except:
                pass 
    
    thresholds = np.sort(thresholds)
    
    lower = np.quantile(thresholds, alpha/2)
    upper = np.quantile(thresholds, 1-alpha/2)   
    
    return (lower, upper)


def find_threshold(x, y, logit=None):
    """
    Calculates the threshold value for a binary outcome using a logistic regression model.

    Parameters:
    -----------
    x : numpy.ndarray
        A one-dimensional numpy array of predictor values.

    y : numpy.ndarray
        A one-dimensional numpy array of binary outcome values (0 or 1).

    logit : statsmodels LogitResult, optional (default=None)
        A fitted logistic regression model with x as the predictor variable and y as the binary outcome.

    Returns:
    --------
    threshold : float
        The threshold value for a binary outcome based on the logistic regression model.

    Notes:
    ------
    The function fits a logistic regression model to the predictor variable `x` and the binary outcome 
    `y` using the `logistic_regression` function defined elsewhere if `logit` is None. If `logit` is 
    provided, the function uses it to calculate the threshold directly. The threshold is calculated as 
    the negative ratio of the intercept to the coefficient of the predictor variable in the logistic 
    regression model.

    Explanation:
    ------------
    The threshold value is found by calculating the value of x where the probability of the dependent variable (y) equals 0.5. 
    This is done by solving the equation derived from the logistic regression model: log(p/(1-p)) = b0 + b1*x, where p is the 
    probability of the dependent variable, and b0 and b1 are the intercept and slope of the logistic regression model, respectively. 
    By setting p=0.5, we can solve for x to get the threshold value: x = -b0/b1.
    """
    
    if logit is None:
        logit = logistic_regression(x, y)
        
    threshold = -logit.params[0]/logit.params[1]
    
    return threshold


def logistic_regression(x, y):
    """
    Fits a logistic regression model to the given data.
    
    Parameters:
    -----------
    x : numpy.ndarray
        The input data array of shape (n_samples, n_features).
    y : numpy.ndarray
        The target data array of shape (n_samples,).
    
    Returns:
    statsmodels.genmod.generalized_linear_model.GLMResultsWrapper: A fitted logistic regression model.
    """
    
    return Logit(endog=y, exog=add_constant(x), method='bfgs').fit_regularized(disp=False)


def collective_binary_response(events, quorum_threshold=0.9, duration_threshold=300, exclude=True):
    """
    Computes the binary collective response for each event in a given DataFrame of events.

    Parameters:
    -----------
    events: pandas.DataFrame
        DataFrame containing information about the events, such as the number of individuals, the time at which the 
        event started, and the temperature.
    quorum_threshold: float, optional
        The proportion of individuals needed to be active at the same time to meet the quorum threshold. The default 
        value is 0.9.
    duration_threshold: int, optional
        The minimum duration (in seconds) for which the collective behavior should persist. The default value is 300.
    exclude: bool, optional
        If True, exclude events that start with high activity from the analysis. The default value is True.

    Returns:
    --------
    events: pandas.DataFrame
        The same DataFrame as the input, but with an additional column called 'collective_binary' which indicates 
        whether each event exhibits collective behavior or not. A value of 1 indicates collective behavior, while 0 
        indicates the absence of collective behavior. If exclude is True, some events may have a value of NaN 
        in the 'collective_binary' column.
    """

    events['collective_binary'] = np.nan

    for ix, ev in events.iterrows():
        gb = groupby(ev['Np'] > quorum_threshold * ev['N'])
        events.loc[ix, 'collective_binary'] = int(max([0]+[len(list(v)) for k, v in gb if k]) > duration_threshold)

    # ignore events that starts with high activity
    if exclude:
        cnt = 0
        activity_threshold = 2 * np.nanmedian([ev['Np'][0] / ev['N'] for _, ev in events.iterrows()])

        for ix, ev in events.iterrows():
            if ev['Np'][0] / ev['N'] > activity_threshold or np.isnan(ev['Np'][0]):
                cnt += 1
                events.loc[ix, 'collective_binary'] = np.nan

        print('activity threshold is %.2f ' % activity_threshold)
        print('%d events excluded due to initial high activity' % cnt)

    return events


def find_threshold_from_events(events, alpha=0.05):
    """
    Calculates the threshold temperature for collective action, the confidence interval for the threshold, 
    and the logistic regression model for a subset of events.

    Parameters:
    -----------
    events : pandas DataFrame
        A DataFrame with columns 'S' (temperature), 'collective_binary' (binary variable indicating whether 
        an event involves collective action or not), and any additional variables describing the events.

    alpha : float, optional (default=0.05)
        The significance level for computing the confidence interval.

    Returns:
    --------
    th : float
        The threshold temperature for collective action.

    cil : float
        The lower confidence interval for the threshold.

    cih : float
        The upper confidence interval for the threshold.

    logit : statsmodels LogitResult
        The result of the logistic regression model with temperature as a predictor and collective 
        action as a binary outcome.

    Notes:
    ------
    The function assumes that the 'collective_binary' column contains binary values (0 or 1) indicating 
    whether an event involves collective action or not. The function removes any events with missing 
    values for the collective action variable. The logistic regression model is fit using the 
    `logistic_regression` function defined elsewhere. The threshold temperature is calculated using the 
    `find_threshold` function defined elsewhere. The confidence intervals for the threshold are 
    calculated using the `find_threshold_ci` function defined elsewhere.
    """

    events = events[~np.isnan(events['collective_binary'])]
    x = events['S'].values.reshape(-1, 1)
    y = events['collective_binary'].values.astype(float).reshape(-1, 1)
    
    logit = logistic_regression(x, y)
    th = find_threshold(x, y, logit=logit)
    cil, cih = find_threshold_ci(x, y, alpha=alpha)
          
    return th, cil, cih, logit


def response_curve_from_events(events, alpha=0.05):
    """
    Calculates the response curve (proportion of events involving collective action as a function of group size) 
    for a subset of events and computes confidence intervals for the mean proportion.

    Parameters:
    -----------
    events : pandas DataFrame
        A DataFrame with columns 'S' (temperature), 
        and any additional variables describing the events.

    alpha : float, optional (default=0.05)
        The significance level for computing the confidence intervals.

    Returns:
    --------
    rc : pandas DataFrame
        A DataFrame with columns 'S', 'mean', 'count', 'bin_ci_up', and 'bin_ci_dn', where:
            - 'S' is the temperature.
            - 'mean' is the mean proportion of events involving collective action at that temperature.
            - 'count' is the total number of events at that temperature.
            - 'bin_ci_up' is the upper confidence interval for the mean proportion.
            - 'bin_ci_dn' is the lower confidence interval for the mean proportion.

    Notes:
    ------
    The function assumes that the 'collective_binary' column contains binary values (0 or 1) indicating 
    whether an event involves collective action or not. The function removes any events with missing 
    values for the collective action variable. The confidence intervals are computed using the Wilson 
    score interval, which provides a conservative estimate of the true interval.
    """
    
    # Plot the response curve
    events = events[~np.isnan(events['collective_binary'])]
    rc = events.groupby(by='S').agg(['mean','count'])['collective_binary']

    # Compute confidence intervals for the response probability
    for ix, row in rc.iterrows():
        bin_ci_dn, bin_ci_up = proportion_confint(int(row['count']*row['mean']), row['count'], alpha=alpha, method='wilson')
        rc.loc[ix, 'bin_ci_up'] = bin_ci_up
        rc.loc[ix, 'bin_ci_dn'] = bin_ci_dn
        
    return rc


def cm2in(x):

    if isinstance(x, collections.Sequence):
        return tuple([cm2in(xi) for xi in x])
    else:
        return x / 2.54

#%%
