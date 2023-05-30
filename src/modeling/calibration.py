import numpy as np
import src.modeling.agents_construction as ac
from scipy.stats import norm
import pandas as pd
from mesa.batchrunner import BatchRunner
import time

def preprocess_data(price_data):
    """
    Preprocesses the price data by converting it into returns.
    
    Parameters:
    - price_data: numpy array or list of stock prices
    
    Returns:
    - returns_data: numpy array of log returns
    """
    price_data = pd.Series(price_data)
    return price_data.pct_change().fillna(0)

def compute_log_likelihood(y_hat, y_true):
    """
    Computes the log likelihood of the observed data under the Normal distribution.
    
    Parameters:
    - y_hat: numpy array of simulated log returns
    - y_true: numpy array of observed log returns
    
    Returns:
    - log_likelihood: float representing the log likelihood
    """
    log_likelihood = np.sum(norm.logpdf(y_hat, y_true))
    return log_likelihood

def MSE(y_hat, y_true):
    return -np.sum((y_hat - y_true)**2)

def multi_run(model, epochs, iterations):

    G = model.G
    rt = model.external_var
    alpha = model.alpha
    k = model.k
    price_history = model.price_history
    prob_type = model.prob_type
    p_f = model.p_f
    cash_low = model.cash_low
    cash_high = model.cash_high
    stocks_low = model.stocks_low
    stocks_high = model.stocks_high
    debug = model.debug

    prices = []
    volumes = []
    states = []
    for epoch in range(iterations):
        model = ac.Nest_Model(
            interaction_graph= G,
            external_var = rt,
            alpha = alpha,                    
            k = k,
            price_history = price_history,
            prob_type = prob_type,
            p_f = p_f,
            cash_low = cash_low,
            cash_high = cash_high,
            stocks_low = stocks_low,
            stocks_high = stocks_high,
            debug=debug
            )
        df_model, df_agents = ac.run_model(model, epochs)
        prices.append(df_model['price'])
        volumes.append(df_model['volume'])
        states.append(df_model['magnetization'])

    output = {
        'prices':np.array(prices), 
        'volumes':np.array(volumes),
        'states':np.array(states)
    }
    return output

def metropolis_hastings(
    model, iterations, internal_iterations, loss, preprocess,
    param_start, std, true_data, burn_in, multi=False, fit_alpha=True):
    """
    Metropolis Hastings algorithm.

    Parameters:
    - model: an ABM model
    - iterations: number of iterations of the mh
    - param_start: the parameter to fit
    - std: standard deviation of the proposal of the Normal distribution
    - true_data: data 
    - burn_in: numner of iterations to discard
    
    Returns:
    - tuple with 
    """

    G = model.G
    rt = model.external_var
    alpha_start = param_start

    # we use a lambda function to store the model with all fixed but the 
    # parameter we wish to fit, in this case alpha but the code can be generalized easily
    # to fit the cutoff or something else
    if fit_alpha:
        model_pivot = lambda x: ac.Nest_Model(interaction_graph= model.G,
                                                external_var = model.external_var,
                                                alpha = x,  #!!!!! here we change !!!!!                     
                                                k = model.k,
                                                price_history = model.price_history,
                                                prob_type = model.prob_type,
                                                p_f = model.p_f,
                                                cash_low = model.cash_low,
                                                cash_high = model.cash_high,
                                                stocks_low = model.stocks_low,
                                                stocks_high =  model.stocks_high,
                                                debug=False)
    else:
        model_pivot = lambda x: ac.Nest_Model(interaction_graph= model.G,
                                                external_var = model.external_var,
                                                alpha = model.alpha,                      
                                                k = x,   #!!!!! here we change !!!!! 
                                                price_history = model.price_history,
                                                prob_type = model.prob_type,
                                                p_f = model.p_f,
                                                cash_low = model.cash_low,
                                                cash_high = model.cash_high,
                                                stocks_low = model.stocks_low,
                                                stocks_high =  model.stocks_high,
                                                debug=False)


    epochs = len(true_data) 
    print('Calibration of model runs for: ', epochs, 'epochs')

    y_true = true_data.reset_index(drop=True)
    if multi:
        t1 = time.time()
        model.alpha = alpha_start
        df_multi_run = multi_run(model, epochs, internal_iterations)
        y_hat = df_multi_run['prices'].mean(axis=0)
        t2 = time.time()
        exec_time = t2 - t1
        est_time = exec_time * iterations  / 60
        print('Estimated time:', np.round(est_time, 3), ' minutes')

    else:
        t1 = time.time()
        model.alpha = alpha_start
        df_model, df_agents = ac.run_model(model, epochs) 
        y_hat = np.append(df_model['price'].values, model.price)[1:]
        t2 = time.time()
        exec_time = t2 - t1
        est_time = exec_time * iterations / 60
        print('Estimated time:', np.round(est_time, 3), ' minutes')

    if preprocess:
        y_hat = preprocess_data(y_hat)
        y_true = preprocess_data(y_true) 

    llkh_start = loss(y_hat, y_true)
    ALPHAS =[alpha_start]
    llkh = [llkh_start]
    PROPOSALS = [alpha_start]
    accepted = 0
    for ITER in range(iterations):

        
        alpha_current = ALPHAS[-1]
        llkh_current = llkh[-1]

        alpha_new = np.random.normal(alpha_current, std)
        if alpha_new > 0:
            PROPOSALS.append(alpha_new)
            ALPHAS.append(alpha_current)
            llkh.append(llkh_current)
            continue
        PROPOSALS.append(alpha_new)
        model_new = model_pivot(alpha_new)

        if multi:
            df_multi_run = multi_run(model_new, epochs, internal_iterations)
            y_hat = df_multi_run['prices'].mean(axis=0)

        else:
            df_model, df_agents = ac.run_model(model_new, epochs) 
            y_hat = np.append(df_model['price'].values, model_new.price)[1:]

        if preprocess:
            y_hat = preprocess_data(y_hat)

        llkh_new = loss(y_hat, y_true)

        score_MH  = np.exp(llkh_new - llkh_current)

        if score_MH <= np.random.rand(): 
            ALPHAS.append(alpha_current)
            llkh.append(llkh_current)

        else: 
            ALPHAS.append(alpha_new)
            llkh.append(llkh_new)
            accepted += 1
        if ITER % 50 == 0:
            print('Reached ITERATION number {}'.format(np.round(ITER / iterations*100, 2)))
            print('Value found until now is {}'.format(np.round(np.mean(ALPHAS), 2)))
            print('acceptance rate: ', np.round(accepted / (ITER+1), 2)) 
    ALPHAS = ALPHAS[burn_in:] 
    llkh = llkh[burn_in:]
    
    return {'parameter estimate' : np.mean(ALPHAS),
            'parameter realizations' : np.array(ALPHAS).flatten(),
            'likelihood realizations': np.array(llkh).flatten(),
            'proposals': np.array(PROPOSALS).flatten()}


def batch_run(model, variable_params, fixed_params, reporters, 
            epochs, iterations=1):
    batch_run = BatchRunner(
                model,
                variable_params, 
                fixed_params,
                iterations=iterations,
                max_steps=epochs,
                model_reporters=reporters
            )
    batch_run.run_all()
    run_data = batch_run.get_model_vars_dataframe()
    return run_data


def extract_price(model):
    return model.price_history + [model.price]

