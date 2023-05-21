import numpy as np
import agents_construction as ac
from scipy.stats import norm
import pandas as pd
from mesa.batchrunner import BatchRunner

def preprocess_data(price_data):
    """
    Preprocesses the price data by converting it into returns.
    
    Parameters:
    - price_data: numpy array or list of stock prices
    
    Returns:
    - returns_data: numpy array of log returns
    """
    price_data = pd.Series(price_data)
    return price_data.pct_change()

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

def metropolis_hastings(model, iterations, param_start, std, true_data, burn_in):
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
    initial_stock = model.price
    rt = model.external_var
    alpha_start = model.alpha
    # we use a lambda function to store the model with all fixed but the 
    # parameter we wish to fit, in this case alpha but the code can be generalized easily
    # to fit the cutoff or something else
    model_pivot = lambda x: ac.Nest_Model(interaction_graph= model.G,
                                            external_var = model.external_var,
                                            alpha = x,  #!!!!! here we change !!!!!                     
                                            cutoff = model.cutoff,
                                            k = model.k,
                                            price_history = model.price_history,
                                            p = model.p,
                                            prob_type = model.prob_type,
                                            p_f = model.p_f,
                                            cash_low = model.cash_low,
                                            cash_high = model.cash_high,
                                            stocks_low = model.stocks_low,
                                            stocks_high =  model.stocks_high)

    epochs = len(true_data) 
    print('Calibration of model runs for: ', epochs, 'epochs')
    y_true = preprocess_data(true_data)

    df_model, df_agents = ac.run_model(model, epochs)
       
    price = df_model['price'].astype(float)
    print('length of price is', len(price))
    y_hat = preprocess_data(price)
    y_hat = y_hat[:epochs]        # TODO do not understand why it gives a 161 long array with 61 epochs
                                            # keep only the matching part
    # print(y_hat.shape, y_true.shape)

    llkh_start = compute_log_likelihood(y_hat, true_data)
    ALPHAS =[alpha_start]
    llkh = [llkh_start]
    
    for ITER in range(iterations):
        if ITER % 50 == 0:
            print('Reached iteration number {}'.format(np.round(ITER / iterations*100, 2)))
        
        alpha_current = ALPHAS[-1]
        llkh_current = llkh[-1]

        alpha_new = np.random.normal(alpha_current, std, size = 1)
        model_new = model_pivot(alpha_new)
        df_model, df_agents = ac.run_model(model_new, epochs)
        
        price = df_model['price'].astype(float)

        y_hat = preprocess_data(price)        
        print(y_hat.shape, y_true.shape)
        llkh_new = compute_log_likelihood(y_hat, true_data)

        score_MH  = np.e**(llkh_new - llkh_current)

        if score_MH <= np.random.uniform(0,1): 
            ALPHAS.append(alpha_current)
            llkh.append(llkh_current)

        else: 
            ALPHAS.append(alpha_new)
            llkh.append(llkh_new)
      
        print('Last Value found is {}'.format(np.round(ALPHAS[-1], 2)))     #TODO maybe here we need the mean?? Not the last
    ALPHAS = ALPHAS[burn_in:] 
    llkh = llkh[burn_in:]
    
    return {'parameter estimate' : np.mean(ALPHAS),
            'parameter realizations' : np.array(ALPHAS).flatten(),
            'likelihood realizations': np.array(llkh).flatten()}


def batch_run(model, fixed_params, variable_params, reporters, 
            epochs, iterations=1):
    batch_run = BatchRunner(
                model,
                variable_params, 
                fixed_params,
                iterations,
                epochs,
                reporters
            )
    batch_run.run_all()
    run_data = batch_run.get_model_vars_dataframe()
    return run_data

