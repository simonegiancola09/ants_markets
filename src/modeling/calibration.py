import numpy as np
from modeling import agents_construction as ac
from scipy.stats import norm

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
    num_stocks = model.num_available
    initial_stock = model.stock_price
    rt = model.external_var
    beta_start = model.beta

    epochs = len(true_data) + 1
    print('model run for: ', epochs)
    y_true = preprocess_data(true_data)

    df_model, df_agents = ac.run_model(model, epochs)
       
    price = df_model['price'].astype(float)

    y_hat = preprocess_data(price)
    print(y_hat.shape, y_true.shape)

    llkh_start = compute_log_likelihood(y_hat, true_data)
    BETAS =[beta_start]
    llkh = [llkh_start]
    
    for ITER in range(iterations):
        if ITER % 50 == 0:
            print(np.round(ITER / iterations*100, 2))
        
        beta_current = BETAS[-1]
        llkh_current = llkh[-1]

        beta_new = np.random.normal(beta_current, std, size = 1)
        model = ac.Nest_Model(beta_new, initial_stock, rt, G, num_stocks)
        df_model, df_agents = ac.run_model(model, epochs)
        
        price = df_model['price'].astype(float)

        y_hat = preprocess_data(price)        
        print(y_hat.shape, y_true.shape)
        llkh_new = compute_log_likelihood(y_hat, true_data)

        alpha  = np.e**(llkh_new - llkh_current)

        if alpha <= np.random.uniform(0,1): 
            BETAS.append(beta_current)
            llkh.append(llkh_current)

        else: 
            BETAS.append(beta_new)
            llkh.append(llkh_new)
      
        print(BETAS[-1])
    BETAS = BETAS[burn_in:] 
    llkh = llkh[burn_in:]
    
    return np.array(BETAS).flatten(), np.array(llkh).flatten()
