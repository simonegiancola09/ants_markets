from modeling import agents_construction#, calibration
from engineering import interaction_builder
import matplotlib.pyplot as plt
# from visuals import basic_views
import numpy as np
import pandas as pd




if __name__ == '__main__':

    df_price = pd.read_csv(r'..\data\raw\financial_US_NVDA_raw.csv')
    df_covid = pd.read_csv(r'..\data\raw\covid_US_raw.csv')
    df_covid['daily_cases'] = df_covid['cases'].diff().fillna(0)

    df_covid['start'] = np.arange(df_covid.shape[0])

    df = pd.merge(df_price, df_covid, left_on='Date', right_on='date', how='left')
    df.loc[df['start'].isna(), 'start'] = np.arange(-1, -(df['start'].isna().sum()+1), -1)[::-1]
    df['change'] = df['cases'].pct_change()
    df = df.fillna(0)
    price = df.loc[df['start'] <= -20, 'Close']
    Rt_real = df.loc[df['start'] > -20, 'change'].values
    ## PARAMS OF THE MODEL ##

    graph_type = None
    # number of nodes
    N = 600  
    # num of edges per node for cluster          
    M = 10      
    # p of connection for erdos renyi and cluster graph            
    P = 0.5     
    # num of iterations            
    epochs = 120
    # parameter controlling variance of price decision for random traders        
    k = 0.5
    # distribution over types of traders
    prob_type = [1, 0., 0., 0.]
    # price value for fundamentalist traders 
    # (set to None to automatically give mean price of price history)
    p_f = None
    ## parameter controlling role of Temperature
    alpha = 3
    ## paramenters on distribution of wealth
    cash_low = 50000
    cash_high = 10000
    stocks_low = 1000
    stocks_high = 100



    ##########################################################
    ############### CHOICES FOR TOY MODELS #####################
    # below are different types of Rt to test our dynamics
    Rt_hyperbolic = np.sin(np.linspace(0,3, epochs))+1
    Rt_ascending = np.linspace(0,2, num = epochs + 1)
    Rt_null = np.zeros(epochs + 1)
    weights_distribution = np.random.uniform(1,1.5)
    
    # ascending descending
    Rt_ascending_descending = np.concatenate([np.linspace(0,2, num = epochs // 2),np.linspace(2,0, num = epochs // 2)])
    Rt_fake = np.r_[np.zeros(epochs//4), np.linspace(0,2, num = epochs // 4),np.linspace(2,0, num = epochs // 4), np.zeros(epochs//4)]
    ######################
    print('Building graphs')
    # below are different types of interaction graphs to test our dynamics
    G_1 = interaction_builder.graph_generator(type = 'Erdos-Renyi',
                        weights_distribution = lambda : weights_distribution,
                        **{'n':N, 'p':P})
    G_2 = interaction_builder.graph_generator(type = 'Clique', 
                        weights_distribution = lambda : weights_distribution,
                        **{'n' : N})
    G_3 = interaction_builder.graph_generator(type = 'Null',
                        weights_distribution = lambda : weights_distribution,
                        **{'n' : N})
    G_4 = interaction_builder.graph_generator(type = 'Powerlaw-Cluster',
                        weights_distribution = lambda : weights_distribution, 
                                              **{'n' : N, 'm' : M, 'p' : P})

    G = G_4
    Rt = Rt_real

    model = agents_construction.Nest_Model(
                                        k=k, 
                                        price_history = list(price),
                                        date = '',
                                        external_var= Rt,
                                        interaction_graph = G,
                                        prob_type=prob_type,
                                        alpha = alpha,
                                        p_f = p_f,
                                        cash_low = cash_low,
                                        cash_high = cash_high,
                                        stocks_low = stocks_low,
                                        stocks_high = stocks_high
                                        )
    print('Running ABM...')
    df_model, df_agents = agents_construction.run_model(model=model, epochs=epochs)
    full_price = pd.concat([price, df_model['price']]).reset_index(drop=True)
    plt.plot(full_price)
    plt.show()
    grouped = df_agents.groupby('Step').mean()
    # results = calibration.metropolis_hastings(model, 5, 0.01, std=0.1, true_data=price, burn_in=1)


