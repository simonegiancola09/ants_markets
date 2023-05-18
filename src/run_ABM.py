from modeling import agents_construction, calibration
from engineering import interaction_builder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




if __name__ == '__main__':

    df = pd.read_csv(r'..\data\raw\financial_US_NVDA_raw.csv')
    price = df['Close']
    # df_covid = pd.read_csv(r'..\data\raw\covid_US_raw.csv')

    ## PARAMS OF THE MODEL ##
    # number of nodes
    N = 1000  
    # num of edges per node for cluster          
    M = 20      
    # p of connection for erdos renyi and cluster graph            
    P = 0.5     
    # num of iterations            
    epochs = 300   
    # parameter controlling variance of price decision for random traders        
    k = 0.0000002
    # probability of being active (placing an order)
    p = 0.1
    # distribution over types of traders
    prob_type = [0.97, 0.01, 0.01, 0.01]
    # price value for fundamentalist traders 
    # (set to None to automatically give mean price of price history)
    p_f = None
    ## paramenters on distribution of wealth
    cash_low = 100
    cash_high = 200
    stocks_low = 100
    stocks_high = 500



    ##########################################################
    ############### CHOICES FOR TOY MODELS #####################
    # below are different types of Rt to test our dynamics
    Rt_hyperbolic = np.sin(np.linspace(0,3, epochs))+1
    Rt_ascending = np.linspace(0,2, num = epochs + 1)
    Rt_null = np.zeros(epochs + 1)
    
    # ascending descending
    Rt_ascending_descending = np.concatenate([np.linspace(0,2, num = epochs // 2),np.linspace(2,0, num = epochs // 2)])
    Rt_fake = np.r_[np.zeros(epochs//2), np.linspace(0,2, num = epochs // 4),np.linspace(2,0, num = epochs // 4+1)]
    ######################
    print('Building graphs')
    # below are different types of interaction graphs to test our dynamics
    G_1 = interaction_builder.graph_generator(type = 'Erdos-Renyi',
                        weights_distribution = lambda : np.random.uniform(0,1),
                        **{'n':N, 'p':P})
    G_2 = interaction_builder.graph_generator(type = 'Clique', **{'n' : N})
    G_3 = interaction_builder.graph_generator(type = 'Null', **{'n' : N})
    G_4 = interaction_builder.graph_generator(type = 'Powerlaw-Cluster', 
                                              **{'n' : N, 'm' : M, 'p' : P})

    G = G_4
    Rt = Rt_null

    model = agents_construction.Nest_Model(
                                        k=k, 
                                        price_history = list(price),
                                        shock= Rt,
                                        G = G,
                                        p=p,
                                        prob_type=prob_type,
                                        p_f = p_f,
                                        cash_low = cash_low,
                                        cash_high = cash_high,
                                        stocks_low = stocks_low,
                                        stocks_high = stocks_high
                                        )
    print('Running ABM...')
    df_model, df_agents = agents_construction.run_model(model=model, epochs=epochs)
    plt.plot(pd.concat([price, df_model['price']]).reset_index(drop=True))
    plt.show()
    # results = calibration.metropolis_hastings(model, 5, 0.01, std=0.1, true_data=price, burn_in=1)


