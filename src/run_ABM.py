from modeling import agents_construction, calibration
from engineering import interaction_builder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




if __name__ == '__main__':

    df = pd.read_csv(r'..\data\raw\financial_US_NVDA_raw.csv')
    price = df['Close']
    df_covid = pd.read_csv(r'..\data\raw\covid_US_raw.csv')
    N = 200                 # num of nodes
    M = 20                  # num of edges per node for cluster
    P = 0.5                 # p of connection for erdos renyi and cluster graph
    epochs = len(price)            # num of iterations
    beta = 0.3
    initial_stock_price = 60 # initial stock price
    num_stocks = N * 20   # num of available stocks
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
    print(Rt.shape)

    model = agents_construction.Nest_Model(
                                        beta=beta, 
                                        initial_stock_price = initial_stock_price,
                                        external_var = Rt,
                                        interaction_graph = G,
                                        num_stocks = num_stocks)

    df_model, df_agents = agents_construction.run_model(model=model, epochs=epochs)
    plt.plot(df_model['price'])
    plt.show()
    # results = calibration.metropolis_hastings(model, 5, 0.01, std=0.1, true_data=price, burn_in=1)


