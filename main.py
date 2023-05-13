# supposedly, main file for running everything from one script when we are ready
from src import global_configs
from src import data_loading
from src.engineering import create_main_df
from src.engineering import interaction_builder
from src.visuals import basic_views
from src.modeling import agents_construction
from src import run_ABM
import numpy as np
global_configs.init()


if __name__ == '__main__':
    # here we can attempt to use a function written in another script as below
    print('Starting main.py file. Welcome to the pipeline for our project...')
    # data_loading.load_financial_data()
    # create_main_df.load_R_number_data()
    # create_main_df.retrieve_cases_and_R_number()
    N = 200 # num of nodes
    M = 10  # num of edges per node for cluster
    P = 0.6 # p of connection for erdos renyi and cluster
    epochs = 100 # num of iterations
    beta = 1
    initial_stock_price = 1 # initial stock price
    num_stocks = 200000     # num of available stocks
    Rt_hyperbolic = np.sin(np.linspace(-5,5, epochs))+1
    Rt_ascending = np.linspace(0,2, num = epochs + 1)
    Rt_null = np.zeros(epochs + 1)
    G_1 = interaction_builder.graph_generator(type = 'Erdos-Renyi',
                        weights_distribution = lambda : np.random.uniform(0,1),
                        **{'n':N, 'p':P})
    G_2 = interaction_builder.graph_generator(type = 'Clique', **{'n' : N})
    G_3 = interaction_builder.graph_generator(type = 'Null', **{'n' : N})
    G_4 = interaction_builder.graph_generator(type = 'Powerlaw-Cluster', 
                                              **{'n' : N, 'm' : M, 'p' : P})
    G = G_3
    model = agents_construction.Nest_Model(
                                        beta=beta, 
                                        initial_stock_price = initial_stock_price,
                                        external_var = Rt_null,
                                        interaction_graph = G,
                                        num_stocks = num_stocks)

    df_model, df_agents = agents_construction.run_model(model=model, epochs=epochs)
    print(df_agents.columns)
    basic_views.plot_graph(G, save = True, title = 'Graph viz')
    basic_views.plot_price_dynamics(df_model, save = True, title = 'Price Dynamics')
    basic_views.plot_agents_dynamics(df_model, df_agents, title = 'Nest_all_steps', 
                                             hue = 'state', save = True)
    # print(t, len(new_df))
    # fig_2 = basic_views.plot_price_dynamics(df_model, save = True, title = 'Price Dynamics')
    # fig_2.show()
    print('Computation is finished')