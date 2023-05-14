# supposedly, main file for running everything from one script when we are ready
# main imports
import time
import numpy as np
import pandas as pd
# our coded functions imports
from src import global_configs
from src import data_loading
from src.engineering import create_main_df
from src.engineering import interaction_builder
from src.visuals import basic_views
from src.modeling import agents_construction
from src import run_ABM
# initialization of Global Variables
global_configs.init()


if __name__ == '__main__':
    # here we can attempt to use a function written in another script as below
    ############## GREETINGS ########################################################
    print('Starting main.py file. Welcome to the pipeline for our project...')
    print('Today we will try to run our model, you are free to choose parameters in the main.py file')
    time.sleep(5)
    print('WARNING: if you encounter issues due to missing modules, there is a requirements.txt file ready for you.')
    print('In this case, please run on your terminal pip install requirements.txt or what you prefer in your envinroment of choice')
    time.sleep(5)
    ################################################################################################################
    tot_time_start = time.time()
    # Please choose here the stock and the dates for calibration 
    stock_name = 'NVDA'                     # name of stock
    dates = ('2020-01-01', '2020-03-31')    # dates for stock data, must be larger than what we
                                            # use for covid to calibrate
    print('Your stock of choice is {} in the historical dates {}'.format(stock_name, dates))
    ############## DATA LOADING ###########################################
    loaded = True                          # run only once, then set as True
    # if data is not loaded, we load it here
    if not loaded:
        print('Loading data...')
        load_time_start = time.time()
        # load total number of cases
        data_loading.load_covid_data()
        # load stock name under specific dates
        data_loading.load_financial_data(stock_name=stock_name, dates=dates)
        # create series of R number data
        create_main_df.load_R_number_data()
        # create dataset with R number and cases for simplicity
        create_main_df.retrieve_cases_and_R_number()
        load_time_end = time.time()
        load_time = load_time_end - load_time_start
        print('Finished downloading datasets in {} seconds'.format(np.round(load_time, 2)))
    ################################################################################################################
    ############## Stocks data ###############################
    print('Retrieving stock data...')
    df_stocks = pd.read_csv('data/raw/financial_US_{}_raw.csv'.format(stock_name))
    ##########################################################
    ############## Covid Data ###############################
    print('Retrieving Covid-19 data...')
    # this dataset has more information than just the Rt
    df_covid = pd.read_csv('data/engineered/df_covid.csv')
    # for now, we only retrieve the Rt
    Rt_real = df_covid['R_mean']
    # work out additional length
    additional_length_for_calibration = (pd.to_datetime(dates[0]) - pd.to_datetime('2020-01-01')).days # decide calibration length
    # obtain an enlarged version of the Rt
    R_t_real = np.concatenate([np.zeros(additional_length_for_calibration), Rt_real])
    # enlarge it as to account for when the pandemic was not present
    # we use the first iterations to calibrate parameters
    # then we perturb it with an Rt different than zero
    ##########################################################
    ############## HYPERPARAMETERS ###############################
    print('Setting hyperparameters...')
    # here we store some parameters of choice
    N = 100                 # num of nodes
    M = 10                  # num of edges per node for cluster
    P = 0.6                 # p of connection for erdos renyi and cluster graph
    epochs = 1000            # num of iterations
    beta = 1 
    initial_stock_price = 1 # initial stock price
    num_stocks = 200000     # num of available stocks
    ##########################################################
    ############### CHOICES FOR TOY MODELS #####################
    # below are different types of Rt to test our dynamics
    Rt_hyperbolic = np.sin(np.linspace(-5,5, epochs))+1
    Rt_ascending = np.linspace(0,2, num = epochs + 1)
    Rt_null = np.zeros(epochs + 1)
    # ascending descending
    Rt_ascending_descending = np.concatenate([np.linspace(0,2, num = epochs // 2),np.linspace(2,0, num = epochs // 2)])
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
    ######################
    ################ SPECIFIC CHOICES FOR RUNNING ########################
    G = G_4
    Rt = Rt_ascending_descending
    ##########################################################
    ################ MODEL RUN #######################################################
    print('Running ABM...')
    model_time_start = time.time()
    model = agents_construction.Nest_Model(
                                        beta=beta, 
                                        initial_stock_price = initial_stock_price,
                                        external_var = Rt,
                                        interaction_graph = G,
                                        num_stocks = num_stocks
                                        )

    df_model, df_agents = agents_construction.run_model(model=model, epochs=epochs)
    model_time_end = time.time()
    model_time = model_time_end - model_time_start
    print('Run ABM finished in {} seconds'.format(np.round(model_time, 2)))
    ##########################################################
    ################ PLOTS #######################################################
    print('Creating Plots...')
    plots_time_start = time.time()
    basic_views.plot_graph(G, save = True, title = 'Graph viz')
    basic_views.plot_price_dynamics(df_model, save = True, title = 'Price Dynamics')
    basic_views.plot_agents_dynamics(df_model, df_agents, title = 'Nest_all_steps', 
                                             hue = 'state', save = True)
    plots_time_end = time.time()
    plots_time = plots_time_end - plots_time_start
    print('Plots finished in {} seconds'.format(np.round(plots_time, 2)))
    ##########################################################
    # print(t, len(new_df))
    # fig_2 = basic_views.plot_price_dynamics(df_model, save = True, title = 'Price Dynamics')
    # fig_2.show()

    tot_time_end = time.time()
    tot_time = tot_time_end - tot_time_start
    print('Computation is finished in a total of {} seconds'.format(np.round(tot_time, 2)))