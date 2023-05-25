# main imports
import time
import numpy as np
import pandas as pd
# our coded functions imports
from src import global_configs, data_loading
from src.engineering import create_main_df, interaction_builder
from src.visuals import basic_views, make_gif
from src.modeling import agents_construction, calibration
from src.utils import utils
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
    print("creating the necessary folders...")
    utils.create_directory(r'\reports')
    utils.create_directory(r'\reports\figures')
    utils.create_directory(r'\reports\outputs')
    utils.create_directory(r'\reports\figures\nest_dynamics')

    ################################################################################################################
    tot_time_start = time.time()
    
    # Please choose here the stock and the dates for calibration 
    stock_name = 'NVDA'                     # name of stock
    dates = ('2019-01-01', '2020-06-30')    # dates for stock data, must be larger than what we
                                            # use for covid to calibrate
    print('Your stock of choice is {} in the historical dates {}'.format(stock_name, dates))
    
    ############## DATA LOADING ###########################################
    # run only once, then set as True
    loaded = True                          
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

    ############# Merge Datasets ############################
    df = pd.merge(df_stocks, df_covid, left_on='Date', right_on='date', how='left')
    # proprocess data to obtain the "temperature" values
    df = utils.preprocess_data(df) 
    ##########################################################
    
    ############## HYPERPARAMETERS ###############################
    print('Setting hyperparameters...')
    # here we store some parameters of choice
    #### General hypeparams ############
    N = 600                             # num of nodes
    M = 10                              # num of edges per node for cluster
    P = 0.5                             # p of connection for erdos renyi and cluster graph
    debug=False                         # set to True to visualize supply/demand curves every 5 steps
    ##########
    #### Ant-Like hypeparams ############
    alpha = -0.8                        # parameter controlling how much the temperature contributes
    ##########
    #### DemandSupply-Like hypeparams ############
    k = 0.09                            # parameter controlling variance of price decision for random traders        
    prob_type = [1., 0.0, 0.0, 0.0]     # distribution over types of traders
    p_f = None                          # price value for fundamentalist traders 
                                        # (set to None to automatically give mean price of price history)
    cash_low = 50000                    # paramenters on distribution of wealth
    cash_high = 200000
    stocks_low = 1000
    stocks_high = 5000
    ##########
    price = df.loc[df['start'] <= -start, 'Close']
    ##########################################################
    
    ############### GRAPHS CONSTRUCTION #####################
    # below are different types of Rt to test our dynamics
    print('Building graphs...')
    # below are different types of interaction graphs to test our dynamics
    # choice for the weights distribution
    weights_distribution = np.random.uniform(0.8,1.2)

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

    ##################################################################
    
    ################ SPECIFIC CHOICES FOR MODEL INITIALIZATION ########################
    G = G_4                             # graph chosen
    start = 2                           # simulation begins 'start' days before pandemic
    start_pre = 30                      # simulation before covid
    epochs = df['start'].max() + start  # number of iterations            
    fixed_kwargs = {
                'k':k, 
                'interaction_graph' : G,
                'prob_type':prob_type,
                'alpha' : alpha,
                'p_f' : p_f,
                'cash_low' : cash_low,
                'cash_high' : cash_high,
                'stocks_low' : stocks_low,
                'stocks_high' : stocks_high,
                'debug':debug
                }
    ##########################################################

    ################ MODEL INITIALIZATION #######################################################

    model = utils.build_model(df, start, fixed_kwargs)
    model_pre = utils.build_model(df, start_pre, fixed_kwargs)

    ############# CALIBRATION ##################################
    time.sleep(3)
    print('Calibration of Pandemic contribution...')
    calibration_time_start = time.time()

    std = 0.2
    param_start = -0.5
    iterations_mh = 100
    internal_iterations = 5
    true_data = df.loc[df['start'] > -start, 'Close']
    burn_in = 0
    multi=False
    fit_alpha=True
    loss = calibration.compute_log_likelihood
    preprocess = True

    calibration_output = calibration.metropolis_hastings(
        model, 
        iterations_mh, 
        internal_iterations,
        loss,
        preprocess,
        param_start, 
        std, 
        true_data, 
        burn_in, 
        multi=multi,
        fit_alpha=fit_alpha)

    utils.save_dictionary_to_file(calibration_output, r'reports\outputs\MH_output.txt')
    calibration_time_end = time.time()
    calibration_time = calibration_time_end - calibration_time_start
    print('Calibration finished in {} seconds'.format(np.round(calibration_time, 2)))


    ################ MODEL SINGLE RUN (COVID ONLY) #######################################################
    print('Running ABM...')
    model_time_start = time.time()          
    df_model, df_agents = agents_construction.run_model(model=model, epochs=epochs)
    model_time_end = time.time()
    model_time = model_time_end - model_time_start
    print('Run ABM finished in {} seconds'.format(np.round(model_time, 2)))

    ################ PLOTS #######################################################
    time.sleep(3)
    print('Creating Plots...')
    plots_time_start = time.time()
    
    basic_views.plot_graph(G, save = True, title = 'Graph viz')
    basic_views.plot_simulation(df, df_model, start, pct=False, save=True, save_name='Single_run_post')
    basic_views.plot_macro_dynamics(df_model, save = True)
    basic_views.plot_agents_dynamics(df_model, df_agents, title = 'Nest_all_steps', 
                                             hue = 'buy_sell', save = True)
    make_gif.GIF_creator(directory_source = 'reports/figures/nest_dynamics/', 
                         filename = 'Nest_all_steps', 
                         directory_destination = 'reports/figures/')

    plots_time_end = time.time()
    plots_time = plots_time_end - plots_time_start
    print('Plots finished in {} seconds'.format(np.round(plots_time, 2)))
    ##########################################################

    ##########################################################
    ################ MODEL MULTI RUN (COVID ONLY) #######################################################
    
    iterations_multirun = 10
    model_time_start = time.time()          
    results_post_covid = calibration.multi_run(model=model, epochs=epochs, iterations=iterations_multirun)
    model_time_end = time.time()
    basic_views.plot_multi_run(df, results_post_covid['prices'], start, save=True, save_name='Multi_run_post_covid')
    model_time = model_time_end - model_time_start
    print('Multi Run ABM finished in {} seconds'.format(np.round(model_time, 2)))

    ##########################################################
    ################ MODEL MULTI RUN (BEFORE COVID) #######################################################
    model_time_start = time.time()          
    results_pre_covid = calibration.multi_run(model=model_pre, epochs=epochs, iterations=iterations_multirun)
    model_time_end = time.time()
    basic_views.plot_multi_run(df, results_pre_covid['prices'], start_pre, save=True, save_name='Multi_run_pre_covid')
    model_time = model_time_end - model_time_start
    print('Multi Run ABM finished in {} seconds'.format(np.round(model_time, 2)))

    ##########################################################
    ################ BATCH RUN (VARYING N) #######################################################

    print('Starting batch run varying N (number of agents)...')
    iterations_batch = 4
    save = True
    save_name = 'batch_run_graph'
    t1 = time.time()

    all_graphs = []
    if test_N:
        for n in [50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]:

            M = 10
            P = 0.5
            Graph = interaction_builder.graph_generator(type = 'Powerlaw-Cluster',
                        weights_distribution = lambda : weights_distribution, 
                                            **{'n' : n, 'm' : M, 'p' : P})
            all_graphs.append(Graph)

    fixed_params = {
        'k' : k, 
        'price_history' : list(price),
        'external_var': Rt,
        'prob_type':prob_type,
        'alpha' : alpha,
        'p_f' : p_f,
        'cash_low' : cash_low,
        'cash_high' : cash_high,
        'stocks_low' : stocks_low,
        'stocks_high' : stocks_high,
        'debug':False
    }

    variable_params = {
        'interaction_graph' : all_graphs
    }

    reporters = {'price' : calibration.extract_price}

    df_batch_N = calibration.batch_run(
        agents_construction.Nest_Model, variable_params, fixed_params, reporters, 
        epochs, iterations=iterations_batch)
    
    t2 = time.time()

    df_N = utils.group_batch(df_batch_N, 'N', 'price')

    basic_views.plot_aggregate(df, df_N, start, 'N', save=True, save_name='batch_run_N')

    ##########################################################
    ################ BATCH RUN (VARYING GRAPH) #######################################################
    print('Starting batch run varying graph type...')

    graphs = ['Null', 'Clique', 'Erdos-Renyi', 'Powerlaw-Cluster']
    all_graphs = [G_3, G_2, G_1, G_4]

    variable_params = {
        'interaction_graph' : all_graphs
    }
    
    t1 = time.time()

    df_batch_graph = calibration.batch_run(
        agents_construction.Nest_Model, variable_params, fixed_params, reporters, 
        epochs, iterations=iterations_batch)

    df_batch_graph['graph_type'] = [g for g in graphs for _ in range(iterations_batch)]

    t2 = time.time()

    df_graph = utils.group_batch(df_batch_graph, 'graph_type', 'price')

    basic_views.plot_aggregate(df, df_graph, start, 'graph_type', save=True, save_name='batch_run_graph')
    ######################################################################

    tot_time_end = time.time()
    tot_time = tot_time_end - tot_time_start
    print('Computation is finished in a total of {} seconds'.format(np.round(tot_time, 2)))


