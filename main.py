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
    print("Creating the necessary folders...")
    # if windows OS maybe the first option is better
    windows = False
    if windows:
        utils.create_directory(r'\reports')
        utils.create_directory(r'\reports\figures')
        utils.create_directory(r'\reports\outputs')
        utils.create_directory(r'\reports\figures\nest_dynamics')
    else:
        utils.create_directory('./reports')
        utils.create_directory('./reports/figures')
        utils.create_directory('./reports/outputs')
        utils.create_directory('./reports/figures/nest_dynamics')

    ################################################################################################################
    tot_time_start = time.time()
    
    # Please choose here the stock and the dates for calibration 
    stock_name = 'NVDA'                     # name of stock
    dates = ('2019-12-01', '2020-06-30')    # dates for stock data, must be larger than what we
                                            # use for covid to calibrate
    print('Your stock of choice is {} in the historical dates {}'.format(stock_name, dates))
    
    ############## DATA LOADING ###########################################
    # run only once, then set as True
    loaded = False                          
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
    df_stocks = pd.read_csv('data/raw/financial_US_{}_raw.csv'.format(stock_name),
                            parse_dates=['Date'])
    ##########################################################
    
    ############## Covid Data ###############################
    print('Retrieving Covid-19 data...')
    # this dataset has more information than just the Rt
    df_covid = pd.read_csv('data/engineered/df_covid.csv',
                            parse_dates=True, index_col=0)

    ############# Merge Datasets ############################
    df = pd.merge(df_stocks, df_covid, left_on='Date', right_index=True, how='left')
    # proprocess data to obtain the "temperature" values
    df.head()
    df = utils.preprocess_data(df) 
    ##########################################################
    ##### QUICK ATTEMPT VS HEAVY SIMULATION ##################
    # to let users try the model we will report here two combinations
    # of parameters that highly influence the duration of the script
    # once one chooses the parameter below it will interpolate
    # between two main configs
    short_sim = 0          # if 1 very short simulation
    if short_sim:
        print('Today we will just check that everything works as expected.')
        print('Do not trust these small size results')
        time.sleep(3)
    else:
        print('Today we will run a heavy simulation')
        print('it might take a lot of time if your PC is not powerful')
        time.sleep(3)


    ############## HYPERPARAMETERS ###############################
    print('Setting hyperparameters...')
    # here we store some parameters of choice
    #### General hypeparams ############
    N = 20 * short_sim + 600 * (1 - short_sim)    # num of nodes
    M = 10                                      # num of edges per node for cluster
    P = 0.5                                     # p of connection for erdos renyi and cluster graph
    debug=False                                 # set to True to visualize supply/demand curves every 5 steps
    ##########
    #### Ant-Like hypeparams ############
    alpha = -0.69                       # parameter controlling how much the temperature contributes
    ##########
    #### DemandSupply-Like hypeparams ############
    k = 0.105                           # parameter controlling variance of price decision for random traders        
    prob_type = [1., 0.0, 0.0, 0.0]     # distribution over types of traders
    p_f = None                          # price value for fundamentalist traders 
                                        # (set to None to automatically give mean price of price history)
    cash_low = 50000                    # paramenters on distribution of wealth
    cash_high = 200000
    stocks_low = 1000
    stocks_high = 5000
    #########################
    
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
    price = df.loc[df['start'] <= -start, 'Close']
    Rt = df.loc[df['start'] > -start, 'change_daily'].values
      
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
    print('Building the first model...')
    model = utils.build_model(df, start, fixed_kwargs)
    model_pre = utils.build_model(df, start_pre, fixed_kwargs)

    ############# CALIBRATION ##################################
    '''
    time.sleep(3)
    print('Calibration of Pandemic contribution...')
    calibration_time_start = time.time()

    std = 0.2
    param_start = -0.5
    iterations_mh = 3 * short_sim + 500 * (1 - short_sim) 
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
    if windows:
        utils.save_dictionary_to_file(calibration_output, r'reports\outputs\MH_output.txt')
    else:
        utils.save_dictionary_to_file(calibration_output, 'reports/outputs/MH_output.txt')
 
    calibration_time_end = time.time()
    calibration_time = calibration_time_end - calibration_time_start
    print('Calibration finished in {} seconds'.format(np.round(calibration_time, 2)))
    '''

    ### UPDATE ALPHA ACCORDING TO THE VALUE FOUND IN CALIBRATION ####
    # alpha = calibration_output['parameter estimate']
    alpha = -1.2664124139390913     # value found
    ################ MODEL SINGLE RUN (COVID ONLY) #######################################################
    print('Running ABM...')
    model_time_start = time.time()          
    df_model, df_agents = agents_construction.run_model(model=model, epochs=epochs)
    model_time_end = time.time()
    model_time = model_time_end - model_time_start
    print('Run ABM finished in {} seconds'.format(np.round(model_time, 2)))
    #######################################################
    ################ PLOTS #######################################################
    time.sleep(3)
    
    print('Creating Plots...')
    plots_time_start = time.time()
    
    basic_views.plot_graph(G, save = True, title = 'Graph viz')
    basic_views.plot_simulation(df, df_model, start, pct=False, save=True, save_name='Single_run_post')
    basic_views.plot_macro_dynamics(df_model, save = True, save_name='first_attempt')
    basic_views.plot_agents_dynamics(df_model, df_agents, title = 'Nest_all_steps', 
                                             hue = 'buy_sell', save = True)
    
    print('Making a GIF might take a lot of time...')
    make_gif.GIF_creator(directory_source = 'reports/figures/nest_dynamics/', 
                         filename = 'Nest_all_steps', 
                         directory_destination = 'reports/figures/',
                         duration=50)
    plots_time_end = time.time()
    plots_time = plots_time_end - plots_time_start
    print('Plots finished in {} seconds'.format(np.round(plots_time, 2)))
    
    ##########################################################

    ##########################################################
    ################ MODEL MULTI RUN (COVID ONLY) #######################################################
    print('Starting ABM multiple runs (Covid period only)...')

    iterations_multirun = 2 * short_sim + 10 * (1 - short_sim)
    model_time_start = time.time()          
    results_post_covid = calibration.multi_run(model=model, epochs=epochs, iterations=iterations_multirun)
    model_time_end = time.time()
    basic_views.plot_multi_run(df, 
                               results_post_covid['prices'], 
                               start, save=True, 
                               save_name='Multi_run_post_covid',
                                title='Stock price simulation - multiple runs'
                                )
    model_time = model_time_end - model_time_start
    print('Multi Run ABM finished in {} seconds'.format(np.round(model_time, 2)))

    ##########################################################
    ################ MODEL MULTI RUN (BEFORE COVID) #######################################################
    print('Starting ABM multiple runs...')

    model_time_start = time.time()          
    results_pre_covid = calibration.multi_run(model=model_pre, epochs=epochs, iterations=iterations_multirun)
    model_time_end = time.time()
    basic_views.plot_multi_run(df, 
                                results_pre_covid['prices'], 
                                start_pre, save=True, 
                                save_name='Multi_run_pre_covid',
                                title='Stock price simulation - multiple runs'
                                )
    model_time = model_time_end - model_time_start
    print('Multi Run ABM finished in {} seconds'.format(np.round(model_time, 2)))

    ##########################################################
    ################ BATCH RUN (VARYING N) #######################################################

    print('Starting batch run varying N (number of agents)...')
    start_time_varying_N = time.time()
    iterations_batch = 2 * short_sim + 4 * (1 - short_sim)
    save = True
    save_name = 'batch_run_graph'
    t1 = time.time()
    test_N = True                   # may take a lot of time
    all_graphs = []
    if not short_sim:
        # attention, very long!!!!!!!!
        n_list = [50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]
    else:
        n_list = [50, 100, 250]
    for n in n_list: 
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

    basic_views.plot_aggregate(df, df_N, start, 'N', save=True, save_name='batch_run_N',
    title='Stock price simulation - varying size')
    end_time_varying_N = time.time()
    tot_time_varying_N = end_time_varying_N - start_time_varying_N
    print('Varying number of investors run finished in {} seconds'.format(np.round(tot_time_varying_N, 2)))

    ##########################################################
    ################ BATCH RUN (VARYING GRAPH) #######################################################
    print('Starting batch run varying graph type...')
    start_time_varying_graph = time.time()
    iterations_batch = 2 * short_sim + 10 * (1 - short_sim) 
    graphs = ['Null', 'Clique', 'Erdos-Renyi', 'Powerlaw-Cluster']
    all_graphs = [G_3.copy(), G_2.copy(), G_1.copy(), G_4.copy()]

    variable_params = {
        'interaction_graph' : all_graphs
    }
    
    df_batch_graph = calibration.batch_run(
        agents_construction.Nest_Model, variable_params, fixed_params, reporters, 
        epochs, iterations=iterations_batch)

    df_batch_graph['graph_type'] = [g for g in graphs for _ in range(iterations_batch)]

    df_graph = utils.group_batch(df_batch_graph, 'graph_type', 'price')

    basic_views.plot_aggregate(df, df_graph, start, 'graph_type', save=True, 
                            save_name='batch_run_graph', title='Stock price simulation - varying graph')
    end_time_varying_graph = time.time()
    tot_time_varying_graph = end_time_varying_graph - start_time_varying_graph
    print('Multi Run ABM finished in {} seconds'.format(np.round(tot_time_varying_graph, 2)))
    ######################################################################

    tot_time_end = time.time()
    tot_time = tot_time_end - tot_time_start
    print('Computation is finished in a total of {} seconds'.format(np.round(tot_time, 2)))



