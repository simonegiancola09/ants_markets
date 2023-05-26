from modeling import agents_construction, calibration
from engineering import interaction_builder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from utils import utils
from visuals import basic_views, make_gif


if __name__ == '__main__':

    np.random.seed(1999)
    df_price = pd.read_csv(r'..\data\raw\financial_US_NVDA_raw.csv')
    # df_covid = pd.read_csv(r'..\data\raw\covid_US_raw.csv')
    df_rt = pd.read_csv(r'..\data\engineered\df_covid.csv')

    df_rt.columns = ['date', 'cases', 'Rt', 'R_var']
    # df_covid['daily_cases'] = df_covid['cases'].diff().fillna(0)

    # compute daily cases and pct change in daily cases

    window = 5

    # merge two datasets
    df = pd.merge(df_price, df_rt, left_on='Date', right_on='date', how='left')

    df = utils.preprocess_data(df)

    ## PARAMS OF THE MODEL ##

    # days before pandemic starts
    start = 2

    # num of iterations            
    epochs = df['start'].max() + start 

    # epochs = 10
    run_batch = True
    mh = False
    multi_run = False
    debug = False
    price = df.loc[df['start'] <= -start, 'Close']
    graph_type = None
    # number of nodes
    N = 600  
    # num of edges per node for cluster          
    M = 10      
    # p of connection for erdos renyi and cluster graph            
    P = 0.5  


    # parameter controlling ROLE OF NEIGHBORS ON PRICE PROPOSAL offers
    # k = 0.09
    # alpha = -0.8
    #        
    k = .105
    ## parameter controlling role of Temperature on buy/sell
    alpha = -0.69

    # distribution over types of traders
    prob_type = [1., 0., 0.0, 0.0]
    # price value for fundamentalist traders 
    # (set to None to automatically give mean price of price history)
    p_f = 75

    ## paramenters on distribution of wealth
    cash_low = 50000
    # cash_low = 10000
    cash_high = 100000
    stocks_low = 1000
    stocks_high = 100

    Rt = df.loc[df['start'] > -start, 'change_daily'].values

    ######################
    print('Building graphs')
    weights_distribution = np.random.uniform(0.8,1.2)
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

    model = utils.build_model(df, start, fixed_kwargs)

    print('Running ABM...')

    if mh:
        print('Starting MH...')
        filename = 'final_MH'

        t1 = time.time()
        std = 0.2
        param_start = -0.5
        iterations = 150
        internal_iterations = 5
        true_data = df.loc[df['start'] > -start, 'Close']
        burn_in = 0
        multi=False
        fit_alpha=True
        loss = calibration.compute_log_likelihood
        preprocess = True
        results_mh = calibration.metropolis_hastings(
            model, 
            iterations, 
            internal_iterations,
            loss,
            preprocess,
            param_start, 
            std, 
            true_data, 
            burn_in, 
            multi=multi,
            fit_alpha=fit_alpha)

        t2 = time.time()
        elapsed = (t2 - t1) / 60
        print('MH running time:', np.round(elapsed, 2), ' minutes')
        utils.save_dictionary_to_file(results_mh, rf'..\reports\outputs\{filename}.txt')


    if run_batch:

        print('Starting batch run...')
        iterations = 4
        save = True
        test_N = False
        test_graph=True
        save_name = 'batch_run_N'
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
        if test_graph:
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
            graphs = ['Null', 'Clique', 'Erdos-Renyi', 'Powerlaw-Cluster']
            all_graphs = [G_3, G_2, G_1, G_4]
 


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

        df_batch = calibration.batch_run(
            agents_construction.Nest_Model, variable_params, fixed_params, reporters, 
            epochs, iterations=iterations)

        

        t2 = time.time()
        if test_graph:

            df_batch['graph_type'] = [g for g in graphs for _ in range(iterations)]

        elapsed1 = (t2 - t1) / 60
        print('Batch running time:', np.round(elapsed1, 2), ' minutes')


        if save:
            df_batch.to_csv(r'..\reports\outputs\{}.csv'.format(save_name))

        grouped_batch = utils.group_batch(df_batch, 'graph_type', 'price')
        basic_views.plot_aggregate(df, grouped_batch, start, 'graph_type', plot_true=False,
                    save=True, save_name='batch_graph', title='Stock price simulation - varying graph')

    if multi_run:

        iterations = 10
        results_multi = calibration.multi_run(model, epochs, iterations)
        basic_views.plot_multi_run(df, results_multi['prices'], start,
        save=True, save_name='multi_run_pre', title='Stock price simulation - multiple runs')


    else:
        df_model, df_agents = agents_construction.run_model(model=model, epochs=epochs)
        
        basic_views.plot_simulation(df, df_model, start, pct=False, 
                    save=False, save_name=None)

        basic_views.plot_agents_dynamics(
                        df_model, df_agents,
                         radius = 1, hue = 'buy_sell', 
                         save = True, save_name = None,
                         title = 'A plot')

        make_gif.GIF_creator(
            directory_source=r'../reports/figures/nest_dynamics/', 
            filename='GIF1', 
            directory_destination=r'../reports/figures/nest_dynamics/', 
            duration=50)
        
        
        full_price = pd.concat([price, df_model['price']]).reset_index(drop=True)


