# TODO epysestim notebook on git hub https://github.com/lo-hfk/epyestim
# has some nice plots about the number of cases and the R number
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import pandas as pd
import numpy as np
from  modeling.agents_construction import Ant_Financial_Agent, Nest_Model
from scipy.stats.mstats import winsorize

########## GRAPH STRUCTURE PART ##############
# here we report basic functions to view
# the graph structure chosen

def plot_graph(G,
               show_weights = True,
               save = False, save_name = None, 
               title = 'Graph Visualization'):
    '''
    Plots the graph, either with or without weights
    With the save option allows for prompt saving
    in the reports/figure/ directory
    '''
    pos = nx.spring_layout(G)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=10)
    # edges
    nx.draw_networkx_edges(G, pos, width=6)
    # edge weight labels
    if show_weights:
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.title(title)
    # save figure for later use
    if save:
        if save_name is None:
            plt.savefig('./reports/figures/{}.png'.format(title))
        else:
            plt.savefig('./reports/figures/{}.png'.format(save_name))
    plt.close()
    return None






##############################################

def plot_agents(df, nest_pos, radius = 1, hue = 'utility',
                save = False, save_name = None,
                title = 'A plot'):
    '''
    Takes a nest and plots its agents on a square and its approximate nest. 
    nest is a df_agents resulting from a run call. 
    '''
    # initialize figure
  
    fig, ax = plt.subplots(figsize = (20, 20))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_box_aspect(1)
    ax.set_xlabel('Cash')
    ax.set_ylabel('Stock')
    # plot the nest center
    center_coordinates = nest_pos

    # plot the nest radius circle if it exists
    if radius is not None:
        ax.add_patch(plt.Circle(center_coordinates, radius,
                                color = 'blue',
                                fill=False,
                                lw = 10)
                    )
    # plot each agent position
    ax.scatter(x = df['cash'], y = df['stocks'],
                     s = 0.1,
                    c = df[hue], cmap = plt.cm.get_cmap('RdYlBu')
                    )
    ax.set_title(title)
    # plot also center of nest
    ax.plot(center_coordinates[0], center_coordinates[1],
                'go', label='nest center', markersize = 2)
    ax.legend()
    
    # plt.colorbar() #TODO maybe
    if save:
        if save_name is None: 
            fig.savefig('./reports/figures/{}.png'.format(title))
        else:
            fig.savefig('./reports/figures/{}.png'.format(save_name))
    plt.close()
    return None

def plot_agents_dynamics_diagonal(df_model, df_agents,
                         radius = 0.1, hue = 'utility', 
                         save = False, save_name = None,
                         title = 'A plot'):
    '''
    Takes a nest and plots its agents on a square and its approximate nest. 
    nest is a df_agents resulting from a run call. 
    '''
    # initialize figure
    # plot the nest center
    # epochs = df_model.shape[0]
    nest_centers = df_model.nest_location
    for i,timely_df in df_agents.groupby(level = 0): #extract dataframes according to first index
        # first multiindex is timestep, so we extract time step and data from that step
        fig, ax = plt.subplots()
        ims = []
    
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_box_aspect(1)
        ax.set_xlabel('Cash Proportion')
        ax.set_ylabel('Stock Proportion')
        # get nest center as coordinates in percentages of portfolio
        center_coordinates = nest_centers[i] / sum(nest_centers[i])
        # plot circle of hypothetic nest
        ax.add_patch(plt.Circle(center_coordinates, radius,
                                color = 'green',
                                fill=False,
                                lw = 2)
                    )
        # plot each agent position
        ax.scatter(x = timely_df['x'], y = timely_df['y'],
                     s = 0.1,
                    c = timely_df[hue], cmap = plt.cm.get_cmap('RdYlBu')
                    )
        ax.set_title(title + ' time = {}'.format(i))
        # plot also center of nest
        ax.plot(center_coordinates[0], center_coordinates[1],
                'go', label='nest center', markersize = 2)
        ax.legend()
    # plt.colorbar() #TODO maybe
        if save:
            if save_name is None:
                fig.savefig('../reports/figures/nest_dynamics/{}.png'.format(title + f'_{i}'))
            else:
                fig.savefig('../reports/figures/nest_dynamics/{}.png'.format(save_name + f'_{i}'))

        plt.close()
    return None

def plot_agents_dynamics(df_model, df_agents,
                         radius = 0.1, hue = 'buy_sell', 
                         save = False, save_name = None,
                         title = 'A plot'):
    '''
    Takes a nest and plots its agents on a square and its approximate nest. 
    nest is a df_agents resulting from a run call. 
    '''
    # initialize figure
    # plot the nest center
    # epochs = df_model.shape[0]

    nest_centers = df_model.nest_location
    max_cash = df_agents['cash'].mean() * 2
    max_stocks = df_agents['stocks'].mean() * 2

    # df_agents['cash_std'] = (df_agents['cash_std'] - df_agents['cash_std'].min()) / df_agents['cash_std'].max()
    # df_agents['stocks_std'] = (df_agents['stocks_std'] - df_agents['stocks_std'].min()) / df_agents['stocks_std'].max()
    # nest_centers_x = df_agents.groupby('Step')['cash_std'].apply('median')
    # nest_centers_y = df_agents.groupby('Step')['stocks_std'].apply('median')

    # retrieve box size
    # biggest_x = np.max(df_agents['cash_std'])
    # biggest_y = np.max(df_agents['stocks_std'])
    # smallest_x = np.min(df_agents['cash_std'])
    # smalles_y = np.min(df_agents['stocks_std'])

    # size_square_max = np.max([biggest_x, biggest_y])
    # size_square_min = np.min([smallest_x, smalles_y])
    colors = {-1:'red', 0:'blue', 1:'green'}
    df_agents['col'] = df_agents[hue].apply(lambda x: colors[x])

    for i,timely_df in df_agents.groupby(level = 0): #extract dataframes according to first index
        # first multiindex is timestep, so we extract time step and data from that step
        fig, ax = plt.subplots()
        # ims = []
    
        ax.set_xlim(0,max_cash)
        ax.set_ylim(0,max_stocks)
        ax.set_box_aspect(1)
        ax.set_xlabel('Cash')
        ax.set_ylabel('Stocks')
        # get nest center as coordinates with no normalization
        center_coordinates = nest_centers[i]
        # center_coordinates = nest_centers_x[i], nest_centers_y[i]

        # plot circle of hypothetic nest
        ax.add_patch(plt.Circle(center_coordinates, radius,
                                color = 'black',
                                fill=False,
                                lw = 2)
                    )
        # plot each agent position
        ax.scatter(
            x = np.minimum(timely_df['cash'], max_cash), 
            y = np.minimum(timely_df['stocks'], max_stocks),
            s = 10,
            c = timely_df['col']#, cmap = plt.cm.get_cmap('RdYlBu')
                    )
        ax.set_title(title + ' time = {}'.format(i))
        # plot also center of nest
        ax.plot(center_coordinates[0], center_coordinates[1],
                'go', label='nest center', markersize = 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1.))
    # plt.colorbar() #TODO maybe
        if save:
            if save_name is None:
                fig.savefig('../reports/figures/nest_dynamics/{}.png'.format(title + f'_{i}'),dpi=600,bbox_inches='tight')
            else:
                fig.savefig('../reports/figures/nest_dynamics/{}.png'.format(save_name + f'_{i}'), dpi=600,bbox_inches='tight')

        plt.close()
    return None

def plot_macro_dynamics(df, 
                        save = False, save_name = None,
                        ):
    for col in df.columns:
        plt.plot(df[col], label=col)
        plt.title(col + 'dynamics')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel(col)
    if save:
        if save_name is None:
            plt.savefig('./reports/figures/{}.png'.format(col))
        else:
            plt.savefig('./reports/figures/{}.png'.format(save_name))
    plt.close()
    return None

def plot_simulation(df, df_model, start, pct=False, 
                    save=False, save_name=None):

    price_pre = df.loc[df['start'] <= -start, 'Close']
    price_post = df.loc[df['start'] >= -start, 'Close']
    
    full_price = pd.concat([price_pre, df_model['price']]).reset_index(drop=True)
    idx_start = df.loc[df['start']==0, :].index[0]
    plt.plot(price_post, ls='--', c='orange', linewidth=1.5, label='True data')
    plt.plot(full_price, linewidth=2)
    plt.vlines(price_pre.size, full_price.min(), price_pre.values[-1], ls='dotted', colors='red', label='Simulation starts')
    plt.vlines(idx_start, full_price.min(), full_price[idx_start], ls='dotted', colors='green', label='Covid starts')
    plt.legend()       
    plt.xticks(np.arange(0, df.shape[0], 15), df['Date'][::15], rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock price simulation') 
    if save:
        if save_name is None:
            plt.savefig('../reports/figures/{}.png'.format('Price simulation'), dpi=600,bbox_inches='tight')
        else:
            plt.savefig('../reports/figures/{}.png'.format(save_name), dpi=600,bbox_inches='tight')
    plt.close()
    return None


def plot_multi_run(df, results, start, pct=False, 
                    save=False, save_name=None, title='A plot'):

    price_pre = df.loc[df['start'] <= -start, 'Close']
    price_post = df.loc[df['start'] >= -start, 'Close']

    # index where simulation begins
    idx = df.loc[df['start'] == -start, 'Close'].index[0]

    # index where covid begins
    idx_start = df.loc[df['start']==0, :].index[0]

    # mean and std of simulations
    sim_mean = results.mean(axis=0)
    sim_std = results.std(axis=0)

    # full price concatenation
    full_price = pd.concat([price_pre, pd.Series(sim_mean)]).reset_index(drop=True)

    # index for plotting area
    x = np.arange(idx, idx + sim_mean.size)


    plt.plot(price_post, ls='--', c='orange', linewidth=1.5, label='True data')
    plt.plot(full_price, linewidth=2)

    plt.fill_between(x, sim_mean + sim_std, sim_mean - sim_std, alpha=0.5)
    plt.vlines(price_pre.size, full_price.min(), price_pre.values[-1], ls='dotted', colors='red', label='Simulation starts')
    plt.vlines(idx_start, full_price.min(), full_price[idx_start], ls='dotted', colors='green', label='Covid starts')
    plt.xticks(np.arange(0, df.shape[0], 15), df['Date'].values[::15], rotation=90)
    plt.legend()       
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title) 
    # plt.show()
    if save:
        if save_name is None:
            plt.savefig('../reports/figures/{}.png'.format('Price_simulation'), dpi=600,bbox_inches='tight')
        else:
            plt.savefig('../reports/figures/{}.png'.format(save_name), dpi=600,bbox_inches='tight')
    plt.close()
    return None

def plot_aggregate(df, df_batch, start, col, plot_true=False,
                    save=False, save_name=None, title='A plot'):

    price_pre = df.loc[df['start'] <= -start, 'Close']
    price_true = df.loc[df['start'] >= -start, 'Close']

    # index where simulation begins
    idx = df.loc[df['start'] == -start, 'Close'].index[0]
    idx_start = df.loc[df['start']==0, :].index[0]

    sim_size = df.index[-1]
    x = np.arange(idx, sim_size+1)

    plt.plot(price_pre, linewidth=2)

    min_price = min(price_pre)

    for i in range(df_batch.shape[0]):
        n = df_batch.loc[i, col]
        price_post = df_batch.loc[i, 'price'][idx:]
        min_post = min(price_post)
        if min_post < min_price:
            min_price = min_post
        plt.plot(x, price_post, ls='--', label=n)


    if plot_true:
        plt.plot(x, price_true, ls='--', linewidth=1.5, label='True data')

    # plt.vlines(price_pre.size, min_price, price_pre.values[-1], ls='dotted', colors='red', label='Simulation starts')
    plt.vlines(idx_start, min_price, price_true[idx_start], ls='dotted', colors='green', label='Covid starts')
    plt.xticks(np.arange(0, df.shape[0], 15), df['Date'].values[::15], rotation=90)
    plt.legend()       
    plt.xlabel('Date')
    plt.ylabel('Price')
    # ax = plt.gca()  # Get the current axes
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.title(title) 
    if save:
        if save_name is None:
            plt.savefig('../reports/figures/{}.png'.format('Price_simulation'), dpi=600,bbox_inches='tight')
        else:
            plt.savefig('../reports/figures/{}.png'.format(save_name), dpi=600,bbox_inches='tight')
    plt.close()
    return None
