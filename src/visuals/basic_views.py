# TODO epysestim notebook on git hub https://github.com/lo-hfk/epyestim
# has some nice plots about the number of cases and the R number
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from  src.modeling.agents_construction import Ant_Financial_Agent, Nest_Model

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
                fig.savefig('./reports/figures/nest_dynamics/{}.png'.format(title + f'_{i}'))
            else:
                fig.savefig('./reports/figures/nest_dynamics/{}.png'.format(save_name + f'_{i}'))

        plt.close()
    return None

def plot_agents_dynamics(df_model, df_agents,
                         radius = 10, hue = 'utility', 
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
    # retrieve box size
    biggest_x = np.max(df_agents['cash'])
    biggest_y = np.max(df_agents['wealth'] - df_agents['cash'])
    size_square = np.max([biggest_x, biggest_y])

    for i,timely_df in df_agents.groupby(level = 0): #extract dataframes according to first index
        # first multiindex is timestep, so we extract time step and data from that step
        fig, ax = plt.subplots()
        # ims = []
    
        ax.set_xlim(0,size_square)
        ax.set_ylim(0,size_square)
        ax.set_box_aspect(1)
        ax.set_xlabel('Cash')
        ax.set_ylabel('Stock')
        # get nest center as coordinates with no normalization
        center_coordinates = nest_centers[i] 
        # plot circle of hypothetic nest
        ax.add_patch(plt.Circle(center_coordinates, radius,
                                color = 'green',
                                fill=False,
                                lw = 2)
                    )
        # plot each agent position
        ax.scatter(x = timely_df['cash'], y = timely_df['wealth'] - timely_df['cash'],
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
                fig.savefig('./reports/figures/nest_dynamics/{}.png'.format(title + f'_{i}'))
            else:
                fig.savefig('./reports/figures/nest_dynamics/{}.png'.format(save_name + f'_{i}'))

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
