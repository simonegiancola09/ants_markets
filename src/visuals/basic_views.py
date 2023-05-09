# TODO epysestim notebook on git hub https://github.com/lo-hfk/epyestim
# has some nice plots about the number of cases and the R number
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from  modeling.agents_construction import Ant_Financial_Agent, Nest_Model

########## GRAPH STRUCTURE PART ##############
# here we report basic functions to view
# the graph structure chosen

def plot_graph(G,
               show_weights = True,
               save = False, 
               title = 'A plot'):
    '''
    Plots the graph, either with or without weights
    With the save option allows for prompt saving
    in the reports/figure/ directory
    '''
    pos = nx.spring_layout(G)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
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
        plt.savefig('./reports/figures/{}.png'.format(title))
    return plt






##############################################

def plot_agents(nest, save = False, title = 'A plot'):
    '''
    Takes a nest and plots its agents on a square and its approximate nest. 
    nest is a NestModel instance
    '''
    if not isinstance(nest, Nest_Model):
        raise Exception('Please use as input a NestModel instance')
    fig, ax = plt.figure(figsize = (20, 20))
    ax.set_xlim(0,100)
    ax.set_box_aspect(1)
    # plot the nest center
    center_coordinates = nest.get_nest_location()

    # plot the nest radius circle if it exists
    try:
        radius = nest.radius
    except:
        radius = False
    if radius:
        ax.add_patch(plt.Circle(center_coordinates, radius,
                                c = 'blue',
                                fill=False,
                                lw = 10)
                    )
    # plot each agent position
    for investor in nest.grid.get_all_cell_contents():
        investor_coordinates = investor.pos
        investor_state = investor.state
        # plot a marker, color is chosen by the investor state
        ax.plot(investor_coordinates[0], investor_coordinates[1],
                (investor_state == -1) * 'bo' + (investor_state == +1) * 'ro', 
                markersize = 10
                )
    ax.set_title(title)
    if save: 
        plt.savefig('./reports/figures/{}.png'.format(title))



