# TODO epysestim notebook on git hub https://github.com/lo-hfk/epyestim
# has some nice plots about the number of cases and the R number
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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