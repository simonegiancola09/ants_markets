import networkx as nx
import numpy as np

def graph_generator(type = 'Erdos-Renyi',
                    weights_distribution = lambda : np.random.normal(0,1),
                    **params):
    '''
    Generates a networkx instance of some graph types
    weights_distribution must be some function that assigns
    a weight to two nodes, it might also be dependent on the nodes
    but in the end we avoid this case for simplicity
    '''
    potential_graphs = ['Erdos-Renyi',
                       'Null',
                       'Clique',
                       'Powerlaw-Cluster',
                       'Custom'
                       ]
    
    if type not in potential_graphs:
        raise Exception('type {} is not allowed, please choose one of {}'.format(type, potential_graphs))
    print('params are', params)
    if type == 'Erdos-Renyi':
        # need n, p params
        G = nx.erdos_renyi_graph(params['n'], params['p'])
        for e in G.edges():
            G[e[0]][e[1]]['weight'] = weights_distribution()
    elif type == 'Null':
        # need n parameter
        G = nx.empty_graph(params)
        # no edge weights 
    elif type == 'Clique':
        # need n parameter
        G = nx.complete_graph(params)
        for e in G.edges():
            G[e[0]][e[1]]['weight'] = weights_distribution()
    elif type == 'Powerlaw-Cluster':
        # need n, m, p params
        nx.powerlaw_cluster_graph(params)
        for e in G.edges():
            G[e[0]][e[1]]['weight'] = weights_distribution()
    else:
        # graph is custom, give an adjacency matrix
        G = nx.from_numpy_matrix(np.matrix(params['A'])) 
        for e in G.edges():
            # in the case of custom edges we allow
            # for potentially edge specific relations
            # but this increases the computational overhead
            G[e[0]][e[1]]['weight'] = weights_distribution(e)
    return G
    
