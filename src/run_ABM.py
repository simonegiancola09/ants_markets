from src.modeling import agents_construction
from src.engineering import interaction_builder
import matplotlib.pyplot as plt
import numpy as np




if __name__ == '__main__':
    N = 100
    p = 0.6
    epochs = 1000
    beta = 1
    initial_stock_price = 1
    num_stocks = 200000
    rt = np.sin(np.linspace(-5,5, epochs))+1
    rt = np.linspace(0,2, num = epochs + 1)
    rt = np.zeros(epochs + 1)
    G = interaction_builder.graph_generator(type = 'Erdos-Renyi',
                        weights_distribution = lambda : np.random.uniform(0,1),
                        **{'n':N, 'p':p})

    model = agents_construction.Nest_Model(
                                        beta=beta, 
                                        initial_stock_price = initial_stock_price,
                                        external_var = rt,
                                        interaction_graph = G,
                                        num_stocks = num_stocks)

    df_model, df_agents = agents_construction.run_model(model=model, epochs=epochs)



