from modeling import agents_construction
from engineering import interaction_builder
import matplotlib.pyplot as plt
import numpy as np


N = 200
p = 0.3
epochs = 500
beta = 1
initial_stock_price = 1
num_stocks = 200000
rt = np.sin(np.linspace(-5,5, epochs))+1
G = interaction_builder.graph_generator(type = 'Erdos-Renyi',
                    weights_distribution = lambda : np.random.uniform(0,1),
                    **{'n':N, 'p':p})

model = agents_construction.Nest_Model(
                                    beta=beta, 
                                    initial_stock_price = initial_stock_price,
                                    external_var = rt,
                                    interaction_graph = G,
                                    num_stocks = num_stocks)

df_model, agents_df = agents_construction.run_model(model, epochs=epochs)


