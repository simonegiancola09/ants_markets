# supposedly, main file for running everything from one script when we are ready
from src import global_configs
from src import data_loading
from src.engineering import create_main_df
from src.engineering import interaction_builder
from src.visuals import basic_views
global_configs.init()


if __name__ == '__main__':
    # here we can attempt to use a function written in another script as below
    print('Starting main.py file. Welcome to the pipeline for our project...')
    # create_main_df.retrieve_cases_and_R_number()
    G = interaction_builder.graph_generator(n = 10, p = 0.5)
    fig = basic_views.plot_graph(G, save = True)
    fig.show()
    print('Computation is finished')