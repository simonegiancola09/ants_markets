# supposedly, main file for running everything from one script when we are ready
from src import global_configs
from src import data_loading
from src.engineering import create_main_df
global_configs.init()


if __name__ == '__main__':
    # here we can attempt to use a function written in another script as below
    print('Starting main.py file. Welcome to the pipeline for our project...')
    create_main_df.retrieve_cases_and_R_number()