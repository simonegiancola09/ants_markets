import os
import sys
from src.modeling import agents_construction
import numpy as np
import pandas as pd

def build_model(df, start, kwargs, T='change_daily'):
    price = df.loc[df['start'] <= -start, 'Close']
    Rt = df.loc[df['start'] > -start, T].values
    model = agents_construction.Nest_Model(external_var=Rt,
                                    price_history=list(price), 
                                    **kwargs)
    return model

def save_dictionary_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    
def preprocess_data(df, window=5):
    df['cases'] = df['cases'].fillna(0)
    df['daily_cases'] = df['cases'].diff().fillna(0)
    df['change_daily'] = df['daily_cases'].rolling(window=window, min_periods=1).mean().pct_change()
    df.loc[np.isinf(df['change_daily']), 'change_daily'] = 0.1

    df['start'] = 0
    idx_covid = df['date'].isna()==False

    df.loc[idx_covid, 'start'] = np.arange(0, idx_covid.sum())
    df.loc[df['date'].isna(), 'start'] = np.arange(-1, -(df['date'].isna().sum()+1), -1)[::-1]

    ## compute Rt percentage change
    df['change'] = df['Rt'].pct_change()
    df.loc[df['start'] == 0, 'change'] = 0.1

    df = df.fillna(0)

    return df

def group_batch(df_batch, group_by, col):
    df_batch['N'] = df_batch['interaction_graph'].apply(lambda x: len(x.nodes()))
    grouped = df_batch.groupby(group_by)[col].apply(lambda x: np.mean(x.tolist(), axis=0)).reset_index()
    return grouped


        