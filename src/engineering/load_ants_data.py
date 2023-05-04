import pandas as pd
import os

def load_events_data():
    '''
    Load ants' events data 3 DataFrames: 

    1)  events_collectivity dataset. 
        For each event in the dataset, the following fields are provided:
        * colony - colony identifier
        * N - number of ants in colony
        * S - the set temperature of the perturbation
        * thermistor - the time course of the thermistor measurements (spatially averaged)
        * thcam - the time course of the thermal camera measurements (spatially averaged)
        * ant_binary - a vector of the binary response of each ant
        * ant_latency - a vector of response latency for each ant
        * ant_direction - a vecor of response direction as defined in the paper

    2)  events_threshold_tagged.
        For each event in the dataset, the following fields are provided:
        * colony - colony identifier
        * N - number of ants in colony
        * S - the set temperature of the perturbation
        * thermistor - the time course of the thermistor measurements (spatially averaged)
        * thcam - the time course of the thermal camera measurements (spatially averaged)
        * Np - approximate number of ants outside the nest for each time period
        * collective_binary - 1 if > 90% of the ants evacuated the nest, 0 otherwise

    3)  events_groupsize_untagged.
        For each event in the dataset, the following fields are provided:
        * colony - colony identifier
        * N - number of ants in colony
        * S - the set temperature of the perturbation
        * thermistor - the time course of the thermistor measurements (spatially averaged)
        * thcam - the time course of the thermal camera measurements (spatially averaged)
        * Np - approximate number of ants outside the nest for each time period
        * collective_binary - 1 if > 90% of the ants evacuated the nest, 0 otherwise

    '''

    filename1 = 'events_collectivity.h5'
    filename2 = 'events_groupsize_untagged.h5'
    filename3 = 'events_threshold_tagged.h5'

    events_collectivity = pd.read_hdf('data/raw/' + filename1, key='events')
    events_untagged = pd.read_hdf('data/raw/' + filename2, key='events')
    events_tagged = pd.read_hdf('data/raw/' + filename3, key='events')

    if exclude_perturbed:
        good = [ev['Np'][0]<N/2 for ix, ev in events_tagged.iterrows()]
        events_tagged = events_tagged[good]

    events_untagged['collective_binary'] = 0
    for ix, ev in events_untagged.iterrows():
        if ev['Np'].max() == 36:
            events_gs.loc[ix, 'Np'][:] = ev['Np'] * ev['N'] / 36
        ev['collective_binary'] = ((ev['Np'] / ev['N']) >= 0.9).astype(int)
    return events_collectivity.reset_index(drop=True), events_tagged.reset_index(drop=True), events_untagged.reset_index(drop=True)

def load_relaxation_data(normalize=True):
    '''
    Collection of 24 experiments in which colonies of 36 ants are perturbed at 40°C for a period of 15 minutes during a timeframe of 2 hours.
    Temperature perturbation starts at index 3000.

    Parameters:
    -----------
    normalize : bool
        Normalizes data between 0 and 1, default True. 

    Returns:
    --------
    X : numpy.array
        Array with data loaded.
    '''
    filename = 'RelaxationFigureData.txt'
    X = np.loadtxt('data/raw/' + filename)
    if normalize:
        X = X / 36
    return X







    