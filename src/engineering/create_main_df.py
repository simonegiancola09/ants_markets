# in this notebook, we perform operations on the 
# raw dataframes to make them ready for our analysis

###### R number estimator loading ####

# we retrieve a function for the R number from 
# the package epiestym that makes use of three other works
# see their website https://github.com/lo-hfk/epyestim for details

import epyestim
import epyestim.covid19 as covid19
import pandas as pd
import os

def retrieve_cases_and_R_number():
    '''
    Joins the datasets to have the external factors dataset 
    (i.e. temperature and time of temperature, where in our case
    temperature is the R number).
    '''
    path = 'data/raw/'
    filename_covid_cases = 'covid_US_raw.csv'
    filename_R_number = 'R_number_data.csv'

    df_covid_US = pd.read_csv(path + filename_covid_cases,
                              index_col='date')
    df_R_number = pd.read_csv(path + filename_R_number, 
                              index_col=0 # date is first column in df
                              )
    df_R_number.index.name = 'date'

    # set both indexes to datetime
    df_R_number.index = pd.to_datetime(df_R_number.index)
    df_covid_US.index = pd.to_datetime(df_covid_US.index)
    # retrieve only the data we are interested in
    df_covid_US_cases = df_covid_US[['cases']]
    df_R_number_cases = df_R_number[['R_mean','R_var']]

    df_covid = pd.merge(df_covid_US_cases,df_R_number_cases,    # which dfs
                        how='inner',                            # how (i.e. keep only intersection)
                        left_index=True,right_index=True)       # keep indexes
    
    # check that there are non NaN values
    # not really needed, just for sanity
    assert df_covid.isnull().values.any() == False   






