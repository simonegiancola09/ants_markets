# load here data from financial markets and from the paper
# "The emergence of a collective sensory response threshold in ant colonies" 
# by Gal and Kronauer.
import global_configs
import pandas as pd
import os
from zipfile import ZipFile

import yfinance as yf





def load_publication_data():
    '''
    Loads data from publication TODO name
    '''
    # save root directory in case it is needed
    newpath = global_configs.ROOT_DIR + './data/raw'
    # create directory if it does not exists
    # it should exist
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    # change directory
    os.chdir('./data/raw')
    # call zenodo function to download zip data
    os.system('zenodo_get 10.5281/zenodo.6569620')
    with ZipFile('Gal2022_data_and_code.zip', 'r') as zip_ref:
        zip_ref.extractall('./publication_data/')
    zip_ref.close()
    # TODO unzip and organize folders by deleting the useless ones


###### COVID DATA LOADING ######
def load_covid_data():
    '''
    Loads data from US confirmed cases of covid available on Github
    from the NY times dataset
    '''
    os.chdir(global_configs.ROOT_DIR)
    # the website is long and ugly looking so we compose it for better reading
    covid_data_website_location = 'https://raw.githubusercontent.com'
    covid_data_website_location += '/nytimes/covid-19-data/master/us.csv'

    # save data for first time use to never download it again locally
    
    df_covid_US = pd.read_csv(covid_data_website_location)
    
    # reading command
    df_covid_US.to_csv('data/raw/covid_US_raw.csv')

######## FINANCIAL DATA LOADING ########################
def load_financial_data(stock_name = 'NVDA', dates = ('2020-01-01', '2020-03-31')):
    '''
    Load Financial Data from a common stock to populate the dataframe, choose dates accordingly
    '''
   

    # Get data via Yahoo Finance API
    df_financial = yf.get(stock_name, dates[0], dates[1])
    # save
    df_financial.to_csv('data/raw/financial_US_{}_raw.csv'.format(stock_name))

################################################

if __name__ == '__main__':
    load_publication_data()
    load_covid_data()
    load_financial_data()




