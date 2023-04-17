# load here data from financial markets and from the paper
# "The emergence of a collective sensory response threshold in ant colonies" 
# by Gal and Kronauer.
from src import global_configs
import pandas as pd
import os
from zipfile import ZipFile
import epyestim
import epyestim.covid19 as covid19
# save root directory in case it is needed

# create directory if it does not exists
# it should exists

def load_publication_data():
    '''
    Loads data from publication TODO name
    '''
    newpath = global_configs.ROOT_DIR + './data/raw' 
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
    
    df_covid_US = pd.read_csv(covid_data_website_location, )
    
    # reading command
    df_covid_US.to_csv('data/raw/covid_US_raw.csv')

def load_R_number_data():
    filename = 'covid_US_raw.csv'
    df_covid_US = pd.read_csv('data/raw/' 
                            + filename,
                            parse_dates=['date'],  # declare that date is a datetime variable
                            index_col='date'       # set date as index
                            ) # load dataset of cases saved in data/raw folder
    # retrieve series of cases only
    series_covid_US_cases = pd.Series(data = df_covid_US['cases'].values,
                                      name = 'cases',
                                      index = df_covid_US.index
                                      )
    # use epyestim
    df_R_number = covid19.r_covid(series_covid_US_cases) # estimate R number
    df_R_number.to_csv('data/raw/R_number_data.csv')


if __name__ == '__main__':
    load_publication_data()
    load_covid_data()
    load_R_number_data()




