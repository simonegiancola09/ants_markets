# load here data from financial markets and from the paper
# "The emergence of a collective sensory response threshold in ant colonies" 
# by Gal and Kronauer.

import pandas as pd
import os
from zipfile import ZipFile
# save root directory in case it is needed
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# create directory if it does not exists
# it should exists
newpath = r'./data/raw' 
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
os.chdir(ROOT_DIR)
# the website is long and ugly looking so we compose it for better reading
covid_data_website_location = 'https://github.com/CSSEGISandData/COVID-19'
covid_data_website_location += '/blob/master/csse_covid_19_data/csse_covid_19_time_series'
covid_data_website_location += '/time_series_covid19_confirmed_US.csv'
df_covid_US = pd.read_csv(covid_data_website_location)
# save data for first time use to never download it again locally
df_covid_US.to_csv('data/raw/covid_US_raw.csv')