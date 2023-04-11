# load here data from financial markets and from the paper
# "The emergence of a collective sensory response threshold in ant colonies" 
# by Gal and Kronauer.

import pandas as pd
import os
from zipfile import ZipFile
# create directory if it does not exists
# it should exists
newpath = r'./data/raw' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# change directory
os.chdir('./data/raw')
# call zenodo function to download zip data

# os.makedirs('./publication_data')
with ZipFile('Gal2022_data_and_code.zip', 'r') as zip_ref:
    zip_ref.extractall('./publication_data/')
zip_ref.close()
# TODO unzip and organize folders by deleting the useless ones

