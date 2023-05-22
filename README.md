# ants_markets

Simulation and Modeling exam Project


Git commands to work on it
------------
We take the simplest possible structure for now, and see how it goes. 
There will be one remote branch (i.e. `main`) and one local branch on our 
computer (i.e. `main` as well). 

To start we go to a folder we want to work on and call on the terminal 
`git clone (link to clone repository)`. This will copy over the project in 
our local pc. 
We then need to create a virtual env to keep the project independent from the other
Python packages, for this, while on the main folder, we do:
`python3 -m venv local_python_environment`
`source local_python_environment/bin/activate`
`pip install -r requirements.txt`
This takes the `requirements.txt` file and adds the packages to the local environment. 
Whenever we will `pip install` a new package it will then be on our local machine. Thanks to the next command, 
it will however also be saved in the `requirements.txt` file for later use.

Then suppose we make some changes on the code, and save them, we would 
like to apply those changes. 
If we added a package we must first call 
`pip freeze > requirements.txt`
making sure that the file requirements is saved on the main directory (i.e. it is not in any secondary folder)

Secondly, we would like to make the changes also in the remote folder on Github to make them accessible for every 
collaborator. To do so we call on the terminal
`git add .` (the dot means everything)
`git commit -m "message to explain what has been done"`
`git push origin main` (to apply the changes)

Say some days pass by and our local project is not yet updated, to recover 
the remote changes it should be sufficient to call on the terminal the 
command 
`git pull origin main`


Project Organization
------------
``` bash
  | - local_python_environment        # the envinroment that will be installed
  | - references                      # references
  | - source                          # windows specific
  |  | - Scripts                      
  | - saved_models                    # models saved, .json format
  | - data                            # data repository
  |  | - raw                  
  |  |  | - publication_data          # data from Gal 22, eventually not used but there is a script to download it
  |  |  |  | - figures
  |  |  |  | - data
  |  | - engineered
  | - notebooks                       # notebooks folder for potential reports, empty
  | - reports                         # graphical and numerical results in tables and .png figures 
  |  | - tables
  |  | - figures
  |  |  | - nest_dynamics             # sub folder with the .png transition to make the .gif of dynamics
  | - src
  |  | - __pycache__
  |  | - modeling                     # scripts for modeling
  |  |  | - __pycache__
  |  | - visuals                      # scripts for plots
  |  |  | - __pycache__     
  |  | - engineering                  # scripts for data manipulation
  |  |  | - __pycache__
  | - main.py                         # main file to be run, accesses all built in functions automatically, will output basic info on
                                      # the terminal
  ```
