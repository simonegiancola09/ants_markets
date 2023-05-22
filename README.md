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
```
  | - local_python_environment
  | - references
  | - source
  |  | - Scripts
  | - saved_models
  | - data
  |  | - raw
  |  |  | - publication_data
  |  |  |  | - figures
  |  |  |  | - data
  |  | - engineered
  | - notebooks
  | - reports
  |  | - tables
  |  | - figures
  |  |  | - nest_dynamics
  | - src
  |  | - __pycache__
  |  | - modeling
  |  |  | - __pycache__
  |  | - visuals
  |  |  | - __pycache__
  |  | - engineering
  |  |  | - __pycache__
  ```
