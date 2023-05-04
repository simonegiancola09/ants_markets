###### here we build the agents and their rules #######
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner

import random
import numpy as np
import sys


# here we code a class that 

class ANT_FINANCIAL_AGENT(Agent):
    '''
    Agent describing possible moves of a shareholder
    '''
    # TODO should see financial data, R number and make decisions on it
    # then pass it on to the nest as a collective behavior
    # TODO should have some investing preferences defined ex ante
    # TODO these investing preferences could also be random as a start
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        pass 
    def move(self):
        pass
    def interact(self):
        pass
    def try_to_come_back(self):
        pass
    def step(self):
        pass

class NEST_MODEL(model):
    '''
    Model describing investors interacting on a nest structure
    '''
    def __init__(self, N):
        self.num_agents = N
        
        # create agents
        for i in range(self.num_agents):
            a = ANT_FINANCIAL_AGENT(i, self)
            self.schedule.add(a)

            # need to embed graph probably, assign randomly
        # collect relevant data
        self.datacollector = DataCollector(model_reporters = {})
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        #do somehting
