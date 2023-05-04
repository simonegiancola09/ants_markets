###### here we build the agents and their rules #######
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner

import random
import numpy as np
import sys


# here we code a class that 

class Ant_Financial_Agent(Agent):
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

class Nest_Model(Model):
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





    ################## TENTATIVE ####################################
    import random
import numpy as np
import networkx as nx
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

class Investor(Agent):
    def __init__(self, unique_id, model, cash, stock, likelihood):
        super().__init__(unique_id, model)
        self.cash = cash
        self.stock = stock
        self.likelihood = likelihood
        self.state = -1  # -1 for silent, 1 for active
        self.last_price = None

    def step(self):
        # Observe external variable
        external_var = self.model.external_var

        # Observe state of neighbors
        neighbor_states = [self.model.schedule.agents[neighbor].state for neighbor in self.model.grid.get_neighbors(self.pos)]

        # Observe current stock price
        current_price = self.model.stock_price

        # Calculate expected return on investment
        expected_return = self.likelihood * (current_price - self.last_price) / self.last_price

        # Decide whether to buy, sell, or remain silent
        if self.cash <= 0:
            self.state = -1
        elif self.stock <= 0:
            self.state = 1
        else:
            if max(neighbor_states) == 1 and random.random() < 0.5:
                self.state = -1
            elif min(neighbor_states) == -1 and random.random() < 0.5:
                self.state = 1
            elif expected_return > 0 and random.random() < expected_return:
                self.state = 1
            elif expected_return < 0 and random.random() < abs(expected_return):
                self.state = -1
            else:
                self.state = -1

        # Buy, sell, or remain silent
        if self.state == 1:
            amount = min(self.cash, self.model.max_trade_size)
            self.stock += amount / current_price
            self.cash -= amount
        elif self.state == -1:
            pass
        else:
            amount = min(self.stock * current_price, self.model.max_trade_size)
            self.stock -= amount / current_price
            self.cash += amount

    def calculate_wealth(self):
        return self.cash + self.stock * self.model.stock_price

class InvestorModel(Model):
    def __init__(self, num_investors, max_trade_size, external_var, initial_stock_price):
        self.num_investors = num_investors
        self.max_trade_size = max_trade_size
        self.external_var = external_var
        self.stock_price = initial_stock_price
        self.schedule = RandomActivation(self)
        self.grid = NetworkGrid(nx.erdos_renyi_graph(num_investors, 0.5), torus=False)

        # Create investors with random initial capital
        for i in range(self.num_investors):
            cash = random.uniform(0, 100)
            stock = random.uniform(0, 100)
            likelihood = random.uniform(0, 1)
            investor = Investor(i, self, cash, stock, likelihood)
            self.schedule.add(investor)
            self.grid.place_agent(investor, i)

        # Collect data on investor wealth over time
        self.datacollector = DataCollector(
            model_reporters={"Stock Price": lambda m: m.stock_price},
            agent_reporters={"Wealth": lambda a: a.calculate_wealth()}
        )

    def step(self):
        self.stock_price += random.uniform(-self.external_var, self.external_var)
        self.schedule.step()
        self.datacollector.collect(self)


