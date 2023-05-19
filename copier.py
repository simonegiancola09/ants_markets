###### here we build the agents and their rules #######
import random
import numpy as np
import networkx as nx
import sys
import pandas as pd
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
from mesa.space import NetworkGrid



# here we code a class that 

def run_model(model, epochs, save_info=True):
    for _ in range(epochs-1):
        model.step()

    if save_info:
        model_df = model.datacollector.get_model_vars_dataframe()
        model_df[['pos_x', 'pos_y']] = model_df['nest_location'].apply(pd.Series)
        agents_df = model.datacollector.get_agent_vars_dataframe()
        return model_df, agents_df



def determine_price(model):

    dif = model.demand - model.supply
    # volume = model.demand + model.supply
    # 
    pct = dif / model.num_stocks
    return pct

def activate(beta, h):
    return np.where((0.5+0.5*np.tanh(beta*h)) > np.random.rand(),1,-1)


def compute_magnetization(model):
    agents_state = [agent.state for agent in model.schedule.agents]
    num_agents = len(agents_state)
    return np.sum(agents_state) / num_agents

def get_nest_location(model):
    '''
    In the main paper, the nest location is estimated as the median of the investors positions
    '''
    cash_array = []
    stock_array = []
    # retrieve all positions
    for investor in model.grid.get_all_cell_contents():
        cash_array.append(investor.cash)
        stock_array.append(investor.stocks_owned)
    # return median
    return np.median(cash_array), np.median(stock_array)

class Ant_Financial_Agent(Agent):
    '''
    Agent describing possible moves of a shareholder
    '''

    def __init__(self, unique_id, model, cash, stock, risk_propensity, sensitivity=1):
        super().__init__(unique_id, model)
        # individual parameters
        self.cash = cash
        self.stocks_owned = stock
        self.utility = 0
        self.wealth = self.calculate_wealth()
        self.risk_propensity = risk_propensity
        self.sensitivity = sensitivity
        self.state = np.random.choice([1,-1]) # -1 if non actively investing, 1 if investing
        self.last_price = self.model.stock_price
        # the position is cash and stock
        self.x = cash / self.wealth
        self.y = stock * self.model.stock_price / self.wealth

    def move(self):
        self.update_position()

    def update_position(self):
        wealth = self.calculate_wealth()
        self.x = self.cash / wealth
        self.y = self.stocks_owned * self.model.stock_price / wealth


    def interact(self):
        pass

    def check_status(self):
        self.cash = np.maximum(5, self.cash)
        self.stocks_owned = np.maximum(0, self.stocks_owned)

    def step(self):
        '''
        Steps:
            1. checks cash is minimum 5 and stocks are nonnegative
            2. draws alpha randomly -- to be updated TODO
            3. calculates utility depending on alpha
            4. computes score based on utility: score in [-1, 1] where
                -1 represents selling everything and 1 buying as much as one can
            5. willingness to buy/sell evaluated in comparison with stocks alrady owned
            6. based on willingness, buy/sell a random amount drawn from a poisson distribution
            7. move agents on quadrant based on new cash/stocks'''

        self.check_status()

        risk = self.risk_propensity

        uncertainty_level = 0

        alpha = np.random.rand()

        u = self.calculate_utility(alpha)
        self.utility = u

        score = 0.5+0.5*np.tanh(risk*u)

        willingness = score - self.y

        if willingness > 0:
            self.state = 1
            can_buy = np.minimum(self.cash // self.model.stock_price, 1000)
            if (can_buy * willingness) < 0:
                print('Attention!!!')
            # TODO @Dario added np.abs willingness, does this make sense?
            tentative = np.random.poisson((can_buy * np.abs(willingness)).astype(np.float64))
            quantity = np.minimum(np.minimum(tentative, can_buy), self.model.num_available)
            self.stocks_owned += quantity
            self.cash -= quantity * self.model.stock_price
            self.model.demand += quantity

        else:
            self.state = -1
            can_sell = self.stocks_owned
            # TODO @Dario, here np.abs to willingness was already present
            tentative = np.random.poisson((can_sell * np.abs(willingness)).astype(np.float64))
            quantity = np.minimum(can_sell, tentative)
            self.stocks_owned -= quantity
            if self.stocks_owned < 0:
                print(quantity, can_sell)
            self.cash += quantity * self.model.stock_price
            self.model.supply += quantity

        self.move()
        

    def calculate_wealth(self):
        '''
        Calculate total wealth of agent
        '''
        return self.cash + self.stocks_owned * self.model.stock_price

    def get_neighbors(self):
        '''
        get neighbors of agent
        '''

        return [self.model.schedule.agents[neighbor] for neighbor in self.model.grid.get_neighbors(self.pos, include_center=False)]

    def calculate_utility(self, alpha):
        '''
        Should compute some sort of local utility:
            - mixture of Temperature (Rt) and state behavior of neighbors depending on alpha
            - state represents buying/selling behavior 
        '''
        # TODO utility should depend on price somehow
        # will fix how it calculates it to account for no neighbors case
        neighbor_states = np.array([nh.state for nh in self.get_neighbors()])
        num_nh = len(neighbor_states)
        # TODO need to account for weights
        # notice that the G.nodes() function that gives numbers 0, ..., N - 1
        # is the assignment to the ID of the Mesa Model Class
        # thus, to access weights we can go back to the original interaction_graph instance
        # stored in self.G and retrieve each weight 
        # namely, access the double dictionary of 
        edges_weights = [self.model.G[self.unique_id][nh.unique_id]['weight'] for nh in self.get_neighbors()]
        # it could be that if no neighbors are present, the value of the two variables above is 
        # invalid (empty lists etc), we use a fill value to account for this case and just ignore
        # the second term of the expression
        temperature_contrib = alpha * (-self.model.T)
        # NOTICE: works but given a RuntimeWarning, we could 
        # fix this in later versions
        neigh_contrib = np.where((num_nh == 0),
                                  0, 
                                  (1 - alpha) * np.sum(edges_weights * neighbor_states) / num_nh)
        # reads do the multiplication but when the num of neighbors is zero
        # just put zero

        u = temperature_contrib + neigh_contrib
        return u


    def set_aside(self):
        '''
        In principle, we want to restrict those investors that tend too much to invest / disinvest
        for this reason we need to understand where to respawn investors when they get to their
        limiting behavior at one of the corners, or even out of it. 
        The idea could be that we just imagine that they set aside money or if they finish it just
        get an injection from the outside. 
        '''
        return None

class Nest_Model(Model):
    '''
    Model describing investors interacting on a nest structure
    '''
    def __init__(self, beta, initial_stock_price, external_var, interaction_graph, num_stocks):
        # instantiate model at the beginning
        self.num_agents = interaction_graph.number_of_nodes()
        self.stock_price = initial_stock_price
        self.num_stocks = num_stocks
        self.num_available = num_stocks
        self.demand = 0
        self.supply = 0
        self.pct_change = 0
        self.beta = beta
        self.external_var = external_var
        self.T = external_var[0]
        # self.max_trade_size = max_trade_size
        self.schedule = RandomActivation(self)
        # graph is an input, can be chosen
        self.G = interaction_graph
        self.grid = NetworkGrid(interaction_graph)
        self.t = 0

        
        # create agents
        for i, node in enumerate(self.G.nodes()):
            cash = random.uniform(500, 1000) 
            stock = random.uniform(0, 10)
            risk_propensity = random.uniform(0, 1)
            self.num_available -= stock
            sensitivity = 1
            investor = Ant_Financial_Agent(i, self, cash, stock, risk_propensity, sensitivity)
            self.grid.place_agent(investor, node)
            self.schedule.add(investor)

        # collect relevant data
        self.datacollector = DataCollector(
            agent_reporters = {'cash': 'cash',
                               'wealth': 'wealth',
                               'x' : 'x',
                                'y' : 'y',
                               'utility': 'utility',
                               'state' : 'state'
                               },
            model_reporters = {'state': compute_magnetization,
                               'T':'T',
                               'price':'stock_price',
                               'nest_location': get_nest_location,
                               'demand':'demand',
                               'supply':'supply',
                               'pct_change':'pct_change',
                               'stocks_available': 'num_available'
                               }
        )


    def step(self):
        self.datacollector.collect(self)
        self.demand = 0
        self.supply = 0
        self.schedule.step()
        self.model_update()


    def model_update(self):
        self.t += 1
        old_price = self.stock_price
        pct = determine_price(self)
        new_price = old_price + old_price * pct
        self.stock_price = new_price
        self.num_available = self.num_available + self.supply - self.demand


        self.T = self.external_var[self.t]
        # pct_time = self.t / self.max_steps
        self.pct_change = (new_price - old_price) / old_price
        



  

''''
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.ModularVisualization import ModularServer

# Create an instance of the InvestorModel with 50 investors, a maximum trade size of 10, an external variable of 0.1, and an initial stock price of 100
model = InvestorModel(num_investors=50, max_trade_size=10, external_var=0.1, initial_stock_price=100)

# Create a chart to visualize the stock price over time
stock_price_chart = ChartModule([{"Label": "Stock Price", "Color": "Black"}], data_collector_name="datacollector")

# Create a network visualization of the investor network
network_viz = NetworkModule(model.grid, model.schedule)

# Create a server to run the simulation
server = ModularServer(InvestorModel, [network_viz, stock_price_chart], "Investor Model", {"num_investors": 50, "max_trade_size": 10, "external_var": 0.1, "initial_stock_price": 100})

# Start the server
server.port = 8521 # Choose a port to run the server on
server.launch()

'''

'''
This code creates an instance of the InvestorModel class with 50 investors, a maximum trade size of 10, an external variable of 0.1, and an initial stock price of 100.
 It also creates a chart to visualize the stock price over time and a network visualization of 
 the investor network. Finally, it creates a server to run the simulation and starts the server 
 on a chosen port.

You can access the visualization by opening a web browser and navigating to http://localhost:8521/. 
You should see a network visualization of the investor network and a chart of the stock 
price over time. You can then click the "Run" button to start the simulation and watch it 
evolve over time.
'''