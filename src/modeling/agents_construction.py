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
    # run an ABM model

    for _ in range(epochs+1):
        model.step()

    if save_info:
        model_df = model.datacollector.get_model_vars_dataframe()
        model_df[['pos_x', 'pos_y']] = model_df['nest_location'].apply(pd.Series)
        agents_df = model.datacollector.get_agent_vars_dataframe()
        return model_df, agents_df


def find_crossing_points(line1, line2):
    # Find the line with the smaller size
    if line1.shape[0] <= line2.shape[0]:
        smaller_line = line1
        larger_line = line2
    else:
        smaller_line = line2
        larger_line = line1
    
    # Calculate the size difference between the two lines
    size_diff = larger_line.shape[0] - smaller_line.shape[0]
    
    # Enlarge the smaller line by appending the last value
    enlarged_smaller_line = np.concatenate([smaller_line, [smaller_line[-1]] * size_diff])
    
    # Calculate the differences between the corresponding y-coordinates
    y_diff = enlarged_smaller_line[:, 1] - larger_line[:, 1]
    
    # Calculate the differences between the corresponding x-coordinates
    x_diff = enlarged_smaller_line[:, 0] - larger_line[:, 0]
    
    # Find the indices where the differences change sign (i.e., where the lines cross)
    crossings = np.where(np.diff(np.sign(y_diff)) != 0)[0]
    
    # Interpolate to find the x-coordinate values at the crossing points
    for crossing in crossings:
        x1, y1 = enlarged_smaller_line[crossing]
        x2, y2 = enlarged_smaller_line[crossing+1]
        x_crossing = np.interp(0, [y1, y2], [x1, x2])
        y_crossing = np.interp(x_crossing, [x1, x2], [y1, y2])
        crossing_points = (x_crossing, y_crossing)
    
    return crossing_points



def determine_price(model, elasticity):

    excess_demand = model.demand - model.supply
    # excess_stocks = model.num_stocks - model.num_available
    # volume = model.demand + model.supply
    pct = excess_demand / model.num_stocks
    pct = elasticity * pct * model.price
    new_price = np.maximum(model.price + pct, 0.1)
    return new_price

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

class Ant_financial_Agent(Agent):
    def __init__(self, unique_id, model, trader_type, tau, cash, stocks):
        super().__init__(unique_id, model)
        
        # Check if type is a number among [0, 1, 2, 3]
        if trader_type not in [0, 1, 2, 3]:
            raise ValueError("Invalid type value. Must be one in {0, 1, 2, 3}.")
        self.type = trader_type

        self.active = 0
        self.cash = cash
        self.stocks = stocks
        self.tau = tau
        self.p_f = None


        
    def step1(self):
        self.activate(self)
        if self.active:
            decision, price = self.buy_sell(self)
            qty = self.determine_quantity(self, decision, price)
            if decision:
                self.model.demand.append([qty, price])
            else:
                self.model.supply.append([qty, price])

    def move(self):
        
        self.update_position()

    def update_position(self):
        wealth = self.calculate_wealth()
        self.x = self.cash / wealth
        self.y = self.stocks_owned * self.model.price / wealth


    def interact(self):
        pass

    def check_status(self):
        self.cash = np.maximum(5, self.cash)
        self.stocks_owned = np.maximum(0, self.stocks_owned)
        self.active = np.random.choice([1,0])

    def update_alpha(self):
        pass

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

        alpha = self.model.rt_perceived

        self.utility += self.calculate_utility(alpha)
        u = self.utility

        score = 0.5 + 0.5*np.tanh(risk*u)

        willingness = score - self.y
        self.willingness = willingness

        if willingness > 0:
            self.state = 1
            can_buy = np.minimum(self.cash // self.model.price, 200)
            if (can_buy * willingness) < 0:
                print('Attention!!!', self.cash, self.model.price, can_buy)
            # TODO @Dario added np.abs willingness, does this make sense?
            tentative = np.random.poisson((can_buy * np.abs(willingness)))
            quantity = np.minimum(np.minimum(tentative, can_buy), self.model.num_available)
            self.stocks_owned += quantity
            self.cash -= quantity * self.model.price
            self.model.demand += quantity
            self.model.num_available -= quantity

        else:
            self.state = -1
            can_sell = self.stocks_owned

            try:
                tentative = np.random.poisson((can_sell * np.abs(willingness)))
            
            except:
                tentative = 0
                print('wrong selling behavior')
                print(can_sell)
            quantity = np.minimum(can_sell, tentative)
            self.stocks_owned -= quantity
            if self.stocks_owned < 0:
                print(quantity, can_sell)
            self.cash += quantity * self.model.price
            self.model.supply += quantity
            self.model.num_available += quantity

        self.move()
        

    def calculate_wealth(self):
        '''
        Calculate total wealth of agent
        '''
        return self.cash + self.stocks_owned * self.model.price

    def get_neighbors(self, include_center=True):
        '''
        get neighbors of agent
        '''

        return [self.model.schedule.agents[neighbor] for neighbor in self.model.grid.get_neighbors(self.pos, include_center=include_center)]

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
        # notice that the G.nodes() function that gives numbers 0, ..., N - 1
        # is the assignment to the ID of the Mesa Model Class
        # thus, to access weights we can go back to the original interaction_graph instance
        # stored in self.G and retrieve each weight 
        # namely, access the double dictionary of 
        
        # edges_weights = [self.model.G[self.unique_id][nh.unique_id]['weight'] for nh in self.get_neighbors()]
        
        edges_weights = 1
        # it could be that if no neighbors are present, the value of the two variables above is 
        # invalid (empty lists etc), we use a fill value to account for this case and just ignore
        # the second term of the expression
        temperature_contrib = -alpha*(self.model.T)
        # NOTICE: works but given a RuntimeWarning, we could 
        # fix this in later versions
        neigh_contrib = np.where((num_nh == 0),
                                  0, 
                                  self.model.beta * np.sum(edges_weights * neighbor_states) / num_nh)
        # reads do the multiplication but when the num of neighbors is zero
        # just put zero

        price_contrib = -2 * self.model.price_change

        u = temperature_contrib + neigh_contrib #+ price_contrib
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
    def __init__(self, k, price_history, shock, p, G):
        super().__init__()
        self.k = k
        self.price_history = price_history
        self.price = price
        self.shock = shock
        self.p = p
        self.G = G
        self.demand = []
        self.supply = []
        self.schedule = RandomActivation(self)
        self.grid = NetworkGrid(G)
        self.t = 0

        # create agents
        for i, node in enumerate(self.G.nodes()):

            #instantiate the agents
            cash = random.uniform(500, 1000) 
            stock = random.uniform(0, 10)
            tau = random.uniform(10, 100)
            trader_type = np.random.choice([0,1,2,3])
            agent = Ant_Financial_Agent(i, self, trader_type, tau, cash, stocks)
            self.grid.place_agent(agent, node)
            self.schedule.add(agent)

        # collect relevant data
        self.datacollector = DataCollector(
            agent_reporters = {
                            'cash': 'cash',
                            'active': 'active'
                               },
            model_reporters = {
                            'state': compute_magnetization,
                            'T':'T',
                            'price':'price',
                            'nest_location': get_nest_location
                               }
                            )


    def step(self):
        self.datacollector.collect(self)
        self.demand = 0
        self.supply = 0
        self.schedule.step()
        self.model_update()

    def determine_price(self):
        supply = np.array(self.supply)
        demand = np.array(self.demand)

        idx_s = np.argsort(supply[:,0])
        supply = supply[idx_s]
        supply[:,1] = supply[:,1].cumsum()
        
        idx_d = np.argsort(demand[:,0])
        demand = demand[idx_d[::-1]]
        demand[:,1] = demand[:,1].cumsum()
        demand = demand[idx_d]

        new_price, quantity = find_crossing_points(supply, demand)


    def model_update(self):
        self.t += 1
        old_price = self.price
        new_price = determine_price(self, elasticity=0.2)
        self.price = new_price
        # new_price = old_price + old_price * pct
        if new_price < 0:
            print('negative_price')
        self.price_change = (new_price - old_price) / old_price

        self.num_available = self.num_available + self.supply - self.demand

        if self.t < len(self.external_var):
            self.T = self.external_var[self.t]
            self.rt_perceived = self.rt_change[self.t]

    def reset(self):
        self.t = 0
        self.price_change = 0
        self.rt_perceived = self.external_var[0]
        



  

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
