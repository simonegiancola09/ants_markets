###### here we build the agents and their rules #######
# %%
import random
import numpy as np
import networkx as nx
import sys
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
from mesa.space import NetworkGrid


np.random.randn()

# here we code a class that 

def activate(beta, h):

    return np.where((0.5+0.5*np.tanh(beta*h)) > np.random.rand(),1,-1)

def compute_magnetization(model):
    agents_state = [agent.state for agent in model.schedule.agents]
    return np.sum(agents_state)

class Ant_Financial_Agent(Agent):
    '''
    Agent describing possible moves of a shareholder
    '''
    # TODO should see financial data, R number and make decisions on it
    # then pass it on to the nest as a collective behavior
    # TODO should have some investing preferences defined ex ante
    # TODO these investing preferences could also be random as a start
    def __init__(self, unique_id, model, cash, stock, risk_propensity, sensitivity=1):
        super().__init__(unique_id, model)
        # individual parameters
        self.cash = cash
        self.stocks_owned = stock
        self.utility = 0
        self.total_owned = self.calculate_wealth()
        self.risk_propensity = risk_propensity
        self.sensitivity = sensitivity
        self.state = -1 # -1 if non actively investing, 1 if investing
        self.last_price = self.model.stock_price
        # the position is cash and stock
        self.x = cash / self.total_owned
        self.y = stock * self.model.stock_price / self.total_owned

    def move(self):
        pct_change = self.model.pct_change
        direction = activate(self.risk_propensity, pct_change)
        
        if self.state == 1:
            if direction < 0:
                if self.stocks_owned > 0:
                    try:
                        quantity = np.random.randint(int(0.6*self.stocks_owned), self.stocks_owned) 
                    except:
                        print('here')
                        quantity = 0
                    self.stocks_owned -= quantity
                    self.cash += quantity * self.model.stock_price

            else:
                max_stocks = np.minimum(self.cash // self.model.stock_price, self.model.num_stocks)
                if max_stocks > 0:
                    quantity = np.random.randint(max_stocks//2, max_stocks)
                    self.stocks_owned += quantity
                    self.cash -= quantity * self.model.stock_price

        else:
            if direction < 0:
                if self.stocks_owned > 0:
                    quantity = np.random.randint(0, np.maximum(self.stocks_owned//5, 1)) 
                    self.stocks_owned -= quantity
                    self.cash += quantity * self.model.stock_price

            else:
                max_stocks = np.minimum(self.cash // self.model.stock_price, self.model.num_stocks)
                if max_stocks > 0:
                    quantity = np.random.randint(0, np.maximum(max_stocks//5, 1))
                    self.stocks_owned += quantity
                    self.cash -= quantity * self.model.stock_price
        self.update_position()

    def update_position(self):
        wealth = self.calculate_wealth()
        self.x = self.cash / wealth
        self.y = self.stocks_owned * self.model.stock_price / wealth


    def interact(self):
        pass

    def step(self):

        u = self.calculate_utility()
        self.utility = u
        # print(u)
        self.state = activate(self.model.beta, u)
        self.move()
        

    def calculate_wealth(self):
        return self.cash + self.stocks_owned * self.model.stock_price

    def get_neighbors(self):
        '''
        get neighbors of agent
        '''

        return [self.model.schedule.agents[neighbor] for neighbor in self.model.grid.get_neighbors(self.pos, include_center=False)]

    def calculate_utility(self):
        '''
        Should compute some sort of local utility considering:
            - cash (agent-dependent)
            - stock (agent-dependent)
            - risk propensity (agent-dependent)
            - current stock price (market-dependent)
            - the external variable (i.e. Rt) (repulsive)
            - what neighbors are doing (attractive)
        '''

        neighbor_states = np.array([nh.state for nh in self.get_neighbors()])
        edges_weights = 1
        u = (self.model.T - self.sensitivity) * 10 + np.sum(edges_weights * neighbor_states)

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
    def __init__(self, beta, stock_price, external_var, interaction_graph, num_stocks):
        # instantiate model at the beginning
        assert stock_price.shape == external_var.shape
        self.num_agents = interaction_graph.number_of_nodes()
        self.history_stock = stock_price
        self.stock_price = stock_price[0]
        self.avg_stock_price = stock_price[0]
        self.num_stocks = num_stocks
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
        self.max_steps = self.stock_price.shape

        
        # create agents
        for i, node in enumerate(self.G.nodes()):
            cash = random.uniform(50, 100) 
            stock = random.uniform(0, 10)
            risk_propensity = random.uniform(0, 5)
            sensitivity = 1
            investor = Ant_Financial_Agent(i, self, cash, stock, risk_propensity, sensitivity)
            self.grid.place_agent(investor, node)
            self.schedule.add(investor)

        # collect relevant data
        self.datacollector = DataCollector(
            agent_reporters = {'cash': 'x',
                               'stocks': 'y',
                               'utility': 'utility'},
            model_reporters = {'state': compute_magnetization}
        )


    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.model_update()

    def get_nest_location(self):
        '''
        In the main paper, the nest location is estimated as the median of the investors positions
        '''
        cash_array = []
        stock_array = []
        # retrieve all positions
        for investor in self.grid.get_all_cell_contents:
            cash_array.append(investor.cash)
            stock_array.append(investor.stocks_owned)
        # return median
        return np.median(cash_array), np.median(stock_array)

    def model_update(self):
        history = self.history_stock
        self.t += 1
        self.stock_price = history[t]
        self.T = self.external_var[t]
        # pct_time = self.t / self.max_steps
        self.avg_stock_price = np.mean(history[:t+1])
        self.pct_change = (history[t] - history[t-1]) / history[t-1]
        



    ################## TENTATIVE ####################################
# below are some code snippets that could be useful to determine our agents and our model
# they should be merged with the colony class in the other file

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