###### here we build the agents and their rules #######
import random
import numpy as np
import networkx as nx
import sys
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
from mesa.space import NetworkGrid



# here we code a class that 

def test_price_model(p0 = 80, s = 0.1, N = 400, qty_low=0, qty_high=200,
                    factor=1):
    p = np.linspace(0.1, 0.9, 500)

    new_prices = []
    pct_buyers = []
    for q in p:
        supply = []
        demand = []
        buyers = 0
        for _ in range(N):
            
            if q > np.random.rand():
                adj = np.random.normal(1.5, s)
                p_i = p0 * adj
                qty = np.random.randint(qty_low*factor, qty_high*factor)
                demand.append([p_i, qty])
                buyers += 1
            else:
                adj = np.random.normal(1.5, s)
                p_i = p0 / adj
                qty = np.random.randint(qty_low, qty_high)
                supply.append([p_i, qty])
            
        supply = np.array(supply)
        demand = np.array(demand)
        idx_s = np.argsort(supply[:,0])
        supply = supply[idx_s]
        supply[:,1] = supply[:,1].cumsum()
        
        idx_d = np.argsort(demand[:,0])
        demand = demand[idx_d[::-1]]
        demand[:,1] = demand[:,1].cumsum()
        idx_d = np.argsort(demand[:,0])
        demand = demand[idx_d]
        # new_price, quantity = find_crossing_points(supply, demand, p0)
        
        new_price = determine_stock_price(supply, demand)
        new_prices.append(new_price)
        pct_buyers.append(buyers / N)
    plt.plot(p, new_prices)
    plt.hlines(p0, 0.1, 0.9)
    plt.show()


# def determine_stock_price(supply, demand, randomness=0.1):
#     # Get the maximum price from supply and minimum price from demand
#     max_supply_price = np.max(supply[:, 0])
#     min_demand_price = np.min(demand[:, 0])

#     # Check if there is an intersection between supply and demand
#     if max_supply_price >= min_demand_price:
#         # Find the index of the supply price that is greater than or equal to the minimum demand price
#         supply_index = np.argmax(supply[:, 0] >= min_demand_price)

#         # Find the index of the demand price that is less than or equal to the maximum supply price
#         demand_index = np.argmin(demand[:, 0] <= max_supply_price)

#         # Get the quantity at the intersection point
#         supply_quantity = supply[supply_index, 1]
#         demand_quantity = demand[demand_index, 1]

#         # Calculate the new price as the average of the intersecting supply and demand prices with added randomness
#         new_price = (supply[supply_index, 0] + demand[demand_index, 0]) / 2 * (1 + randomness * np.random.randn())

#         # Update the supply and demand quantities based on the intersection
#         supply[supply_index, 1] -= min(supply_quantity, demand_quantity)
#         demand[demand_index, 1] -= min(supply_quantity, demand_quantity)

#     else:
#         # No intersection, set the new price as the average of the maximum supply price and minimum demand price
#         new_price = (max_supply_price + min_demand_price) / 2

#     return new_price

def determine_stock_price(supply, demand, randomness=0.1):
    # Calculate the total supply and demand quantities
    total_supply = np.sum(supply[:, 1])
    total_demand = np.sum(demand[:, 1])

    # Calculate the weighted average of supply and demand prices based on quantities
    weighted_supply_price = np.sum(supply[:, 0] * supply[:, 1]) / total_supply
    weighted_demand_price = np.sum(demand[:, 0] * demand[:, 1]) / total_demand

    # Calculate the new price as the average of the weighted supply and demand prices with added randomness
    new_price = (weighted_supply_price + weighted_demand_price) / 2 #* (1 + randomness * np.random.randn())

    return new_price

def find_crossing_points(line1, line2, old_price, debug=False):
    '''
    Finds crossing points in the demand and supply. 
    '''
    if debug:
        plt.plot(line1[:, 0], line1[:, 1])
        # half = (line1[0, :] + line2[-1, :]) / 2
        # plt.plot(np.r_[line1[0, 0], line2[-1,0]], np.r_[line1[0, 1], line2[-1, 1]], ls='--', c='black')
        # plt.vlines(half[0], 0, np.max(np.r_[line1, line2]))
        # plt.scatter(half[0], half[1], c='red')
        plt.plot(line2[:, 0], line2[:, 1])
        plt.show()
    # Find the line with the smaller size
    if line1.shape[0] <= line2.shape[0]:
        smaller_line = line1
        larger_line = line2
    else:
        smaller_line = line2
        larger_line = line1
    
    # Calculate the size difference between the two lines
    size_diff = larger_line.shape[0] - smaller_line.shape[0]
    # print(smaller_line, smaller_line[-1])
    # Enlarge the smaller line by appending the last value

    if size_diff > 0:
        enlarged_smaller_line = np.concatenate([smaller_line, [smaller_line[-1]] * size_diff])
    else:
        enlarged_smaller_line = smaller_line
    # Calculate the differences between the corresponding y-coordinates
    y_diff = enlarged_smaller_line[:, 1] - larger_line[:, 1]
    
    # Calculate the differences between the corresponding x-coordinates
    x_diff = enlarged_smaller_line[:, 0] - larger_line[:, 0]
    
    # Find the indices where the differences change sign (i.e., where the lines cross)
    crossings = np.where(np.diff(np.sign(y_diff)) != 0)[0]
    
    # Interpolate to find the x-coordinate values at the crossing points
    crossing_points = old_price, 0
    for crossing in crossings:
        x1, y1 = enlarged_smaller_line[crossing]
        x2, y2 = enlarged_smaller_line[crossing+1]
        x_crossing = np.interp(0, [y1, y2], [x1, x2])
        y_crossing = np.interp(x_crossing, [x1, x2], [y1, y2])
        crossing_points = (x_crossing, y_crossing)
    
    return crossing_points

def run_model(model, epochs, save_info=True):
    '''
    Runs the ABM model for a givne number of epochs.
    '''
    # run an ABM model

    for _ in range(epochs):
        model.step()
        if model.t / epochs == 0.5:
            print(f'50% completed')

    if save_info:
        model_df = model.datacollector.get_model_vars_dataframe()
        model_df[['pos_x', 'pos_y']] = model_df['nest_location'].apply(pd.Series)
        agents_df = model.datacollector.get_agent_vars_dataframe()
        return model_df, agents_df




def get_nest_location(model):
    '''
    In the main paper, the nest location is estimated as the median of the investors positions
    '''
    cash_array = []
    stock_array = []
    # retrieve all positions
    for investor in model.grid.get_all_cell_contents():
        cash_array.append(investor.cash)
        stock_array.append(investor.stocks)
    # return median
    return np.median(cash_array), np.median(stock_array)

class Ant_Financial_Agent(Agent):
    '''
    Our Agent Class... TODO description
    '''
    def __init__(self, 
                 unique_id, model,              # super class variables
                 cash, stocks,                  # general variables
                 trader_type, tau, p_f          # DemandSupply-Like variables
                 ):
        super().__init__(unique_id, model)
        
        # Check if type is a number among [0, 1, 2, 3]
        if trader_type not in [0, 1, 2, 3]:
            raise ValueError("Invalid type value. Must be one of {0, 1, 2, 3}.")
        # positions in space, absolute
        self.cash = cash
        self.stocks = stocks    # to properly be a position need to
                                # multiply it by number of stocks
        self.wealth = self.calculate_wealth()   # total wealth
        # positions in space, relative
        self.x = self.cash / self.wealth
        self.y = self.stocks * self.model.price / self.wealth
        # Ant-Like variables
        self.utility = 0
        self.p_buy = 0.5
        self.p_active = 0.1
        self.neighbors = None
        self.edges_weights = None
        self.nh_states = 0
        # DemandSupply-Like variables
        self.tau = tau
        self.std_price = np.std(self.model.price_history[-tau:])
        self.trader_type = trader_type # type of trader
        self.p_f = self.set_pf(p_f)
        self.p_i = 0
        self.quantity = 0
        self.buy_sell = 0 # buy if 1, sell if -1, do almost nothing if 0
        self.last_order = 0 # saves last order (-1, 0, 1) for each agent: this is what neighbors actually see
    
    ##### GENERAL FUNCTIONS ############################

    def config_neighbors(self):
        all_neighbors = self.get_neighbors()
        self.neighbors = all_neighbors
        self.edges_weights = np.array([self.model.G[self.unique_id][nh.unique_id]['weight'] for nh in all_neighbors])

    def update_position(self):
        '''
        After a potential move, update the position attributes of the agent.
        '''
        wealth = self.calculate_wealth()
        self.wealth = wealth
        self.x = self.cash / wealth
        self.y = self.stocks * self.model.price / wealth


    def calculate_wealth(self):
        '''
        Calculate total wealth of agent
        '''
        return self.cash + self.stocks * self.model.price

    ##### AGENT STEP FUNCTIONS ############################

    def step1(self):
        '''
        - compute probability of being active (placing an order)
        - if active, determine whether trader buys or sells and at what price and quantity
        - else, set buy_sell to 0
        '''
        # calcualate prob of placing an order
        p = self.prob_active()
        self.p_active = p
        if p >= np.random.rand():
            # if active, place order
            price, quantity = self.price_and_quantity()

            # record personal buy / sell price
            self.p_i = price     

            # record personal buy / sell qty
            self.quantity = quantity    
            self.model.stocks_left_to_buy -= quantity
            if self.buy_sell == 1:
                # if bought add to demand
                self.model.demand.append([price, quantity]) 
            elif self.buy_sell == -1:   
                # if have sold, add to supply
                self.model.supply.append([price, quantity])

        else:
            # else trader not active
            self.buy_sell = 0   

    def step2(self):
        price = self.model.price
        # agent has placed order
        if self.buy_sell != 0:

            p_i, quantity = self.p_i, self.quantity
            if self.buy_sell == 1:
                ## if decision was to buy 
                if price <= p_i:
                    quantity = np.minimum(quantity, self.model.stocks_left_to_buy)
                    ## buy only if price is at most what offered
                    self.stocks += quantity
                    self.cash -= price * quantity
                    self.model.stocks_left_to_buy -= quantity

                
            else:
                ## decision was to sell
                if price >= p_i:
                    ## sell only if price at least what offered
                    quantity = np.minimum(quantity, self.model.stocks_left_to_sell)
                    self.stocks -= quantity
                    self.cash += price * quantity
                    self.model.stocks_left_to_sell -= quantity

            # from the action of buying and selling the agent has moved
            # we update the position
            self.update_position()
        
        # update last_order with respective order
        self.last_order = self.buy_sell

    def step(self):
        '''
        Overridden by step1 and step2 methods
        '''
        pass
    ########################################################
    ##### ANT-LIKE FUNCTIONS ############################

    def prob_active(self):
        '''
        Calculate probability of being active as a function of what neighbors are doing.
        In particular, if neighbors are evenly distributed between buying/selling, the probability of being active is slow.
        However, as soon there are more buyers or sellers, the probability of being active increases.
        '''
        neighbor_states = np.array([nh.last_order for nh in self.neighbors])
        edges_weights = self.edges_weights

        num_nh = len(neighbor_states)
        # edges_weights = np.ones(num_nh)
        score = np.dot(edges_weights, neighbor_states) / num_nh
        self.nh_states = score
        p_active = 1 - stats.norm(0,.5).pdf(score)
        return p_active

    def prob_buy(self):
        '''
        Calculate probability of buying given external temperature.
        A positive value of T (worsening of a situation) will make traders more prone to sell,
        while a negative value of T (situation getting better) will make traders more prone to buy.
        '''
        t = self.model.T 
        alpha = self.model.alpha
        return stats.norm(loc=0, scale=1).sf(alpha * t)

    def check_status(self):
        '''
        Sanity check already built if needed. 
        '''
        self.cash = np.maximum(5, self.cash)
        self.stocks = np.maximum(0, self.stocks)

    def get_neighbors(self):
        '''
        Get neighbors of agent in the graph of the model
        TODO include center deprecated as it breaks the calculate_utility function
         basically a graph does not have self loops with weight
         nevertheless we have that the proper state is highly influential in the decision
         by the trading type etc.
        '''
        return [self.model.schedule.agents[neighbor]
                for neighbor in self.model.grid.get_neighbors(self.pos,)]
    
    ########################################################
    ##### DEMANDSUPPLY-LIKE FUNCTIONS ############################

    def set_pf(self, value=None):
        '''
        Set true price value fro fundametalist traders. If value is given, set pf to value 
        else set it as a mean of historical price.
        '''
        if value is None:
            return np.mean(self.model.price_history[-self.tau:])
        return value
        
    def price_and_quantity(self):
        '''
        Depending on the type of trader, perform a decision: whether to buy/sell and at which price and quantity.
        '''
        ## random trader
        if self.trader_type == 0: 

            # s_i = self.model.k * np.std(self.model.price_history[-self.tau:])
            # TODO change later
            s_i = self.std_price
            # s_i = 1* np.std(self.model.price_history[-self.tau:])
            p = self.prob_buy() 
            p = np.maximum(np.minimum(p, 0.9), 0.1)
            self.p_buy = p
            value = self.model.price * (1 + self.model.k * self.nh_states)
            if p >= np.random.rand():
                ## trader decides to buy
                self.buy_sell = 1
                adj = np.random.normal(value, s_i)
                p_i =  adj
                max_qty = self.cash // p_i
                max_qty = np.minimum(max_qty, self.model.stocks_left_to_buy) 
                if max_qty > 1:
                    quantity = np.random.randint(1, max_qty)

                else:
                    quantity = max_qty
            else:
                ## trader decides to sell
                self.buy_sell = -1
                adj = np.random.normal(value, s_i)
                p_i = adj
                max_qty = self.stocks 
                if max_qty > 1:
                    quantity = np.random.randint(1, max_qty)
                else:
                    quantity = max_qty

        ## fundamentalist trader
        elif self.trader_type == 1:
            last_price = self.model.price_history[-1]

            if last_price > self.p_f:
                ## trader decides to sell
                self.buy_sell = -1
                p_i = self.p_f
                max_qty = self.stocks
                if max_qty > 1:
                    quantity = np.random.randint(1, max_qty)
                else:
                    quantity = max_qty

            else:
                ## trader decides to buy
                self.buy_sell = 1
                p_i = self.p_f
                max_qty = self.cash // p_i
                max_qty = np.minimum(max_qty, self.model.stocks_left_to_buy) 
                if max_qty > 1:
                    quantity = np.random.randint(1, max_qty)
                else:
                    quantity = max_qty

        ## trend follower (up)
        elif self.trader_type == 2:
            actual_price = self.model.price
            tau = self.tau
            last_price = self.model.price_history[-tau]
            pct_change = (actual_price - last_price) / (tau * last_price)
            p_i = actual_price * (1 + pct_change)
            if pct_change > 0:
                ## trader decides to buy
                self.buy_sell = 1
                max_qty = self.cash // p_i
                max_qty = np.minimum(max_qty, self.model.stocks_left_to_buy) 
                if max_qty > 1:
                    quantity = np.random.randint(1, max_qty)
                else:
                    quantity = max_qty
            else:
                ## trader decides to sell
                self.buy_sell = -1
                max_qty = self.stocks
                if max_qty > 1:
                    quantity = np.random.randint(1, max_qty)
                else:
                    quantity = max_qty

        ## trend follower (down)
        else:
            actual_price = self.model.price
            tau = self.tau
            last_price = self.model.price_history[-tau]
            pct_change = (actual_price - last_price) / (tau * last_price)
            p_i = actual_price * (1 + pct_change)
            if pct_change <= 0:
                ## trader decides to buy
                self.buy_sell = 1
                max_qty = self.cash // p_i
                max_qty = np.minimum(max_qty, self.model.stocks_left_to_buy) 
                if max_qty > 1:
                    quantity = np.random.randint(1, max_qty)
                else:
                    quantity = max_qty
            else:
                ## trader decides to sell
                self.buy_sell = -1
                max_qty = self.stocks
                if max_qty > 1:
                    quantity = np.random.randint(1, max_qty)
                else:
                    quantity = max_qty

        bundle = p_i, quantity
        
        return bundle



    # def calculate_utility(self):
    #     '''
    #     Should compute some sort of local utility:
    #         - mixture of Temperature (Rt) and state behavior of neighbors depending on alpha
    #         - state represents buying/selling behavior 
    #     '''
    #     alpha = self.model.alpha
    #     neighbor_states = np.array([nh.state for nh in self.get_neighbors()])
    #     num_nh = len(neighbor_states)
    #     # notice that the G.nodes() function that gives numbers 0, ..., N - 1
    #     # is the assignment to the ID of the Mesa Model Class
    #     # thus, to access weights we can go back to the original interaction_graph instance
    #     # stored in self.G and retrieve each weight 
    #     # namely, access the double dictionary of 
    #     edges_weights = [self.model.G[self.unique_id][nh.unique_id]['weight'] for nh in self.get_neighbors()]
    #     # it could be that if no neighbors are present, the value of the two variables above is 
    #     # invalid (empty lists etc), we use a fill value to account for this case and just ignore
    #     # the second term of the expression
    #     temperature_contrib =  alpha * (-self.model.T)
    #     # NOTICE: works but given a RuntimeWarning, we could 
    #     # fix this in later versions
    #     neigh_contrib = np.where((num_nh == 0),
    #                               0, 
    #                               (1 - alpha) * np.sum(edges_weights * neighbor_states) / num_nh)
    #     # reads do the multiplication but when the num of neighbors is zero
    #     # just put zero
    #     u = temperature_contrib + neigh_contrib
    #     return u

    # def compute_score(self):
    #     '''
    #     Computes a sigmoid that is between 0 and 1 based on risk propensity and utility
    #     '''
    #     # alpha is common to all agents and can be calibrated
    #     # it is the importance in the utility of the Rt
    #     self.utility = self.calculate_utility()
    #     risk = self.risk_propensity
    #     # score is by construction between 0 and 1
    #     score = 0.5 + 0.5 * np.tanh(risk * self.utility)
    #     return score

    # def step1(self):

    #     # check whether trader is state
    #     self.score = self.compute_score() # result of a sigmoid
    #     #### OLD, CURRENT IMPLEMENTATION IS BELOW ######
    #     # abs_score = np.abs(score) # do specular version of score
    #     # need to decide how the self.state variable
    #     # is determined. 
    #     # this way if the sigmoidazation of the utility with risk propensity
    #     # is too divergent we have a highly selling behavior
    #     # if abs_score > np.random.rand():
    #         # if either highly stimulated or highly not stimulated
    #     #    self.state = 1
    #     #else:
    #     #    self.state = 0
        
    #     # or we just take the score to be a stretcher 
    #     # of excitation in positive and negative
        
        
    #     # for random type, random choice of activation
    #     # for trend follower type, deterministic choice of activation
    #     # for

       
    #     #if self.state:
    #         ## if state, place order
    #     # just decide price and quantity, where quantity is based
    #     # on a stretch of the maximum possible that is the score
    #     # current

    #     if self.score > self.model.cutoff:
    #         self.state = 1
    #         price, quantity = self.price_and_quantity(stretch = 1)
    #     else:
    #         self.state = -1
    #         price, quantity = self.price_and_quantity(stretch = self.score)
    #     # updates in the background the buy / sell strategy of our players
    #     # namely updates the self.buy_sell variable irrespective of self.state
    #     # depending on the type, the behaviors are slightly different
    #     self.p_i = price            # record personal buy / sell price
    #     self.quantity = quantity    # record personal buy / sell qty
    #     if self.buy_sell == 1:      # if have bought    
    #         self.model.demand.append([price, quantity]) # add to demand
    #         self.state = 1 
    #     elif self.buy_sell == -1:   # if have sold
    #         self.model.supply.append([price, quantity])
    #         self.state = 1 
    #     if self.state == -1:         # if it was weakly active then we see it as a zero
    #         self.buy_sell = 0 # in the end the score attribute if recorder will tell 
    #                          # us excited and non excited investors
    #     # if we get here, it means self.buy_sell is zero

    ########################################################
    
        
    
    

class Nest_Model(Model):
    def __init__(self, interaction_graph, external_var, date,   # general
                alpha,                                          # Ant-Like
                k, price_history, prob_type, p_f,               # DemandSupply-Like
                cash_low, cash_high, stocks_low, stocks_high    # DemandSupply-Like
                 ):
        # general
        super().__init__()
        self.G = interaction_graph
        self.grid = NetworkGrid(interaction_graph)
        self.schedule = RandomActivation(self)
        self.date = date                    # starting date for simulation
        self.N = len(interaction_graph.nodes())             # number of agents
        self.external_var = external_var    # external variable, in our case the Rt
        self.t = 0                          # number of times it was run
        self.T = external_var[self.t]
        self.price = price_history[-1]      # take last price as current
        
        # Ant-Like
        self.alpha = alpha
        self.num_stocks_available = 0           # will be populated later
        self.stocks_left_to_buy = 0
        self.stocks_left_to_sell = 0
        self.state = 0

        # DemandSupply-Like
        self.k = k                              # variance of price decision for random traders
        self.price_history = price_history[:-1] # record historical prices
        self.prob_type = prob_type
        self.volume = 0
        self.demand = []                        # or 0
        self.supply = []                        # or 0
        self.pct_change = []                    # or 0 TODO
        self.p_f = p_f                          # stock value for fundamentalist traders
        
        # to simplify the input of other functions we also store
        # internally variables we will not really use apart from the first loop
        # this helps us in creating a nice lambda function for calibration
        self.cash_low, self.cash_high = cash_low, cash_high
        self.stocks_low, self.stocks_high = stocks_low, stocks_high
        # create agents
        for i, node in enumerate(self.G.nodes()):
            #instantiate the agents
            cash = stats.pareto(0.8, cash_low).rvs()
            stocks = int(stats.pareto(1.5, stocks_low).rvs())
            
            # add the stocks to the total number 
            # this will not change after istantiation
            self.num_stocks_available += stocks         
            # choose trader type from a distribution
            # specified as parameter
            # Ant-Like params
            # DemandSupply params
            trader_type = np.random.choice([0,1,2,3], p = prob_type)
            if np.isin(trader_type, [0,1]):
                tau = random.randint(10, 100)
            else:
                tau = random.randint(10, 50)
            
            # agent creation 
            agent = Ant_Financial_Agent(i, self,
                                        cash = cash, stocks = stocks,                       # general
                                        trader_type= trader_type, tau = tau, p_f = p_f      # DemandSupply-Like
                                        )
            # place agent in graph and add it to the schedule
            self.grid.place_agent(agent, node)
            self.schedule.add(agent)
        
        for agent in self.schedule.agents:
            # initialize neighbors for every agent who will be fixed 
            # throughout the model
            agent.config_neighbors()

        # collect relevant data
        self.datacollector = DataCollector(
            agent_reporters = {
                            'trader_type':'trader_type',
                            'cash': 'cash',
                            'stocks':'stocks',
                            'wealth':'wealth',
                            'quantity':'quantity',
                            'price':'p_i', 
                            'buy_sell': 'buy_sell',
                            'p_buy':'p_buy',
                            'p_active':'p_active',
                            'x' : 'x',
                            'y' : 'y'
                               },
            model_reporters = {
                            'magnetization' : 'state',
                            'activity': self.compute_activity,
                            'T':'T',
                            'price':'price',
                            'nest_location': get_nest_location,
                            'volume':'volume'
                               }
                            )

    ############# MODEL STEP FUNCTIONS ######################
    def step(self):
        self.datacollector.collect(self)
        self.stocks_left_to_buy = self.num_stocks_available
        for agent in np.random.permutation(self.schedule.agents):
            # first loop, agents become active and place their orders
            agent.step1()
            

        old_price = self.price
        self.state = self.compute_magnetization()
        debug=False
        if self.t % 5 == 0:
            debug=True
        self.price_history.append(old_price)
        # check if some orders where placed
        if len(self.supply) > 0 and len(self.demand) > 0:
            # if there are both buyers and sellers, determine new price
            new_price, quantity = self.determine_price(debug=debug)
        else:
            print('no buy/sellers')
            new_price = old_price
            quantity = 0
        self.price = new_price
        self.volume = quantity
        self.stocks_left_to_buy = quantity
        self.stocks_left_to_sell = quantity

        for agent in np.random.permutation(self.schedule.agents):
            # second loop, agents buy or sell depending on new price
            agent.step2()
       
        # change historical variables anyways
        self.t += 1
        try:
            self.T = self.external_var[self.t]
        except:
            self.T = self.external_var[-1]
        # number of stocks never changes as supply + demand == 0
        # save percentage change in price
        self.pct_change.append((new_price - old_price) / old_price)
        # reset supply and demand
        self.reset()
        # call actual step command
        
        self.schedule.step()

    def compute_magnetization(self):
        '''
        Computes magnetization of model.
        '''
        agents_state = [agent.buy_sell for agent in self.schedule.agents]
        return np.sum(agents_state) / self.N

    def compute_activity(self):
        '''
        Computes % of investors active in a given moment
        '''
        agents_state = np.array([agent.buy_sell for agent in self.schedule.agents])
        return (agents_state[agents_state != 0].size) / self.N



    
    ############################################
    ############# ANT-LIKE FUNCTIONS ######################
    # we have everything in the agent, at the moment

    ############################################
    ############# DEMANDSUPPLY-LIKE FUNCTIONS ######################
    def determine_price(self, debug):
        '''
        Finds the price of transactions and the quantity given two arrays of supply and demand. 
        '''
        supply = np.array(self.supply)
        demand = np.array(self.demand)

        idx_s = np.argsort(supply[:,0])
        supply = supply[idx_s]
        supply[:,1] = supply[:,1].cumsum()
        
        idx_d = np.argsort(demand[:,0])
        demand = demand[idx_d[::-1], :]
        demand[:,1] = demand[:,1].cumsum()
        idx_d = np.argsort(demand[:,0])
        demand = demand[idx_d, :]
        new_price, quantity = find_crossing_points(supply, demand, self.price, debug=debug)
        return new_price, quantity

    def reset(self):
        self.supply = []
        self.demand = []

    ############################################
    ###################################################################################
    ###################################################################################
    ############# END OF CLASS ######################
    ###################################################################################
    ###################################################################################




# def update_alpha(self):
    #     pass

    # def step(self):
    #     '''
    #     Steps:
    #         1. checks cash is minimum 5 and stocks are nonnegative
    #         2. draws alpha randomly -- to be updated 
    #         3. calculates utility depending on alpha
    #         4. computes score based on utility: score in [-1, 1] where
    #             -1 represents selling everything and 1 buying as much as one can
    #         5. willingness to buy/sell evaluated in comparison with stocks alrady owned
    #         6. based on willingness, buy/sell a random amount drawn from a poisson distribution
    #         7. move agents on quadrant based on new cash/stocks'''

    #     self.check_status()


    #     risk = self.risk_propensity

    #     uncertainty_level = 0

    #     alpha = self.model.rt_perceived

    #     self.utility += self.calculate_utility(alpha)
    #     u = self.utility

    #     score = 0.5 + 0.5*np.tanh(risk*u)

    #     willingness = score - self.y
    #     self.willingness = willingness

    #     if willingness > 0:
    #         self.state = 1
    #         can_buy = np.minimum(self.cash // self.model.price, 200)
    #         if (can_buy * willingness) < 0:
    #             print('Attention!!!', self.cash, self.model.price, can_buy)
    #         # @Dario added np.abs willingness, does this make sense?
    #         tentative = np.random.poisson((can_buy * np.abs(willingness)))
    #         quantity = np.minimum(np.minimum(tentative, can_buy), self.model.num_available)
    #         self.stocks_owned += quantity
    #         self.cash -= quantity * self.model.price
    #         self.model.demand += quantity
    #         self.model.num_available -= quantity

    #     else:
    #         self.state = -1
    #         can_sell = self.stocks_owned

    #         try:
    #             tentative = np.random.poisson((can_sell * np.abs(willingness)))
            
    #         except:
    #             tentative = 0
    #             print('wrong selling behavior')
    #             print(can_sell)
    #         quantity = np.minimum(can_sell, tentative)
    #         self.stocks_owned -= quantity
    #         if self.stocks_owned < 0:
    #             print(quantity, can_sell)
    #         self.cash += quantity * self.model.price
    #         self.model.supply += quantity
    #         self.model.num_available += quantity

    #     self.move()
        

    # def calculate_utility(self, alpha):
    #     '''
    #     Should compute some sort of local utility:
    #         - mixture of Temperature (Rt) and state behavior of neighbors depending on alpha
    #         - state represents buying/selling behavior 
    #     '''
    #     # utility should depend on price somehow
    #     # will fix how it calculates it to account for no neighbors case
    #     neighbor_states = np.array([nh.state for nh in self.get_neighbors()])
    #     num_nh = len(neighbor_states)
    #     # notice that the G.nodes() function that gives numbers 0, ..., N - 1
    #     # is the assignment to the ID of the Mesa Model Class
    #     # thus, to access weights we can go back to the original interaction_graph instance
    #     # stored in self.G and retrieve each weight 
    #     # namely, access the double dictionary of 
        
    #     # edges_weights = [self.model.G[self.unique_id][nh.unique_id]['weight'] for nh in self.get_neighbors()]
        
    #     edges_weights = 1
    #     # it could be that if no neighbors are present, the value of the two variables above is 
    #     # invalid (empty lists etc), we use a fill value to account for this case and just ignore
    #     # the second term of the expression
    #     temperature_contrib = -alpha*(self.model.T)
    #     # NOTICE: works but given a RuntimeWarning, we could 
    #     # fix this in later versions
    #     neigh_contrib = np.where((num_nh == 0),
    #                               0, 
    #                               self.model.beta * np.sum(edges_weights * neighbor_states) / num_nh)
    #     # reads do the multiplication but when the num of neighbors is zero
    #     # just put zero

    #     price_contrib = -2 * self.model.price_change

    #     u = temperature_contrib + neigh_contrib #+ price_contrib
    #     return u

    # def model_update(self):
    #     self.t += 1
    #     old_price = self.price
    #     new_price = determine_price(self, elasticity=0.2)
    #     self.price = new_price
    #     # new_price = old_price + old_price * pct
    #     if new_price < 0:
    #         print('negative_price')
    #     self.price_change = (new_price - old_price) / old_price

    #     self.num_available = self.num_available + self.supply - self.demand

    #     if self.t < len(self.external_var):
    #         self.T = self.external_var[self.t]
    #         self.rt_perceived = self.rt_change[self.t]

# def determine_price(model, elasticity):

#     excess_demand = model.demand - model.supply
#     # excess_stocks = model.num_stocks - model.num_available
#     # volume = model.demand + model.supply
#     pct = excess_demand / model.num_stocks
#     pct = elasticity * pct * model.price
#     new_price = np.maximum(model.price + pct, 0.1)
#     return new_price

# def activate(beta, h):
 #    '''
 #    Activation function, a sigmoid, RESTORED
 #    '''
 #    return np.where((0.5+0.5*np.tanh(beta*h)) > np.random.rand(),1,-1)

        



  

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
