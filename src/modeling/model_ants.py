
import numpy as np
from os.path import join

try:
    __IPYTHON__ #TODO what is this
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False
    

if _in_ipython_session:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
    
    
    
class colony:
    
    def __init__(self, beta=1, theta_m=32, theta_sd=2, N=50, Jp=1, Jr=1, alpha=0, interaction='basic'):
        """
        Create and initialize a simulated colony.
        N - number of ants in the colony
        beta - the thermal noise parameter
        theta_m, theta_sd - parameters of the individual response distribution
        alpha, Jp, Jr - parameters of the social dynamics
        interaction - the interaction type simulated in the paper (basic/asymmetric/fixed_inhibition)
        
        """
        
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.interaction = interaction
        self.Jp = Jp
        self.Jr = Jr
        self.theta_m = theta_m
        self.theta_sd = theta_sd
        self.reset()
        
        
    def reset(self):
        """
        Reset the colony to its initial state
        """
        
        self.theta = self.theta_m + self.theta_sd * np.random.standard_normal(self.N)
        self.sigma = -np.ones((self.N,))
        
    def activate(self, h):
        
        return np.where((0.5+0.5*np.tanh(self.beta*h)) > np.random.rand(*h.shape),1,-1)
    
    def colony_state(self):
    
        return np.count_nonzero(self.sigma == 1)/self.N
        
    def immediate_response(self, T):
        """
        This simulates the immediate (individual) response of the colony.
        """
        
        self.sigma = self.activate(T - self.theta)
        
    def h(self):
        
        m = self.colony_state()
    
        if self.interaction=='basic':
            
            # this is actually alpha=0
            h = self.Jp*m*self.N - self.Jr*(1-m)*self.N
            
        elif self.interaction=='full_asymmetric':
            
            # this is actually alpha=1
            h = self.Jp*m*50 - self.Jr*(1-m)*self.N
            
        elif self.interaction=='asymmetric':
            
            alpha = self.alpha
            
            h = self.Jp*m*(self.N**(1-alpha))*(50**alpha) - self.Jr*(1-m)*(self.N**(1+alpha))*(50**(-alpha))

        elif self.interaction=='fixed_inhibition':
            
            alpha = self.alpha
            N = self.N
            A = 0.5/(110**alpha)
            
            h = self.Jp*m*(N**(1-alpha)) - self.Jr*N*A
            
        else:
            
            print('wrong interaction type')
    
        return np.array([h])
        
    def colony_update(self):
        """
        A single step update of the colony state
        """
        
        m = self.colony_state()
    
        # select an ant to update
        i = np.random.randint(0, self.N)
    
        # calclate the input
        hi = self.h()
  
        # update
        self.sigma[i] = self.activate(hi)
        
    def run(self, T, duration=20):
        """
        This will run a single simulation of the colony under perturbation of temperature <T>
        for <duration> update cycles.
        """
        
        # total number of simulation steps
        nsteps = duration * self.N
    
        # initialize the response array
        m = np.zeros((nsteps+1,))
        
        # simulate the immediate response of the ants
        self.immediate_response(T)
        m[0] = self.colony_state()
    
        # simulate the social dynamics 
        for ix in range(nsteps):
            self.colony_update()
            m[ix+1] = self.colony_state()
            
        return m
            
    def T_sweep(self, Tmin=25, Tmax=40, dT=0.1, duration=20, reps=10, progressbar=True):
        """
        This will run a "temperature sweep". The simulation will run <reps> times for
        each temperature in the range. Will return a 3D array (nReps X nT x nSteps) with the responses of all runs.
        The function will also calculate the response threshold for the colony.
        Note that the all repetitions are run with the same colony - the individual response features are not redrawn.
        """
    
        TS = np.arange(Tmin, Tmax, dT)
        m = np.zeros((reps, TS.shape[0], duration*self.N+1))
        
       
        for ix in tqdm(range(reps)): # tqdm is a progress bar library
            self.reset()
            for jx, T in enumerate(TS): #extract index of perturbation and perturbation
                m[ix,jx] = self.run(T, duration=duration) #assign to magnetization index of perturbation and of progress
                
        # calc threshold
        mi = np.take(m,range(-100,0),-1).mean(axis=-1)
        mi = mi.mean(axis=0)
    
        try:
            th = TS[np.where(mi>0.5)[0].min()]
        except:
            th = np.nan

        
        return m, TS, th
    
    
    def param_sweep(self, param, values, Tmin=25, Tmax=40, dT=0.1, duration=20, reps=10):
        """
        This will run a full "parameter sweep" of the colony.
        """
        
        TS = np.arange(Tmin, Tmax, dT)
        m = np.zeros((len(values), reps, TS.shape[0], duration*self.N+1))
        
        for ix, v in enumerate(values):
            setattr(self, param, v)
            m[ix], TS = self.T_sweep(Tmin=Tmin, Tmax=Tmax, dT=dT, duration=duration, reps=reps)
        
        return m, TS
        

def simulate_colony_thermal_response(*, beta=1.0, theta_m=32.0, theta_sd=2.0, gs=50, Jp=1.0, Jr=1.0, interaction='basic', tempmin=25.0, tempmax=40.0, tempdelta=0.05, duration=20, reps=10, outfile=None, alpha=0.0):
    """
    wrapper function to run a single temperature sweep of a colony.
    """
    c = colony(beta=beta, theta_m=theta_m, theta_sd=theta_sd, N=gs, Jp=Jp, Jr=Jr, interaction=interaction, alpha=alpha)
    m, TS, threshold = c.T_sweep(Tmin=tempmin, Tmax=tempmax, dT=tempdelta, duration=duration, reps=reps)
    
    if outfile is not None:
        print('Saving to ' + outfile)
        with open(outfile, 'wb') as f:
            # Save several arrays into a single file in compressed .npz format.
            np.savez_compressed(f, m=m, threshold=threshold, beta=beta, theta_m=theta_m, theta_sd=theta_sd, N=gs, Jp=Jp, Jr=Jr, interaction=interaction, Tmin=tempmin, Tmax=tempmax, dT=tempdelta, duration=duration, reps=reps, TS=TS, alpha=alpha)
    
    
if __name__ == '__main__':
    
    from clize import run

    run(simulate_colony_thermal_response)    
    print('Finished!')
        
    