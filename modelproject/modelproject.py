from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import math
import matplotlib as plt
import ipywidgets as widgets

class MalthusModelClass:

    def __init__(self):
        self.par = SimpleNamespace(beta=0.5, delta=3.0, tau=0.0, y=100, eta=0.3, mu=0.02, alpha=1/3, A=10, X=10, T=200)
        self.c = None
        self.n = None
        self.u = None

    def utility_func(self, c, n):
        return (1-self.par.beta)*math.log(c) + self.par.beta*math.log(n) 
    
    def solve(self):
        
        obj = lambda x: -self.utility_func(x[0], x[1])
        
        budget_constraint = lambda x: self.par.y*(1-self.par.tau) - x[0] - self.par.delta*x[1]
        constraints = [{'type': 'ineq', 'fun': budget_constraint}]
        x0 = [self.par.y/2, self.par.y/2]
        
        sol = optimize.minimize(obj, x0, method='SLSQP', constraints=constraints)
        self.c = sol.x[0]
        self.n = sol.x[1]
        self.u = self.utility_func(self.c, self.n)
        
    def find_ss_l(self):

        """
        Output: Steady state value of population  
        """
        result = optimize.root_scalar(lambda L: L-((self.par.eta/self.par.mu)**(1-self.par.alpha)*self.par.A*self.par.X), bracket=[-5000,5000], method='brentq')
        return result.root
    
    def find_ss_y(self):

        """ 
        Input: Parameters of interest, and the interval [a,b]. This determines the interval on which the function evaluates 

        Output: Steady state value of population.  

        """
        result = optimize.root_scalar(lambda y: y-(self.par.eta/self.par.mu), bracket=[-300,300], method='brentq')
        return result.root
        
    def simulate_malthus_l(self,T):
        """
        Simulating the model 
        
        """

        # create lists for diagonal line s and population L
        l_values = []
        s_values = []

        # Create the population movement

        l=100  #baseline

        for t in range(0,T):
                l_new = ((1-self.par.beta/self.par.delta)*self.par.A*self.par.X*l**(1-self.par.alpha)+(1-self.par.mu)*l)
                l_values.append(l_new)
                l = l_new

        for t in range(0,self.par.T):
                s_new = t
                s_values.append(s_new)
    
        # steady state
        ss = self.find_ss_l()
        
        # plot
        plt.figure(figsize=(5,5))
        plt.plot(s_values[:T], l_values[:T], label=r'$L_{t+1}=\frac{\gamma}{\rho}(AX)^\alpha L_t^{1-\alpha}$', color = 'blue')
        plt.plot(s_values[:T], s_values[:T], label='45 degree line', color = 'black')
        plt.scatter(ss, ss, c='g', linewidths=3, label='Steady State')
        plt.text(ss, ss, '({}, {})'.format(round(ss,2), round(ss,2)))
        plt.xlim(0,T)
        plt.ylim(0,T)
        plt.ylabel('$L_{t+1}$')
        plt.xlabel('$L_t$')
        plt.grid(True)
        plt.legend()
        return plt.show()