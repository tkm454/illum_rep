from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import math

class MalthusModelClass:

    def __init__(self):
        self.par = SimpleNamespace(beta=0.5, delta=3.0, tau=0.0, y=100)
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

    

 