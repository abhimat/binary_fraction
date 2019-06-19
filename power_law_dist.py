#!/usr/bin/env python

# Basic IMF functionality
# ---
# Abhimat Gautam

import numpy as np

class power_law_dist(object):
    def __init__(self, limits=np.array([10, 100]), pl_exp=1.7):
        ## Save out parameters of the power law distribution
        self.limits = limits
        self.limit_lo = limits[0]
        self.limit_hi = limits[1]
        
        self.pl_exp = exp
        
        ## Calculate constant coefficient to normalize
        self.calculate_pl_coeff()
        
        return
    
    def calculate_pl_coeff(self):
        self.pl_coeff = ((1. - self.pl_exp) /
                         (self.limit_hi**(1. - self.pl_exp) - 
                          self.limit_lo**(1. - self.pl_exp)))
        
        return
    
    def p_x(self, x):
        return self.pl_coeff * (x** (-1. * self.pl_exp))
    
    def cdf_x(self, x):
        cdf = ((x**(1. - self.pl_exp) - 
               self.limit_lo**(1. - self.pl_exp)) /
               (self.limit_hi**(1. - self.pl_exp) - 
                self.limit_lo**(1. - self.pl_exp)))
        return cdf
    
    def inv_cdf_u(self, u):
        temp_var = (u*((self.limit_hi / self.limit_lo)**(1. - self.pl_exp) - 1.) + 1.)
        
        inv_cdf = self.limit_lo * np.exp(np.log(temp_var) / (1. - self.pl_exp))
        
        return inv_cdf
            
    
    def power_law_draw(self, rand=-1.):
        if rand == -1.:
            rand = np.random.rand()
        
        return self.inv_cdf_u(rand)
