#!/usr/bin/env python

# Distributions
# ---
# Abhimat Gautam

__all__ = ['power_law_dist', 'cos_inc_dist', 'lognorm_unimodal']

import numpy as np

class power_law_dist(object):
    def __init__(self, limits=np.array([10, 100]), pl_exp=-1.7):
        ## Save out parameters of the power law distribution
        self.limits = limits
        self.limit_lo = limits[0]
        self.limit_hi = limits[1]
        
        self.pl_exp = pl_exp
        
        ## Calculate constant coefficient to normalize
        self.calculate_pl_coeff()
        
        return
    
    def calculate_pl_coeff(self):
        self.pl_coeff = ((1. + self.pl_exp) /
                         (self.limit_hi**(1. + self.pl_exp) - 
                          self.limit_lo**(1. + self.pl_exp)))
        
        return
    
    def p_x(self, x):
        return self.pl_coeff * (x ** self.pl_exp)
    
    def cdf_x(self, x):
        cdf = ((x**(1. + self.pl_exp) - 
               self.limit_lo**(1. + self.pl_exp)) /
               (self.limit_hi**(1. + self.pl_exp) - 
                self.limit_lo**(1. + self.pl_exp)))
        return cdf
    
    def inv_cdf_u(self, u):
        temp_var = (u*(self.limit_hi**(1. + self.pl_exp) - self.limit_lo**(1. + self.pl_exp))
                    + self.limit_lo**(1. + self.pl_exp))
        
        inv_cdf = np.exp(np.log(temp_var) / (1. + self.pl_exp))
        
        return inv_cdf
            
    
    def draw(self, rand=-1.):
        if rand == -1.:
            rand = np.random.rand()
        
        return self.inv_cdf_u(rand)

class cos_inc_dist(object):
    def __init__(self):
        return
    
    def draw(self):
        cos_i_draw = (2. * np.random.sample()) - 1.
        
        return np.rad2deg(np.arccos(cos_i_draw))

class log_norm_unimodal(object):
    def __init__(self, log_mode=0.0, log_sigma=1.0, log_base=10.):
        # Save out parameters
        self.log_mode = log_mode
        self.log_sigma = log_sigma
        self.log_base = log_base
        
        self.rng = np.random.default_rng()
    
    def draw(self, max_log_draw=None):
        log_draw = self.rng.normal(self.log_mode, self.log_sigma)
        
        if max_log_draw != None:
            while log_draw > max_log_draw:
                log_draw = self.rng.normal(self.log_mode, self.log_sigma)
        
        draw = self.log_base**log_draw
        
        return draw
