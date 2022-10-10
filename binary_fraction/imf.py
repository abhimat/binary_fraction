#!/usr/bin/env python

# Basic IMF functionality
# ---
# Abhimat Gautam

import numpy as np

class IMF(object):
    def  __init__(self, mass_limits=np.array([10, 100])):
        self.mass_limits = mass_limits
        
        return
    
class IMF_power_law(IMF):
    def __init__(self, mass_limits=np.array([10, 100]), alpha=1.7):
        
        self.mass_limits = mass_limits
        self.mass_lo = mass_limits[0]
        self.mass_hi = mass_limits[1]
        
        self.alpha = alpha
        
        self.calculate_pl_coeff()
        
        return
    
    def calculate_pl_coeff(self):
        
        
        self.pl_coeff = ((1. - self.alpha) /
                         (self.mass_hi**(1. - self.alpha) - 
                          self.mass_lo**(1. - self.alpha)))
        
        return
    
    def p_m(self, m):
        return self.pl_coeff * (m ** (-1. * self.alpha))
    
    def cdf_m(self, m):
        cdf = ((m**(1. - self.alpha) - 
               self.mass_lo**(1. - self.alpha)) /
               (self.mass_hi**(1. - self.alpha) - 
                self.mass_lo**(1. - self.alpha)))
        return cdf
    
    def inv_cdf_u(self, u):
        temp_var = (u*((self.mass_hi / self.mass_lo)**(1. - self.alpha) - 1.) + 1.)
        
        inv_cdf = self.mass_lo * np.exp(np.log(temp_var) / (1. - self.alpha))
        
        return inv_cdf
            
    
    def draw_imf_mass(self, rand=-1.):
        if rand == -1.:
            rand = np.random.rand()
        
        return self.inv_cdf_u(rand)