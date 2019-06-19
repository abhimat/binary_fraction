#!/usr/bin/env python

# Class to generate binary parameters
# ---
# Abhimat Gautam

import numpy as np
import imf
import power_law_dist

class binary_parameters(object):
    def  __init__(self, mass_limits=np.array([10, 100])):
        self.mass_limits = mass_limits
        
        return
    
    def make_imf(mass_limits=np.array([10, 100]), alpha=1.7):
        self.imf = imf.IMF_power_law(mass_limits=mass_limits, alpha=alpha)
    
    
    def generate_binary_params():
        mass_1 = draw_mass_imf(self.imf)
        mass_2 = draw uniform 
    
    def draw_mass_imf(imf):
        return imf.draw_imf_mass()
        
