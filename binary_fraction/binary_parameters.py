#!/usr/bin/env python

# Class to generate binary parameters
# ---
# Abhimat Gautam

import os
import numpy as np
from astropy.table import Table
from binary_fraction import imf
from binary_fraction.distributions import (power_law_dist, cos_inc_dist, log_norm_unimodal)

class binary_population(object):
    def  __init__(self):
        # Make default distributions
        self.make_imf()
        self.make_period_dist()
        self.make_q_dist()
        self.make_ecc_dist()
        self.make_inc_dist()
        
        return
    
    # Functions to define different distributions for binary population
    def make_imf(self, mass_limits=np.array([10, 100]), alpha=1.7):
        self.imf = imf.IMF_power_law(mass_limits=mass_limits, alpha=alpha)
        return
    
    def make_period_dist(self, period_limits=np.array([1., 10.**3.5]), pl_exp=-0.55):
        self.period_dist = power_law_dist(limits=period_limits, pl_exp=pl_exp)
        return
    
    def make_period_dist_log_norm_unimodal(
            self, mode_logp=5, sigma_logp=1,
        ):
        self.period_dist = log_norm_unimodal(
            log_mode=mode_logp, log_sigma=sigma_logp,
        )
        return
    
    def make_q_dist(self, q_limits=np.array([0.1, 1.]), pl_exp=-0.1):
        self.q_dist = power_law_dist(limits=q_limits, pl_exp=pl_exp)
        return
    
    def make_ecc_dist(self, ecc_limits=np.array([0., 1.]), pl_exp=-0.45):
        self.ecc_dist = power_law_dist(limits=ecc_limits, pl_exp=pl_exp)
        return
    
    def make_inc_dist(self):
        self.inc_dist = cos_inc_dist()
    
    # Function to generate binary parameters
    def generate_binary_params(self, print_diagnostics=False, log_per_max=None):
        ## Primary star mass
        mass_1 = self.draw_mass_imf()
        
        ## Secondary star mass, derived from drawn mass ratio
        binary_q = self.draw_q()
        mass_2 = binary_q * mass_1
        
        ## Binary period
        binary_period = self.draw_period()
        if log_per_max != None:
            binary_period = self.draw_period(max_log_draw=log_per_max)
        binary_t0_shift = self.draw_t0_shift(binary_period)
        
        ## Binary eccentricity
        binary_ecc = self.draw_ecc()
        
        ## Binary inclination
        binary_inc = self.draw_inc()
        
        
        if print_diagnostics:
            out_str = ''
            out_str += 'Mass 1 = {0:.3f} solMass\n'.format(mass_1)
            out_str += 'Mass 2 = {0:.3f} solMass\n'.format(mass_2)
            out_str += 'q = {0:.3f}\n'.format(binary_q)
            out_str += 'P = {0:.3f} days\n'.format(binary_period)
            out_str += 't0 shift = {0:.3f} days\n'.format(binary_t0_shift)
            out_str += 'e = {0:.3f}\n'.format(binary_ecc)
            out_str += 'i = {0:.3f} deg\n'.format(binary_inc)
            
            print(out_str)
        
        ## Return a tuple of all the generated binary parameters
        return (mass_1, mass_2,
                binary_period, binary_t0_shift,
                binary_q, binary_ecc, binary_inc)
        
    
    # Functions to draw individual parameters from distributions
    def draw_mass_imf(self):
        return self.imf.draw_imf_mass()
    
    def draw_period(self, max_log_draw=None):
        if max_log_draw != None:
            return self.period_dist.draw(max_log_draw=max_log_draw)
        else:
            return self.period_dist.draw()
    
    def draw_t0_shift(self, binary_period):
        return binary_period * np.random.rand()
    
    def draw_q(self):
        return self.q_dist.draw()
    
    def draw_ecc(self):
        return self.ecc_dist.draw()
    
    def draw_inc(self):
        return self.inc_dist.draw()


def generate_binary_population_params(
        binary_population, num_binaries,
        out_dir='./mock_binaries',
        log_per_max=None,
    ):
    # Make sure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Make a table for output binary parameters
    num_params = 7
    
    binary_pop_params = np.empty([num_binaries, num_params])
    
    # Draw binary parameters
    for cur_binary_index in range(num_binaries):
        cur_binary_params = binary_population.generate_binary_params(log_per_max=log_per_max)
        (mass_1, mass_2,
         binary_period, binary_t0_shift,
         binary_q, binary_ecc, binary_inc) = cur_binary_params
        
        binary_pop_params[cur_binary_index] = [mass_1, mass_2,
                                               binary_period, binary_t0_shift,
                                               binary_q, binary_ecc, binary_inc]
    
    # Generate astropy table object
    binary_pop_params_table = Table([np.arange(num_binaries, dtype=int),
                                     binary_pop_params[:,0],
                                     binary_pop_params[:,1],
                                     binary_pop_params[:,2],
                                     binary_pop_params[:,3],
                                     binary_pop_params[:,4],
                                     binary_pop_params[:,5],
                                     binary_pop_params[:,6],
                                    ],
                                    names=('binary_index', 'mass_1', 'mass_2',
                                           'binary_period', 'binary_t0_shift',
                                           'binary_q', 'binary_ecc', 'binary_inc'))
    
    binary_pop_params_table.write(
        '{0}/binary_pop_params.h5'.format(out_dir),
        path='data', serialize_meta=True, compression=True,
        overwrite=True,
    )
    
    binary_pop_params_table.write('{0}/binary_pop_params.txt'.format(out_dir),
                                  overwrite=True, format='ascii.fixed_width')
    
    return binary_pop_params_table
    