#!/usr/bin/env python

# MCMC classes
# for fitting stellar fluxes to a trended sinusoid,
# with long-term polynomial trends
# ---
# Abhimat Gautam

import numpy as np
import copy

class mcmc_fitter(object):
    poly_trend_order_base = 1
    
    def __init__(self):
        return
    
    # Function to set polynomial trend order
    def set_poly_trend_order_base(self, poly_trend_order_base):
        self.poly_trend_order_base = poly_trend_order_base
    
    # Functions to set fit specifics
    def set_t0(self, t0):
        self.t0 = t0
    
    def set_period(self, period):
        self.period = period
        self.omega = 2. * np.pi / period
    
    # Function to set observation filters
    def set_observation_filts(self, obs_filts):
        self.obs_filts = obs_filts
        
        self.kp_obs_filt = np.where(self.obs_filts == b'kp')
        self.h_obs_filt = np.where(self.obs_filts == b'h')
    
    # Function to set observation times
    def set_observation_times(self, obs_days):
        self.obs_days = obs_days
    
    # Function to set observation mags
    def set_observations(self, obs, obs_errors):
        self.obs = obs
        self.obs_errors = obs_errors
        
        self.kp_obs_mags = obs[self.kp_obs_filt]
        self.kp_obs_mag_errors = obs_errors[self.kp_obs_filt]
        
        self.h_obs_mags = obs[self.h_obs_filt]
        self.h_obs_mag_errors = obs_errors[self.h_obs_filt]
        
        print(self.h_obs_mags)
    
    
    # Prior function
    def log_prior(self, theta):
        # Extract model parameters from theta
        theta_index = 0
        
        t0_fit = theta[theta_index]
        theta_index += 1
        
        base_poly_trend_coeffs = np.empty(self.poly_trend_order_base + 1)
        
        for poly_coeff_index in range(self.poly_trend_order_base + 1):
            base_poly_trend_coeffs[poly_coeff_index] = theta[theta_index]
            theta_index += 1
        
        base_cos_coeff = theta[theta_index]
        theta_index += 1
        
        if len(self.h_obs_mags) > 0:
            h_add = theta[theta_index]
            theta_index += 1
            
            h_c1 = theta[theta_index]
            theta_index += 1
        
        # Check all params
        t0_check = ((self.t0 - self.period/2.) < 
                    t0_fit < (self.t0 + self.period/2.))
        
        base_check = (9 < base_poly_trend_coeffs[0] < 22 and
                      -1e-1 < base_poly_trend_coeffs[1] < 1e-1)
        
        if base_check and self.poly_trend_order_base > 1:
            for poly_coeff_index in range(2, self.poly_trend_order_base + 1):
                if not (-1e-2 < base_poly_trend_coeffs[poly_coeff_index] < 1e-2):
                    base_check = False
                    break
        
        cos_check = 1e-2 < base_cos_coeff < 0.8
        
        h_check = True
        if len(self.h_obs_mags) > 0:
            h_check = (-10 < h_add < 10 and
                       -1e-2 < h_c1 < 1e-2)
        
        if t0_check and base_check and cos_check and h_check:
            return 0.0
        
        return -np.inf
    
    # Likelihood function
    def log_likelihood(self, theta, print_checks=False):
        # Extract model parameters from theta
        theta_index = 0
        
        t0_fit = theta[theta_index]
        theta_index += 1
        
        base_poly_trend_coeffs = np.empty(self.poly_trend_order_base + 1)
        
        for poly_coeff_index in range(self.poly_trend_order_base + 1):
            base_poly_trend_coeffs[poly_coeff_index] = theta[theta_index]
            theta_index += 1
        
        base_cos_coeff = theta[theta_index]
        theta_index += 1
        
        if len(self.h_obs_mags) > 0:
            h_add = theta[theta_index]
            theta_index += 1
            
            h_c1 = theta[theta_index]
            theta_index += 1
        
        # Compute model mags
        model_mags = copy.deepcopy(self.obs)*0.
    
        t_term = (self.obs_days - t0_fit)
        
        # Base polynomial model
        for poly_coeff_index in range(self.poly_trend_order_base + 1):
            model_mags += (t_term**poly_coeff_index *
                           base_poly_trend_coeffs[poly_coeff_index])
        
        # Base sinusoid model
        model_mags += base_cos_coeff * np.cos(self.omega * t_term)
        
        # # Kp mags
        # model_mags[kp_obs_filt] += 0.0 +\
        #     ((obs_days[kp_obs_filt] - t0_fit) * kp_c1)
        
        # H mags
        if len(self.h_obs_mags) > 0:
            model_mags[self.h_obs_filt] += h_add +\
                ((self.obs_days[self.h_obs_filt] - t0_fit) * h_c1)
        
        if print_checks:
            print(f'All mags: {model_mags}')
            print(f'Observed mags: {mags}')

            print(f'Kp mags: {model_mags[kp_obs_filt]}')
            print(f'Observed Kp mags: {mags[kp_obs_filt]}')

            print(f'H mags: {model_mags[h_obs_filt]}')
            print(f'Observed H mags: {mags[h_obs_filt]}')
    
        # Uncertainties
        sigma_sq = self.obs_errors ** 2.
    
        return -0.5 * np.sum(((self.obs-model_mags)**2.) / sigma_sq +\
               np.log(2. * np.pi * sigma_sq))
    
    # Posterior probability function
    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)
    
    