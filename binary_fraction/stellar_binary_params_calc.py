#!/usr/bin/env python

# Class to calculate stellar and binary parameters for mock binary systems
# ---
# Abhimat Gautam

import os
import numpy as np
from binary_fraction import imf
from astropy.table import Table
from phitter import observables, filters
from phitter.params import star_params, binary_params, isoc_interp_params
from phoebe import u
from phoebe import c as const
import parmap
from tqdm import tqdm

# Default filters
kp_filt = filters.nirc2_kp_filt()
h_filt = filters.nirc2_h_filt()

# Function to help with parallelization
def params_from_binary_row(cur_binary_row, s_b_params_calc_obj):
    return s_b_params_calc_obj.calc_stellar_binary_params(
                cur_binary_row['binary_index'],
                cur_binary_row['mass_1'],
                cur_binary_row['mass_2'],
                cur_binary_row['binary_period'],
                cur_binary_row['binary_t0_shift'],
                cur_binary_row['binary_q'],
                cur_binary_row['binary_ecc'],
                cur_binary_row['binary_inc'],
            )

class stellar_binary_params_calc(object):
    def  __init__(self):
        # Set up some defaults
        self.set_extLaw_alpha()
        self.set_pop_distance()
        
        return
    
    def set_extLaw_alpha(self, ext_alpha=2.30):
        self.ext_alpha = ext_alpha
    
    def set_pop_distance(self, pop_distance=7.971e3):
        self.pop_distance = pop_distance
    
    def make_pop_isochrone(
            self,
            isoc_age=4.0e6, isoc_ext_Ks=2.54,
            isoc_dist=7.971e3, isoc_phase=None,
            isoc_met=0.0,
            isoc_atm_func = 'merged',
            isoc_filts_list=[
                kp_filt, h_filt,
            ],
        ):
        # Store out isochrone parameters into object
        self.isoc_age=isoc_age,
        self.isoc_ext=isoc_ext_Ks,
        self.isoc_dist=isoc_dist,
        self.isoc_phase=isoc_phase,
        self.isoc_met=isoc_met
        self.isoc_filts_list=isoc_filts_list
        
        # Generate isoc_interp object
        self.pop_isochrone = isoc_interp_params.isoc_mist_stellar_params(
            age=isoc_age, met=isoc_met,
            use_atm_func=isoc_atm_func,
            phase=isoc_phase,
            ext_Ks=isoc_ext_Ks,
            dist=isoc_dist * u.pc,
            filts_list=isoc_filts_list,
            ext_law='NL18',
        )
        
        # Also set population extinction based on isochrone extinction
        ## Filter properties
        lambda_Ks = 2.18e-6 * u.m
        dlambda_Ks = 0.35e-6 * u.m
        
        self.filt_lambdas = []
        self.filt_dlambdas = []
        
        for filt in self.isoc_filts_list:
            self.filt_lambdas.append(filt.lambda_filt)
            self.filt_dlambdas.append(filt.dlambda_filt)
        
        ## Calculate default population extinction
        self.isoc_filt_exts = np.empty(len(self.isoc_filts_list))
        
        for filt_index, filt in enumerate(self.isoc_filts_list):
            self.isoc_filt_exts[filt_index] = isoc_ext_Ks *\
                (lambda_Ks / self.filt_lambdas[filt_index])**self.ext_alpha
    
    
    def calc_stellar_binary_params(
            self, binary_index, mass_1, mass_2,
            binary_period, binary_t0_shift,
            binary_q, binary_ecc, binary_inc):
        """Calculate the stellar and binary parameters for the current mock
        binary system
        """
        # Interpolate stellar parameters from isochrone
        star1_params = self.pop_isochrone.interp_star_params_mass_init(mass_1)
        star2_params = self.pop_isochrone.interp_star_params_mass_init(mass_2)
        
        star1_mass_init = star1_params.mass_init
        star1_mass = star1_params.mass
        star1_rad = star1_params.rad
        star1_lum = star1_params.lum
        star1_teff = star1_params.teff
        star1_logg = star1_params.logg
        star1_mags = star1_params.mags
        star1_pblums = star1_params.pblums
        
        star2_mass_init = star2_params.mass_init
        star2_mass = star2_params.mass
        star2_rad = star2_params.rad
        star2_lum = star2_params.lum
        star2_teff = star2_params.teff
        star2_logg = star2_params.logg
        star2_mags = star2_params.mags
        star2_pblums = star2_params.pblums
        
        star1_den = star1_mass / ((4./3.) * np.pi * star1_rad**3.)
        star2_den = star2_mass / ((4./3.) * np.pi * star2_rad**3.)
        
        # Calculate binary parameters
        binary_sma = ((((binary_period * u.d)**2. * const.G * (star1_mass + star2_mass)) /
                       (4. * np.pi**2.))**(1./3.) )

        binary_q = star2_mass / star1_mass
        binary_q_init = star2_mass_init / star1_mass_init
    
        binary_mass_func = ((star2_mass ** 3.) * (np.sin(binary_inc) ** 3.) /
                                (star1_mass + star2_mass)**2. )
        
        # Semi-amplitude of RV signal
        binary_rv_K = (2. * np.pi * binary_sma * np.sin(binary_inc * u.deg) /
                       (binary_period * u.d)).to(u.km / u.s)
        
        # Fraction Phase of Eclipse
        rad_tot = star1_rad + star2_rad
        mass_tot = star1_mass + star2_mass
        
        binary_frac_phase_ecl = 4 * rad_tot *\
            (2. * np.pi * (binary_period * u.d)**2. * const.G * mass_tot)**(-1./3.)
        
        binary_frac_phase_ecl = binary_frac_phase_ecl.to(1).value
        
        if binary_frac_phase_ecl > 1.5:
            binary_frac_phase_ecl = 1.5
        
        # Generate output tuple
        out_params = (star1_mass_init.to(u.solMass).value,
                      star1_mass.to(u.solMass).value,
                      star1_rad.to(u.solRad).value,
                      star1_den.to(u.solMass * u.solRad**-3).value,
                      star1_lum.to(u.solLum).value,
                      star1_teff.to(u.K).value,
                      star1_logg,
                      star2_mass_init.to(u.solMass).value,
                      star2_mass.to(u.solMass).value,
                      star2_rad.to(u.solRad).value,
                      star2_den.to(u.solMass * u.solRad**-3).value,
                      star2_lum.to(u.solLum).value,
                      star2_teff.to(u.K).value,
                      star2_logg,
                      binary_period, binary_q.value, binary_q_init.value,
                      binary_ecc, binary_inc,
                      binary_sma.to(u.AU).value,
                      binary_mass_func.to(u.solMass).value,
                      binary_rv_K.to(u.km / u.s).value,
                      binary_frac_phase_ecl,
                     )
        return out_params
    
    def calc_population_params(self, binary_pop_params_file,
                               parallelize=True):
        
        # Read in table of binary parameters
        binary_pop_params_table = Table.read(
            binary_pop_params_file,
            format='ascii.fixed_width',
        )
        
        num_binaries = len(binary_pop_params_table)
        
        # Generate stellar and binary parameters for all mock binaries
        out_params = np.array(parmap.map(
                        params_from_binary_row,
                        binary_pop_params_table, self,
                        pm_pbar=True, pm_parallel=parallelize
                     ))
        
        # Empty arrays to store stellar parameters
        star1_mass_init_samps = np.empty(num_binaries) * u.solMass
        star1_mass_samps = np.empty(num_binaries) * u.solMass
        star1_rad_samps = np.empty(num_binaries) * u.solRad
        star1_den_samps = np.empty(num_binaries) * u.solMass * u.solRad**-3
        star1_lum_samps = np.empty(num_binaries) * u.solLum
        star1_teff_samps = np.empty(num_binaries) * u.K
        star1_logg_samps = np.empty(num_binaries)

        star2_mass_init_samps = np.empty(num_binaries) * u.solMass
        star2_mass_samps = np.empty(num_binaries) * u.solMass
        star2_rad_samps = np.empty(num_binaries) * u.solRad
        star2_den_samps = np.empty(num_binaries) * u.solMass * u.solRad**-3
        star2_lum_samps = np.empty(num_binaries) * u.solLum
        star2_teff_samps = np.empty(num_binaries) * u.K
        star2_logg_samps = np.empty(num_binaries)

        # Binary parameters
        binary_period_samps = np.empty(num_binaries) * u.d
        binary_q_samps = np.empty(num_binaries)
        binary_q_init_samps = np.empty(num_binaries)
        binary_ecc_samps = np.empty(num_binaries)
        binary_inc_samps = np.empty(num_binaries) * u.degree
        binary_sma_samps = np.empty(num_binaries) * u.AU
        binary_mass_func_samps = np.empty(num_binaries) * u.solMass
        binary_rv_K_samps = np.empty(num_binaries) * u.km / u.s
        binary_frac_phase_ecl_samps = np.empty(num_binaries)
        
        # Put parameter outputs into arrays
        for cur_bin_index in tqdm(range(num_binaries)):
            if type(out_params[cur_bin_index]) is tuple:
                # print(cur_bin_index)
                (star1_mass_init, star1_mass, star1_rad, star1_den,
                 star1_lum, star1_teff, star1_logg,
                 star2_mass_init, star2_mass, star2_rad, star2_den,
                 star2_lum, star2_teff, star2_logg,
                 binary_period, binary_q, binary_q_init,
                 binary_ecc, binary_inc,
                 binary_sma,
                 binary_mass_func,
                 binary_rv_K, binary_frac_phase_ecl,
                ) = (np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan,
                     np.nan, np.nan,
                     np.nan, np.nan,
                     np.nan, np.nan,
                    )
            else:
                (star1_mass_init, star1_mass, star1_rad, star1_den,
                 star1_lum, star1_teff, star1_logg,
                 star2_mass_init, star2_mass, star2_rad, star2_den,
                 star2_lum, star2_teff, star2_logg,
                 binary_period, binary_q, binary_q_init,
                 binary_ecc, binary_inc,
                 binary_sma,
                 binary_mass_func,
                 binary_rv_K, binary_frac_phase_ecl,
                ) = out_params[cur_bin_index]
            
            # Stellar parameters
            star1_mass_init_samps[cur_bin_index] = star1_mass_init * u.solMass
            star1_mass_samps[cur_bin_index] = star1_mass * u.solMass
            star1_rad_samps[cur_bin_index] = star1_rad * u.solRad
            star1_den_samps[cur_bin_index] = star1_den * u.solMass * u.solRad**-3
            star1_lum_samps[cur_bin_index] = star1_lum * u.solLum
            star1_teff_samps[cur_bin_index] = star1_teff * u.K
            star1_logg_samps[cur_bin_index] = star1_logg

            star2_mass_init_samps[cur_bin_index] = star2_mass_init * u.solMass
            star2_mass_samps[cur_bin_index] = star2_mass * u.solMass
            star2_rad_samps[cur_bin_index] = star2_rad * u.solRad
            star2_den_samps[cur_bin_index] = star2_den * u.solMass * u.solRad**-3
            star2_lum_samps[cur_bin_index] = star2_lum * u.solLum
            star2_teff_samps[cur_bin_index] = star2_teff * u.K
            star2_logg_samps[cur_bin_index] = star2_logg

            # Binary parameters
            binary_period_samps[cur_bin_index] = binary_period * u.d
            binary_q_samps[cur_bin_index] = binary_q
            binary_q_init_samps[cur_bin_index] = binary_q_init
            binary_ecc_samps[cur_bin_index] = binary_ecc
            binary_inc_samps[cur_bin_index] = binary_inc * u.degree
            binary_sma_samps[cur_bin_index] = binary_sma * u.AU
            binary_mass_func_samps[cur_bin_index] = binary_mass_func * u.solMass
            binary_rv_K_samps[cur_bin_index] = binary_rv_K * u.km / u.s
            binary_frac_phase_ecl_samps[cur_bin_index] = binary_frac_phase_ecl
        
        # Make parameter table for output
        
        params_table = Table([np.arange(num_binaries, dtype=int),
                              star1_mass_init_samps, star1_mass_samps,
                              star1_rad_samps, star1_den_samps,
                              star1_lum_samps, star1_teff_samps,
                              star1_logg_samps,
                              star2_mass_init_samps, star2_mass_samps,
                              star2_rad_samps, star2_den_samps,
                              star2_lum_samps, star2_teff_samps,
                              star2_logg_samps,
                              binary_period_samps,
                              binary_q_samps, binary_q_init_samps,
                              binary_ecc_samps, binary_inc_samps,
                              binary_sma_samps, binary_mass_func_samps,
                              binary_rv_K_samps,
                              binary_frac_phase_ecl_samps,
                             ],
                             names=('binary_index',
                                    'star1_mass_init', 'star1_mass',
                                    'star1_rad', 'star1_den',
                                    'star1_lum', 'star1_teff',
                                    'star1_logg',
                                    'star2_mass_init', 'star2_mass',
                                    'star2_rad', 'star2_den',
                                    'star2_lum', 'star2_teff',
                                    'star2_logg',
                                    'binary_period',
                                    'binary_q', 'binary_q_init',
                                    'binary_ecc', 'binary_inc',
                                    'binary_sma', 'binary_mass_func',
                                    'binary_rv_K',
                                    'binary_frac_phase_ecl',
                                   ),
                            )
        
        params_table.write('stellar_binary_params.h5', path='data',
                           serialize_meta=True, compression=True,
                           overwrite=True)
        
        params_table.write('stellar_binary_params.txt',
                           format='ascii.fixed_width', overwrite=True)
        
        return
