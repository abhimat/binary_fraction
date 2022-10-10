#!/usr/bin/env python

# Class to calculate stellar and binary parameters for mock binary systems
# ---
# Abhimat Gautam

import os
import numpy as np
from binary_fraction import imf
from astropy.table import Table
from phoebe_phitter import isoc_interp, lc_calc
from phoebe import u
from phoebe import c as const
import parmap
from tqdm import tqdm

# Function to help with parallelization
def params_from_bin_index(binary_index, obs_params_calc_obj):
    return obs_params_calc_obj.calc_obs_params(int(binary_index))

class obs_params_calc(object):
    def  __init__(self, model_obs_lcs_dir='../model_obs_light_curves/'):
        # Set up any defaults
        self.model_obs_lcs_dir = model_obs_lcs_dir
        return
    
    def calc_obs_params(
            self, binary_index):
        """Calculate the observational parameters (i.e. light curve stats)
        for the current mock binary system
        """
        kp_table_file = f'{self.model_obs_lcs_dir}/binary_{binary_index}_mags_Kp.txt'
        h_table_file = f'{self.model_obs_lcs_dir}/binary_{binary_index}_mags_H.txt'
        
        # Check if table file exists
        if not os.path.exists(kp_table_file):
            # Mag generator code failed for this system
            # Pass fail values
            output_tuple = (np.nan, 0, np.nan, 0)
            
            return output_tuple
        
        model_obs_mags_kp_table = Table.read(
            kp_table_file,
            format='ascii.fixed_width')
        model_obs_mags_h_table = Table.read(
            h_table_file,
            format='ascii.fixed_width')
        
        mag_mean_kp = np.mean(model_obs_mags_kp_table['mags_Kp'])
        num_nights_kp = len(model_obs_mags_kp_table)
    
        mag_mean_h = np.mean(model_obs_mags_h_table['mags_H'])
        num_nights_h = len(model_obs_mags_h_table)
        
        output_tuple = (mag_mean_kp, num_nights_kp, mag_mean_h, num_nights_h)
        
        return output_tuple
    
    def calc_population_params(self, binary_pop_params_file,
            parallelize=True):
        
        # Read in table of binary parameters
        binary_pop_params_table = Table.read(
                                      binary_pop_params_file,
                                      format='ascii.fixed_width')
        
        binary_indexes = binary_pop_params_table['binary_index']
        num_binaries = len(binary_indexes)
        
        # Generate stellar and binary parameters for all mock binaries
        out_params = np.array(parmap.map(
                        params_from_bin_index,
                        binary_indexes, self,
                        pm_pbar=True, pm_parallel=parallelize
                     ))
        
        # Empty arrays to store stellar parameters
        mag_mean_kp = np.empty(num_binaries)
        num_nights_kp = np.empty(num_binaries)
        mag_mean_h = np.empty(num_binaries)
        num_nights_h = np.empty(num_binaries)

        # Put parameter outputs into arrays
        for cur_bin_index in range(num_binaries):
            (mag_mean_kp[cur_bin_index], num_nights_kp[cur_bin_index],
             mag_mean_h[cur_bin_index], num_nights_h[cur_bin_index], 
            ) = out_params[cur_bin_index]
        
        # Make parameter table for output
        params_table = Table([range(num_binaries),
                              mag_mean_kp,
                              num_nights_kp,
                              mag_mean_h,
                              num_nights_h,
                             ],
                             names=('binary_index',
                                    'mag_mean_kp',
                                    'num_nights_kp',
                                    'mag_mean_h',
                                    'num_nights_h',
                                   ),
                            )
        
        params_table.write('binary_obs_params.h5', path='data',
                           serialize_meta=True, compression=True,
                           overwrite=True)
        
        params_table.write('binary_obs_params.txt',
                           format='ascii.fixed_width', overwrite=True)
        
        return
