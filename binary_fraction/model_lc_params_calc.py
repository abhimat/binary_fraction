# Class to calculate statistics for mock binary model light curves
# ---
# Abhimat Gautam

import os
import numpy as np
from astropy.table import Table
import parmap
from tqdm import tqdm

# Function to help with parallelization
def params_from_bin_index(binary_index, model_lc_params_calc_obj):
    return model_lc_params_calc_obj.calc_lc_params(binary_index)

class model_lc_params_calc(object):
    def  __init__(self, model_lcs_dir='./model_light_curves/'):
        # Set up any defaults
        self.model_lcs_dir = model_lcs_dir
        return
    
    def calc_lc_params(
            self, binary_index):
        """
        Calculate the light curve stats for the current mock binary system
        returns(Kp mean mag, )
        
        Parameters
        ----------
        binary_index : int
            The integer index of the binary system to calculate the stats for
        
        Returns
        -------
        peak_mag_kp : float
            Kp mag at peak brightness
        dip_mag_kp : float
            Kp mag at deepest dip
        dip_phase_kp : float
            Phase at deepest Kp mag dip
        delta_mag_kp : float
            Size of the variation in Kp mag
        mean_mag_kp : float
            Mean Kp magnitude
        med_mag_kp : float
            Median Kp magnitude
        peak_mag_h : float
            H mag at peak brightness
        dip_mag_h : float
            H mag at deepest dip
        dip_phase_kp : float
            Phase at deepest H mag dip
        delta_mag_h : float
            Size of the variation in H mag
        mean_mag_h : float
            Mean H magnitude
        med_mag_h : float
            Median H magnitude
        """
        kp_table_file = f'{self.model_lcs_dir}/binary_{binary_index}_mags_Kp.h5'
        h_table_file = f'{self.model_lcs_dir}/binary_{binary_index}_mags_H.h5'
        
        fail_out_tuple = (np.nan, np.nan, np.nan,
                          np.nan, np.nan, np.nan,
                          np.nan, np.nan, np.nan,
                          np.nan, np.nan, np.nan,
                         )
        
        # Check if table file exists
        if not os.path.exists(kp_table_file):
            return fail_out_tuple
        
        model_obs_mags_kp_table = Table.read(
            kp_table_file, path='data')
        model_obs_mags_h_table = Table.read(
            h_table_file, path='data')
        
        # Check if table has failed run output
        if model_obs_mags_kp_table['model_phases_Kp'][0] == -1:
            return fail_out_tuple
        
        # Calculate Kp-band statistics
        peak_mag_kp = np.min(model_obs_mags_kp_table['mags_Kp'])
        
        dip_mag_kp = np.max(model_obs_mags_kp_table['mags_Kp'])
        dip_ind_kp = np.argmax(model_obs_mags_kp_table['mags_Kp'])
        dip_phase_kp = model_obs_mags_kp_table['model_phases_Kp'][dip_ind_kp]
        
        delta_mag_kp = dip_mag_kp - peak_mag_kp
        
        mean_mag_kp = np.mean(model_obs_mags_kp_table['mags_Kp'])
        med_mag_kp = np.median(model_obs_mags_kp_table['mags_Kp'])
        
        # Calculate H-band statistics
        peak_mag_h = np.min(model_obs_mags_h_table['mags_H'])
        
        dip_mag_h = np.max(model_obs_mags_h_table['mags_H'])
        dip_ind_h = np.argmax(model_obs_mags_h_table['mags_H'])
        dip_phase_h = model_obs_mags_h_table['model_phases_H'][dip_ind_h]
        
        delta_mag_h = dip_mag_h - peak_mag_h
        
        mean_mag_h = np.mean(model_obs_mags_h_table['mags_H'])
        med_mag_h = np.median(model_obs_mags_h_table['mags_H'])
        
        # Construct output
        
        output_tuple = (peak_mag_kp, dip_mag_kp, dip_phase_kp,
                        delta_mag_kp, mean_mag_kp, med_mag_kp,
                        peak_mag_h, dip_mag_h, dip_phase_h,
                        delta_mag_h, mean_mag_h, med_mag_h,
                       )
        
        return output_tuple
    
    def calc_population_params(self, binary_pop_params_file,
            parallelize=True):
        
        # Read in table of binary parameters
        if binary_pop_params_file.endswith('.h5'):
            binary_pop_params_table = Table.read(
                binary_pop_params_file,
                path='data',
            )
        elif binary_pop_params_file.endswith('.txt'):
            binary_pop_params_table = Table.read(
                binary_pop_params_file,
                format='ascii.fixed_width',
            )
        
        binary_indexes = binary_pop_params_table['binary_index']
        num_binaries = len(binary_indexes)
        
        # Generate stellar and binary parameters for all mock binaries
        out_params = np.array(parmap.map(
                        params_from_bin_index,
                        binary_indexes, self,
                        pm_pbar=True, pm_parallel=parallelize,
                        pm_chunksize=10,
                     ))
        
        # Empty arrays to store stellar parameters
        peak_mag_kp = np.empty(num_binaries)
        dip_mag_kp = np.empty(num_binaries)
        dip_phase_kp = np.empty(num_binaries)
        delta_mag_kp = np.empty(num_binaries)
        mean_mag_kp = np.empty(num_binaries)
        med_mag_kp = np.empty(num_binaries)
        
        peak_mag_h = np.empty(num_binaries)
        dip_mag_h = np.empty(num_binaries)
        dip_phase_h = np.empty(num_binaries)
        delta_mag_h = np.empty(num_binaries)
        mean_mag_h = np.empty(num_binaries)
        med_mag_h = np.empty(num_binaries)
        
        # Put parameter outputs into arrays
        for cur_bin_index in range(num_binaries):
            (peak_mag_kp[cur_bin_index],
             dip_mag_kp[cur_bin_index],
             dip_phase_kp[cur_bin_index],
             delta_mag_kp[cur_bin_index],
             mean_mag_kp[cur_bin_index],
             med_mag_kp[cur_bin_index],
             peak_mag_h[cur_bin_index],
             dip_mag_h[cur_bin_index],
             dip_phase_h[cur_bin_index],
             delta_mag_h[cur_bin_index],
             mean_mag_h[cur_bin_index],
             med_mag_h[cur_bin_index],
            ) = out_params[cur_bin_index]
        
        # Make parameter table for output
        params_table = Table(
            [np.arange(num_binaries, dtype=int),
             peak_mag_kp, dip_mag_kp, dip_phase_kp,
             delta_mag_kp, mean_mag_kp, med_mag_kp, 
             peak_mag_h, dip_mag_h, dip_phase_h,
             delta_mag_h, mean_mag_h, med_mag_h,
            ],
            names=('binary_index',
                   'peak_mag_kp', 'dip_mag_kp', 'dip_phase_kp',
                   'delta_mag_kp', 'mean_mag_kp', 'med_mag_kp',
                   'peak_mag_h', 'dip_mag_h', 'dip_phase_h',
                   'delta_mag_h', 'mean_mag_h', 'med_mag_h',
                  ),
        )
        
        params_table.write('binary_model_lc_params.h5', path='data',
                           serialize_meta=True, compression=True,
                           overwrite=True)
        
        params_table.write('binary_model_lc_params.txt',
                           format='ascii.fixed_width', overwrite=True)
        
        return
