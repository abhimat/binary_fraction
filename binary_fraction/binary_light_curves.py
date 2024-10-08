#!/usr/bin/env python

# Class to generate model binary light curves
# ---
# Abhimat Gautam

import os
import numpy as np
from binary_fraction import imf
from astropy.table import Table
from phitter import observables, filters
from phitter.params import star_params, binary_params, isoc_interp_params
from phitter.calc import model_obs_calc, phot_adj_calc
from phoebe import u
from phoebe import c as const
import parmap
from tqdm import tqdm

# Function to help with parallelization
def binary_light_curve_from_binary_row(
        cur_binary_row, bin_pop_lc_obj,
        out_dir,
        use_blackbody_atm=False,
    ):
    bin_pop_lc_obj.make_binary_light_curve(
        cur_binary_row['binary_index'],
        cur_binary_row['mass_1'], cur_binary_row['mass_2'],
        cur_binary_row['binary_period'], cur_binary_row['binary_t0_shift'],
        cur_binary_row['binary_q'], cur_binary_row['binary_ecc'],
        cur_binary_row['binary_inc'],
        use_blackbody_atm=use_blackbody_atm,
        out_dir=out_dir,
    )

# Default filters
kp_filt = filters.nirc2_kp_filt()
h_filt = filters.nirc2_h_filt()

# Class for generating light curves in a binary population
class binary_pop_light_curves(object):
    def  __init__(self):
        # Set up some defaults
        self.set_extLaw_alpha()
        self.set_pop_distance()
        
        return
    
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
                (lambda_Ks / filt_lambdas[filt_index])**self.ext_alpha
        
    
    def save_obs_times(self, obs_times_Kp, obs_times_H):
        self.obs_times_Kp = obs_times_Kp
        self.obs_times_H = obs_times_H
        
        self.obs_times = np.concatenate(
            (obs_times_Kp, obs_times_H),
        )
        
        self.obs_filts = np.concatenate((
            np.full(len(obs_times_Kp), kp_filt),
            np.full(len(obs_times_H), h_filt),
        ),)
        
        self.obs_types = np.concatenate((
            np.full(len(obs_times_Kp), 'phot'),
            np.full(len(obs_times_H), 'phot'),
        ),)
        
        
        # Set up a phitter model observables object,
        # which only contains times and types of observations
        self.model_observables = observables.observables(
            obs_times=self.obs_times,
            obs_filts=self.obs_filts, obs_types=self.obs_types,
        )
    
    def set_extLaw_alpha(self, ext_alpha=2.30):
        self.ext_alpha = ext_alpha
    
    def set_population_extinctions(self, ext_Kp=2.70, ext_H_mod=0.0):
        # Filter properties
        lambda_Ks = 2.18e-6 * u.m
        dlambda_Ks = 0.35e-6 * u.m

        lambda_Kp = 2.124e-6 * u.m
        dlambda_Kp = 0.351e-6 * u.m

        lambda_H = 1.633e-6 * u.m
        dlambda_H = 0.296e-6 * u.m
        
        self.ext_Kp = ext_Kp
        self.ext_H = ext_Kp * (lambda_Kp / lambda_H)**self.ext_alpha + ext_H_mod
        
    
    def set_pop_distance(self, pop_distance=7.971e3):
        self.pop_distance = pop_distance
    
    def make_binary_light_curve(
            self, binary_index, mass_1, mass_2,
            binary_period, binary_t0_shift,
            binary_q, binary_ecc, binary_inc,
            use_blackbody_atm=False,
            out_dir='./mock_binaries',
            num_phase_points=100,
            skip_txt_out=True,
        ):
        # Make sure output directory exists
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        if os.path.exists(
                '{0}/binary_{1}_mags_H.h5'.format(
                    out_dir + '/model_light_curves', int(binary_index))):
            return
        
        # print(binary_index)

        # Interpolate stellar parameters from isochrone
        star1_params = self.pop_isochrone.interp_star_params_mass_init(mass_1)
        star2_params = self.pop_isochrone.interp_star_params_mass_init(mass_2)
        
        # Set up binary parameters
        bin_params = binary_params.binary_params(
            period = binary_period * u.d,
            ecc = binary_ecc,
            inc = binary_inc * u.deg,
            t0 = (np.min(self.obs_times_Kp) - binary_period + binary_t0_shift),
        )
        # Pick a random argument of periapse
        bin_params.arg_per0 = np.random.uniform(
            low=0.0, high=360.0,
        ) * u.deg
        
        # Set up model times
        model_phases_Kp = np.linspace(0.0, 1.0,
                                      num=num_phase_points,
                                      endpoint=False)
        model_phases_H = np.linspace(0.0, 1.0,
                                     num=num_phase_points,
                                     endpoint=False)
        
        model_times_Kp = (model_phases_Kp * binary_period)
        model_times_H = (model_phases_H * binary_period)
        
        model_times = np.concatenate(
            (model_times_Kp, model_times_H),
        )
        
        obs_filts = np.concatenate((
            np.full(len(model_times_Kp), kp_filt),
            np.full(len(model_times_H), h_filt),
        ),)
        
        obs_types = np.concatenate((
            np.full(len(model_times_Kp), 'phot'),
            np.full(len(model_times_H), 'phot'),
        ),)
        
        model_observables = observables.observables(
            obs_times=model_times,
            obs_filts=obs_filts, obs_types=obs_types,
        )
        
        # Set up a binary model object
        binary_model_obj = model_obs_calc.binary_star_model_obs(
            model_observables,
            use_blackbody_atm=use_blackbody_atm,
            print_diagnostics=False,
        )
        
        # Obtain model magnitudes
        model_success = True
        
        binary_model_mags_Kp = np.empty(num_phase_points)
        binary_model_mags_H = np.empty(num_phase_points)
        
        num_triangles = 500
        modeled_observables = binary_model_obj.compute_obs(
            star1_params, star2_params, bin_params,
            num_triangles=num_triangles,
        )
        
        # Add distance modulus
        modeled_observables = phot_adj_calc.apply_distance_modulus(
            modeled_observables,
            self.isoc_dist*u.pc,
        )
        
        # Apply reddening from extinction
        modeled_observables = phot_adj_calc.apply_extinction(
            modeled_observables,
            isoc_Ks_ext=isoc_ext,
            ref_filt=kp_filt,
            target_ref_filt_ext=self.ext_Kp,
            isoc_red_law='NL18',
            ext_alpha=self.ext_alpha,
        )
        
        if np.isnan(modeled_observables.obs_times[0]):
            model_success = False
            (binary_model_mags_Kp, binary_model_mags_H) = ([-1], [-1])
        else:
            (binary_model_mags_Kp, binary_model_mags_H) = binary_model_out
            
        
        # Save out binary light curve
        ## Make sure output directory exists
        if not os.path.exists(out_dir + '/model_light_curves'):
            os.makedirs(out_dir + '/model_light_curves')
        
        ## Save out model magnitudes
        if model_success:
            binary_mags_Kp_table = Table(
                [model_phases_Kp, model_times_Kp, binary_model_mags_Kp],
                names=('model_phases_Kp', 'model_times_Kp', 'mags_Kp'),
            )
            binary_mags_H_table = Table(
                [model_phases_H, model_times_H, binary_model_mags_H],
                names=('model_phases_H', 'model_times_H', 'mags_H'),
            )
            
            binary_mags_Kp_table['model_phases_Kp'].info.format = '.2f'
            binary_mags_H_table['model_phases_H'].info.format = '.2f'
            
            binary_mags_Kp_table['model_times_Kp'].info.format = '.3f'
            binary_mags_H_table['model_times_H'].info.format = '.3f'
            
            binary_mags_Kp_table['mags_Kp'].info.format = '.6f'
            binary_mags_H_table['mags_H'].info.format = '.6f'
            
            if not skip_txt_out:
                binary_mags_Kp_table.write(
                    '{0}/binary_{1}_mags_Kp.txt'.format(
                        out_dir + '/model_light_curves', int(binary_index)),
                    overwrite=True, format='ascii.fixed_width',
                )
                
                binary_mags_H_table.write(
                    '{0}/binary_{1}_mags_H.txt'.format(
                        out_dir + '/model_light_curves', int(binary_index)),
                    overwrite=True, format='ascii.fixed_width',
                )
            
            binary_mags_Kp_table.write(
                '{0}/binary_{1}_mags_Kp.h5'.format(
                    out_dir + '/model_light_curves', int(binary_index)),
                path='data', serialize_meta=True, compression=True,
                overwrite=True,
            )
            
            binary_mags_H_table.write(
                '{0}/binary_{1}_mags_H.h5'.format(
                    out_dir + '/model_light_curves', int(binary_index)),
                path='data', serialize_meta=True, compression=True,
                overwrite=True,
            )
            
        else:
            binary_mags_Kp_table = Table(
                [[-1], [-1], binary_model_mags_Kp],
                names=('model_phases_Kp', 'model_times_Kp', 'mags_Kp'),
            )
            binary_mags_H_table = Table(
                [[-1], [-1], binary_model_mags_H],
                names=('model_phases_H', 'model_times_H', 'mags_H'),
            )
            
            if not skip_txt_out:
                binary_mags_Kp_table.write(
                    '{0}/binary_{1}_mags_Kp.txt'.format(
                        out_dir + '/model_light_curves', int(binary_index)),
                    overwrite=True, format='ascii.fixed_width',
                )
                
                binary_mags_H_table.write(
                    '{0}/binary_{1}_mags_H.txt'.format(
                        out_dir + '/model_light_curves', int(binary_index)),
                    overwrite=True, format='ascii.fixed_width',
                )
            
            binary_mags_Kp_table.write(
                '{0}/binary_{1}_mags_Kp.h5'.format(
                    out_dir + '/model_light_curves', int(binary_index)),
                path='data', serialize_meta=True, compression=True,
                overwrite=True,
            )
            
            binary_mags_H_table.write(
                '{0}/binary_{1}_mags_H.h5'.format(
                    out_dir + '/model_light_curves', int(binary_index)),
                path='data', serialize_meta=True, compression=True,
                overwrite=True,
            )
        
        ## Return model magnitudes
        return (binary_model_mags_Kp, binary_model_mags_H)
    
    def make_binary_population_light_curves(
            self, binary_pop_params_file,
            use_blackbody_atm=False,
            out_dir='./mock_binaries',
            parallelize=True,
            par_processes=32, par_chunksize=10):
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
        
        ## Generate light curves for all binaries
        parmap.map(
            binary_light_curve_from_binary_row, binary_pop_params_table,
            self, use_blackbody_atm=use_blackbody_atm, out_dir=out_dir,
            pm_pbar=True, pm_parallel=parallelize,
            pm_processes=par_processes,
            pm_chunksize=par_chunksize,
        )
        
        return
