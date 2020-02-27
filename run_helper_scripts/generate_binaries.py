#!/usr/bin/env python

# Script to generate binary systems
# ---
# Abhimat Gautam

import numpy as np

def generate_params():
    # Make binary population
    from binary_fraction import binary_parameters
    
    ## Make binary population with GC and Sana+ parameters
    bin_pop = binary_parameters.binary_population()
    
    bin_pop.make_imf(mass_limits=np.array([10, 100]), alpha=1.7)
    bin_pop.make_period_dist(period_limits=np.array([1., 10.**3.5]), pl_exp=-0.55)
    bin_pop.make_q_dist(q_limits=np.array([0.1, 1.]), pl_exp=-0.1)
    bin_pop.make_ecc_dist(ecc_limits=np.array([0., 1.]), pl_exp=-0.45)
    bin_pop.make_inc_dist()
    
    ## Generate table with 10,000 mock binary system parameters
    binary_parameters.generate_binary_population_params(bin_pop, 10000, out_dir='./mock_binaries')

def make_model_light_curves():
    # Make binary population
    from binary_fraction import binary_light_curves
    
    bin_pop_lc = binary_light_curves.binary_pop_light_curves()
    bin_pop_lc.make_pop_isochrone(
                   isoc_age=4.0e6, isoc_ext_Ks=2.54,
                   isoc_dist=7.971e3, isoc_phase=None,
                   isoc_met=0.0)
    
    # Read in observation times from align data
    from gc_photdata import align_dataset
    align_data_location = '/g/ghez/abhimat/datasets/align_data/'
    
    align_kp_name = 'phot_19_08_2_Kp'
    align_pickle_loc = '{0}alignPickle_{1}.pkl'.format(align_data_location, align_kp_name)
    align_data = align_dataset.align_dataset(align_pickle_loc)
    kp_epoch_MJDs = align_data.epoch_MJDs
    
    align_h_name = 'phot_19_07_2_H'
    align_pickle_loc = '{0}alignPickle_{1}.pkl'.format(align_data_location, align_h_name)
    align_data = align_dataset.align_dataset(align_pickle_loc)
    h_epoch_MJDs = align_data.epoch_MJDs
    
    # Set up observation times
    bin_pop_lc.save_obs_times(kp_epoch_MJDs, h_epoch_MJDs)
    
    # Make light curves
    bin_pop_lc.make_binary_population_light_curves('./mock_binaries/binary_pop_params.txt',
                                                   out_dir='./mock_binaries', parallelize=True)
    