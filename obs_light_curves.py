#!/usr/bin/env python

# Class to generate observed binary light curves from model light curves
# ---
# Abhimat Gautam

import os
import numpy as np
from scipy import stats
from gc_photdata import align_dataset

class binary_pop_obs_light_curves(object):
    """Class for generating observed light curves from model light curves
    """
    
    def __init__(self):
        # Set up defaults
        
        return
    
    def read_align_data(self, align_kp_name, align_h_name,
            align_data_location = '/g/ghez/abhimat/datasets/align_data_py3/'):
        """Set and read the photometry align data being used for
        determining phot uncertainty to add to observation data points
        """
        self.align_kp_name = align_kp_name
        self.align_h_name = align_h_name
        
        align_kp_pickle_loc = f'{align_data_location}' \
                              + f'alignPickle_{self.align_kp_name}.pkl'
        align_h_pickle_loc = f'{align_data_location}' \
                              + f'alignPickle_{self.align_h_name}.pkl'
        
        self.align_data_kp = align_dataset.align_dataset(align_kp_pickle_loc)
        self.align_data_h = align_dataset.align_dataset(align_h_pickle_loc)
        
        return
    
    def construct_align_mag_arrays(self):
        """Construct the mag arrays needed for subsequent calculation
        """
        # Read in data from Kp-band align dataset
        self.star_names_kp = align_data_kp.star_names
        self.epoch_dates_kp = align_data_kp.epoch_dates
        self.epoch_MJDs_kp = align_data_kp.epoch_MJDs
        
        self.num_stars_kp = len(self.star_names_kp)
        self.num_epochs_kp = len(self.epoch_MJDs_kp)
        
        self.mags_kp = align_data_kp.star_mags_neighCorr
        self.mag_uncs_kp = align_data_kp.star_magErrors_neighCorr
        self.mag_means_kp = align_data_kp.star_magMeans_neighCorr
        
        # Construct empty arrays for storing mags and mag uncertainties
        self.mags_array_kp = np.empty((self.num_stars_kp,
                                       self.num_epochs_kp))
        self.mag_uncs_array_kp = np.empty((self.num_stars_kp,
                                       self.num_epochs_kp))
        
        # Fill in mag and mag uncertainty arrays
        for (star_id, star_name) in enumerate(self.star_names_kp):
            self.mags_array_kp[star_id] = self.mags_kp[star_name]
            self.mag_uncs_array_kp[star_id] = self.mag_uncs_kp[star_name]
        
        
        # Read in data from H-band align dataset
        self.star_names_h = align_data_h.star_names
        self.epoch_dates_h = align_data_h.epoch_dates
        self.epoch_MJDs_h = align_data_h.epoch_MJDs
        
        self.num_stars_h = len(self.star_names_h)
        self.num_epochs_h = len(self.epoch_MJDs_h)
        
        self.mags_h = align_data_h.star_mags_neighCorr
        self.mag_uncs_h = align_data_h.star_magErrors_neighCorr
        self.mag_means_h = align_data_h.star_magMeans_neighCorr
        
        # Construct empty arrays for storing mags and mag uncertainties
        self.mags_array_h = np.empty((self.num_stars_h,
                                       self.num_epochs_h))
        self.mag_uncs_array_h = np.empty((self.num_stars_h,
                                       self.num_epochs_h))
        
        # Fill in mag and mag uncertainty arrays
        for (star_id, star_name) in enumerate(self.star_names_h):
            self.mags_array_h[star_id] = self.mags_h[star_name]
            self.mag_uncs_array_h[star_id] = self.mag_uncs_h[star_name]
        
        
    
    def calc_epochs_mag_uncs(self):
        """Calculate the magnitude uncertainties in bins of magnitude
        for each epoch in the aligns used
        """
        # Go through the Kp align
        num_mag_bins = 20
        mag_cents_kp = np.linspace(9, 22, num=num_mag_bins)
        
        # Go through each epoch
        for (epoch_id, epoch_mjd) in enumerate(self.epoch_MJDs_kp):
            # Extract mags and mag uncs in this particular epoch
            cur_epoch_mags = self.mags_array_kp[:, epoch_id]
            cur_epoch_mag_uncs = self.mag_uncs_array_kp[:, epoch_id]
            
            cur_epoch_mag_unc_meds = np.empty(num_mag_bins)
            cur_epoch_mag_unc_mads = np.empty(num_mag_bins)
            cur_epoch_mag_stars_det = np.empty(num_mag_bins, dtype=np.int8)
            cur_epoch_mag_stars_tot = np.empty(num_mag_bins, dtype=np.int8)
            
            for (bin_id, cur_mag_cent) in enumerate(mag_cents_kp):
                # Find all stars within +/- 0.5 mag of the current mag_cent
                mag_filt = np.logical_and(
                               np.where(cur_epoch_mags >= cur_mag_cent - 0.5),
                               np.where(cur_epoch_mags <= cur_mag_cent - 0.5),
                           )
                
                total_filt = np.logical_and(
                                 np.where(self.mag_means_kp >= cur_mag_cent - 0.5),
                                 np.where(self.mag_means_kp <= cur_mag_cent - 0.5),
                           )
                total_mag_mean_stars = (self.mag_means_kp)[total_filt]
                
                filt_mags = cur_epoch_mags[mag_filt]
                filt_mag_uncs = cur_epoch_mag_uncs[mag_filt]
                
                # Calculate stats and save out in total arrays
                cur_epoch_mag_unc_meds[bin_id] = np.median(filt_mag_uncs)
                cur_epoch_mag_unc_mads[bin_id] = stats.median_abs_deviation(
                                                     filt_mag_uncs)
                cur_epoch_mag_stars_det[bin_id] = len(filt_mags)
                cur_epoch_mag_stars_tot[bin_id] = len(total_mag_mean_stars)