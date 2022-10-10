#!/usr/bin/env python

# Class to generate observed binary light curves from model light curves
# ---
# Abhimat Gautam

from gc_photdata import align_dataset
import os
import numpy as np
from scipy import stats
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import warnings

class calc_obs_uncs(object):
    """Class for calculating uncertainties in each observation epoch
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
        self.star_names_kp = self.align_data_kp.star_names
        self.epoch_dates_kp = self.align_data_kp.epoch_dates
        self.epoch_MJDs_kp = self.align_data_kp.epoch_MJDs
        
        self.num_stars_kp = len(self.star_names_kp)
        self.num_epochs_kp = len(self.epoch_MJDs_kp)
        
        self.mags_kp = self.align_data_kp.star_mags_neighCorr
        self.mag_uncs_kp = self.align_data_kp.star_magErrors_neighCorr
        self.mag_means_kp = self.align_data_kp.star_magMeans_neighCorr
        
        # Construct empty arrays for storing mags and mag uncertainties
        self.mags_array_kp = np.empty((self.num_stars_kp,
                                       self.num_epochs_kp))
        self.mag_uncs_array_kp = np.empty((self.num_stars_kp,
                                           self.num_epochs_kp))
        self.mag_means_array_kp = np.empty((self.num_stars_kp))
        
        # Fill in mag and mag uncertainty arrays
        for (star_id, star_name) in enumerate(self.star_names_kp):
            self.mags_array_kp[star_id] = self.mags_kp[star_name]
            self.mag_uncs_array_kp[star_id] = self.mag_uncs_kp[star_name]
            self.mag_means_array_kp[star_id] = self.mag_means_kp[star_name]
        
        
        # Read in data from H-band align dataset
        self.star_names_h = self.align_data_h.star_names
        self.epoch_dates_h = self.align_data_h.epoch_dates
        self.epoch_MJDs_h = self.align_data_h.epoch_MJDs
        
        self.num_stars_h = len(self.star_names_h)
        self.num_epochs_h = len(self.epoch_MJDs_h)
        
        self.mags_h = self.align_data_h.star_mags_neighCorr
        self.mag_uncs_h = self.align_data_h.star_magErrors_neighCorr
        self.mag_means_h = self.align_data_h.star_magMeans_neighCorr
        
        # Construct empty arrays for storing mags and mag uncertainties
        self.mags_array_h = np.empty((self.num_stars_h,
                                       self.num_epochs_h))
        self.mag_uncs_array_h = np.empty((self.num_stars_h,
                                       self.num_epochs_h))
        self.mag_means_array_h = np.empty((self.num_stars_h))
        
        # Fill in mag and mag uncertainty arrays
        for (star_id, star_name) in enumerate(self.star_names_h):
            self.mags_array_h[star_id] = self.mags_h[star_name]
            self.mag_uncs_array_h[star_id] = self.mag_uncs_h[star_name]
            self.mag_means_array_h[star_id] = self.mag_means_h[star_name]
        
        
    
    def calc_epochs_mag_uncs(self):
        """Calculate the magnitude uncertainties in bins of magnitude
        for each epoch in the aligns used
        """
        # Make sure output directories exist
        out_dir = './obs_uncertainties'
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(out_dir + '/Kp'):
            os.makedirs(out_dir + '/Kp')
        if not os.path.exists(out_dir + '/H'):
            os.makedirs(out_dir + '/H')
        
        # Suppress empty slice and invalid value warnings
        warnings.filterwarnings('ignore', '.*Mean of empty slice.*', )
        warnings.filterwarnings('ignore', '.*invalid value encountered.*', )
        
        # Go through the Kp align
        mag_min = 9
        mag_max = 22
        num_mag_bins = (mag_max - mag_min) * 5
        mag_cents_kp = np.linspace(mag_min, mag_max, num=num_mag_bins)
        
        # Go through each epoch
        for (epoch_id, epoch_mjd) in tqdm(enumerate(self.epoch_MJDs_kp),
                                          total=self.num_epochs_kp):
            # Extract mags and mag uncs in this particular epoch
            cur_epoch_mags = self.mags_array_kp[:, epoch_id]
            cur_epoch_mag_uncs = self.mag_uncs_array_kp[:, epoch_id]
            
            cur_epoch_mag_unc_meds = np.empty(num_mag_bins)
            cur_epoch_mag_unc_mads = np.empty(num_mag_bins)
            cur_epoch_mag_stars_det = np.empty(num_mag_bins, dtype=np.int32)
            cur_epoch_mag_stars_tot = np.empty(num_mag_bins, dtype=np.int32)
            
            for (bin_id, cur_mag_cent) in enumerate(mag_cents_kp):
                # Find all stars within +/- 0.5 mag of the current mag_cent
                mag_filt = np.where(np.logical_and(
                               cur_epoch_mags >= (cur_mag_cent - 0.5),
                               cur_epoch_mags <= (cur_mag_cent + 0.5),
                           ))
                
                total_filt = np.where(np.logical_and(
                                 self.mag_means_array_kp >= (cur_mag_cent - 0.5),
                                 self.mag_means_array_kp <= (cur_mag_cent + 0.5),
                             ))
                total_mag_mean_stars = (self.mag_means_array_kp)[total_filt]
                
                filt_mags = cur_epoch_mags[mag_filt]
                filt_mag_uncs = cur_epoch_mag_uncs[mag_filt]
                
                # Calculate stats and save out in total arrays
                cur_epoch_mag_unc_meds[bin_id] = np.median(filt_mag_uncs)
                cur_epoch_mag_unc_mads[bin_id] = stats.median_abs_deviation(
                                                     filt_mag_uncs)
                cur_epoch_mag_stars_det[bin_id] = len(filt_mags)
                cur_epoch_mag_stars_tot[bin_id] = len(total_mag_mean_stars)
            
            # Create the outputs for the epoch
            
            # Create the output table for the epoch
            epoch_MJD_col = np.repeat([epoch_mjd], len(mag_cents_kp))
            
            epoch_table = Table({
                'mjd': epoch_MJD_col,
                'mag': mag_cents_kp,
                'mag_unc_med': cur_epoch_mag_unc_meds,
                'mag_unc_mad': cur_epoch_mag_unc_mads,
                'stars_det_epoch': cur_epoch_mag_stars_det,
                'stars_det_tot': cur_epoch_mag_stars_tot,
            })
            
            epoch_table.write(f'{out_dir}/Kp/epoch_{epoch_id}.txt',
                              overwrite=True, format='ascii.fixed_width')
            
            # Create a mag, mag uncertainty plot for the epoch
            # plt.style.use(['tex_paper', 'ticks_innie'])
            plt.style.use(['ticks_innie'])
            
            fig, ax = plt.subplots(figsize=(8,4), tight_layout=True)
            
            ax.plot(mag_cents_kp, cur_epoch_mag_unc_meds, 'k-', lw=1.5)
            
            ax.fill_between(
                mag_cents_kp,
                cur_epoch_mag_unc_meds + cur_epoch_mag_unc_mads,
                cur_epoch_mag_unc_meds - cur_epoch_mag_unc_mads,
                color='k', alpha=0.4,
            )
            
            ax.set_xlabel(r"$m_{K'}$")
            ax.set_ylabel(r"$\sigma_{m_{K'}}$")
            
            ax.text(9.5, 0.45,
                    r"$K'$" + f': epoch {epoch_id}, MJD = {epoch_mjd:.3f}',
                    ha='left', va='center',
                   )
            
            ax.set_xlim([np.min(mag_cents_kp), np.max(mag_cents_kp)])
            ax.set_ylim([0.0, 0.5])
            
            x_majorLocator = MultipleLocator(1)
            x_minorLocator = MultipleLocator(0.2)
            ax.xaxis.set_major_locator(x_majorLocator)
            ax.xaxis.set_minor_locator(x_minorLocator)
            
            y_majorLocator = MultipleLocator(0.1)
            y_minorLocator = MultipleLocator(0.02)
            ax.yaxis.set_major_locator(y_majorLocator)
            ax.yaxis.set_minor_locator(y_minorLocator)
            
            fig.savefig(f'{out_dir}/Kp/epoch_{epoch_id}.pdf')
            fig.savefig(f'{out_dir}/Kp/epoch_{epoch_id}.png')
            plt.close(fig)
            
        
        # Go through the H align
        mag_min = 11
        mag_max = 24
        num_mag_bins = (mag_max - mag_min) * 5
        mag_cents_h = np.linspace(mag_min, mag_max, num=num_mag_bins)
        
        # Go through each epoch
        for (epoch_id, epoch_mjd) in tqdm(enumerate(self.epoch_MJDs_h),
                                          total=self.num_epochs_h):
            # Extract mags and mag uncs in this particular epoch
            cur_epoch_mags = self.mags_array_h[:, epoch_id]
            cur_epoch_mag_uncs = self.mag_uncs_array_h[:, epoch_id]
            
            cur_epoch_mag_unc_meds = np.empty(num_mag_bins)
            cur_epoch_mag_unc_mads = np.empty(num_mag_bins)
            cur_epoch_mag_stars_det = np.empty(num_mag_bins, dtype=np.int32)
            cur_epoch_mag_stars_tot = np.empty(num_mag_bins, dtype=np.int32)
            
            for (bin_id, cur_mag_cent) in enumerate(mag_cents_h):
                # Find all stars within +/- 0.5 mag of the current mag_cent
                mag_filt = np.where(np.logical_and(
                               cur_epoch_mags >= (cur_mag_cent - 0.5),
                               cur_epoch_mags <= (cur_mag_cent + 0.5),
                           ))
                
                total_filt = np.where(np.logical_and(
                                 self.mag_means_array_h >= (cur_mag_cent - 0.5),
                                 self.mag_means_array_h <= (cur_mag_cent + 0.5),
                             ))
                total_mag_mean_stars = (self.mag_means_array_h)[total_filt]
                
                filt_mags = cur_epoch_mags[mag_filt]
                filt_mag_uncs = cur_epoch_mag_uncs[mag_filt]
                
                # Calculate stats and save out in total arrays
                cur_epoch_mag_unc_meds[bin_id] = np.median(filt_mag_uncs)
                cur_epoch_mag_unc_mads[bin_id] = stats.median_abs_deviation(
                                                     filt_mag_uncs)
                cur_epoch_mag_stars_det[bin_id] = len(filt_mags)
                cur_epoch_mag_stars_tot[bin_id] = len(total_mag_mean_stars)
            
            # Create the outputs for the epoch
            
            # Create the output table for the epoch
            epoch_MJD_col = np.repeat([epoch_mjd], len(mag_cents_h))
            
            epoch_table = Table({
                'mjd': epoch_MJD_col,
                'mag': mag_cents_h,
                'mag_unc_med': cur_epoch_mag_unc_meds,
                'mag_unc_mad': cur_epoch_mag_unc_mads,
                'stars_det_epoch': cur_epoch_mag_stars_det,
                'stars_det_tot': cur_epoch_mag_stars_tot,
            })
            
            epoch_table.write(f'{out_dir}/H/epoch_{epoch_id}.txt',
                              overwrite=True, format='ascii.fixed_width')
            
            # Create a mag, mag uncertainty plot for the epoch
            # plt.style.use(['tex_paper', 'ticks_innie'])
            plt.style.use(['ticks_innie'])
            
            fig, ax = plt.subplots(figsize=(8,4), tight_layout=True)
            
            ax.plot(mag_cents_h, cur_epoch_mag_unc_meds, 'k-', lw=1.5)
            
            ax.fill_between(
                mag_cents_h,
                cur_epoch_mag_unc_meds + cur_epoch_mag_unc_mads,
                cur_epoch_mag_unc_meds - cur_epoch_mag_unc_mads,
                color='k', alpha=0.4,
            )
            
            ax.set_xlabel(r"$m_{H}$")
            ax.set_ylabel(r"$\sigma_{m_{H}}$")
            
            ax.text(11.5, 0.45,
                    r"$H$" + f': epoch {epoch_id}, MJD = {epoch_mjd:.3f}',
                    ha='left', va='center',
                   )
            
            ax.set_xlim([np.min(mag_cents_h), np.max(mag_cents_h)])
            ax.set_ylim([0.0, 0.5])
            
            x_majorLocator = MultipleLocator(1)
            x_minorLocator = MultipleLocator(0.2)
            ax.xaxis.set_major_locator(x_majorLocator)
            ax.xaxis.set_minor_locator(x_minorLocator)
            
            y_majorLocator = MultipleLocator(0.1)
            y_minorLocator = MultipleLocator(0.02)
            ax.yaxis.set_major_locator(y_majorLocator)
            ax.yaxis.set_minor_locator(y_minorLocator)
            
            fig.savefig(f'{out_dir}/H/epoch_{epoch_id}.pdf')
            fig.savefig(f'{out_dir}/H/epoch_{epoch_id}.png')
            plt.close(fig)
        

class add_obs_uncs(object):
    """Class for adding obs uncertainties from each observation epoch
    to binary star model mags
    """
    
    def __init__(self):
        # Set up defaults
        
        return
    
    def read_epoch_uncs(self,
            num_epochs_kp, num_epochs_h,
            uncs_dir = './obs_uncertainties/',
        ):
        """Read in and store mag uncertainties for each epoch
        """
        
        epoch_unc_tables_kp = {}
        epoch_unc_tables_h = {}
        
        # Read in each epoch's uncertainties table
        for cur_epoch_index in range(num_epochs_kp):
            cur_epoch_table = Table.read(
                f'{uncs_dir}/Kp/epoch_{cur_epoch_index}.txt',
                format='ascii.fixed_width')
            
            epoch_unc_tables_kp[cur_epoch_index] = cur_epoch_table
        
        for cur_epoch_index in range(num_epochs_h):
            cur_epoch_table = Table.read(
                f'{uncs_dir}/H/epoch_{cur_epoch_index}.txt',
                format='ascii.fixed_width')
            
            epoch_unc_tables_h[cur_epoch_index] = cur_epoch_table
        
        # Save out variables into object variables
        self.epoch_unc_tables_kp = epoch_unc_tables_kp
        self.epoch_unc_tables_h = epoch_unc_tables_h
        
        return
    
    def apply_mag_uncs(self,
            model_number,
            model_lcs_dir = './mock_binaries/model_light_curves/',
            model_obs_lcs_dir = './mock_binaries/model_obs_light_curves/',
        ):
        """Apply obs uncertainties to model magnitudes
        """
        
        # Work on Kp epochs
        # Read in model magnitudes
        model_mags_kp_table = Table.read(
            f'{model_lcs_dir}/binary_{model_number}_mags_Kp.txt',
            format='ascii.fixed_width')
        
        num_epochs_kp = len(model_mags_kp_table)
        
        model_mags_kp = model_mags_kp_table['mags_Kp']
        model_obs_mags_kp = np.empty(num_epochs_kp)
        model_obs_mag_uncs_kp = np.empty(num_epochs_kp)
        
        if num_epochs_kp == 1:
            return
        
        # Step through each observation epoch
        for cur_epoch_index in range(num_epochs_kp):
            cur_epoch_model_mag = model_mags_kp[cur_epoch_index]
            cur_epoch_unc_table = self.epoch_unc_tables_kp[cur_epoch_index]
            
            # Apply filter for mags where values are defined
            finite_filter = np.where(
                np.isfinite(cur_epoch_unc_table['mag_unc_med'])
            )
            cur_epoch_unc_table = cur_epoch_unc_table[finite_filter]
            
            
            interp_unc_med = np.interp(cur_epoch_model_mag,
                                       cur_epoch_unc_table['mag'],
                                       cur_epoch_unc_table['mag_unc_med'],
                                       right=np.nan,
                                      )
            
            interp_unc_mad = np.interp(cur_epoch_model_mag,
                                       cur_epoch_unc_table['mag'],
                                       cur_epoch_unc_table['mag_unc_mad'],
                                       right=np.nan,
                                      )
            
            # Pick a mag uncertainty            
            cur_epoch_mag_unc = -1
            # While loop to make sure negative unc doesn't get picked
            while cur_epoch_mag_unc < 0:
                cur_epoch_mag_unc = np.random.normal(
                                        loc=interp_unc_med,
                                        scale=interp_unc_mad)
            
            # Pick an observed mag based on the mag uncertainty
            cur_epoch_model_obs_mag = np.random.normal(
                                          loc=cur_epoch_model_mag,
                                          scale=cur_epoch_mag_unc)
            
            # Save out drawn values
            model_obs_mags_kp[cur_epoch_index] = cur_epoch_model_obs_mag
            model_obs_mag_uncs_kp[cur_epoch_index] = cur_epoch_mag_unc
        
        # Construct and output a new table for the observed model mags
        model_obs_mags_kp_table = Table({
            'MJD': model_mags_kp_table['MJD'],
            'mags_Kp': model_obs_mags_kp,
            'mag_uncs_Kp': model_obs_mag_uncs_kp,
        })
        
        model_obs_mags_kp_table.write(
            f'{model_obs_lcs_dir}/binary_{model_number}_mags_Kp.txt',
            overwrite=True, format='ascii.fixed_width')
        
        
        # Work on H epochs
        # Read in model magnitudes
        model_mags_h_table = Table.read(
            f'{model_lcs_dir}/binary_{model_number}_mags_H.txt',
            format='ascii.fixed_width')
        
        num_epochs_h = len(model_mags_h_table)
        
        model_mags_h = model_mags_h_table['mags_H']
        model_obs_mags_h = np.empty(num_epochs_h)
        model_obs_mag_uncs_h = np.empty(num_epochs_h)
        
        if num_epochs_h == 1:
            return
        
        # Step through each observation epoch
        for cur_epoch_index in range(num_epochs_h):
            cur_epoch_model_mag = model_mags_h[cur_epoch_index]
            cur_epoch_unc_table = self.epoch_unc_tables_h[cur_epoch_index]
            
            # Apply filter for mags where values are defined
            finite_filter = np.where(
                np.isfinite(cur_epoch_unc_table['mag_unc_med'])
            )
            cur_epoch_unc_table = cur_epoch_unc_table[finite_filter]
            
            
            interp_unc_med = np.interp(cur_epoch_model_mag,
                                       cur_epoch_unc_table['mag'],
                                       cur_epoch_unc_table['mag_unc_med'],
                                       right=np.nan,
                                      )
            
            interp_unc_mad = np.interp(cur_epoch_model_mag,
                                       cur_epoch_unc_table['mag'],
                                       cur_epoch_unc_table['mag_unc_mad'],
                                       right=np.nan,
                                      )
            
            # Pick a mag uncertainty            
            cur_epoch_mag_unc = -1
            # While loop to make sure negative unc doesn't get picked
            while cur_epoch_mag_unc < 0:
                cur_epoch_mag_unc = np.random.normal(
                                        loc=interp_unc_med,
                                        scale=interp_unc_mad)
            
            # Pick an observed mag based on the mag uncertainty
            cur_epoch_model_obs_mag = np.random.normal(
                                          loc=cur_epoch_model_mag,
                                          scale=cur_epoch_mag_unc)
            
            # Save out drawn values
            model_obs_mags_h[cur_epoch_index] = cur_epoch_model_obs_mag
            model_obs_mag_uncs_h[cur_epoch_index] = cur_epoch_mag_unc
        
        # Construct and output a new table for the observed model mags
        model_obs_mags_h_table = Table({
            'MJD': model_mags_h_table['MJD'],
            'mags_H': model_obs_mags_h,
            'mag_uncs_H': model_obs_mag_uncs_h,
        })
        
        model_obs_mags_h_table.write(
            f'{model_obs_lcs_dir}/binary_{model_number}_mags_H.txt',
            overwrite=True, format='ascii.fixed_width')
        
        return
    
    def apply_mag_uncs_range(self,
            model_number_range,
            model_lcs_dir = './mock_binaries/model_light_curves/',
            model_obs_lcs_dir = './mock_binaries/model_obs_light_curves/',
        ):
        
        # Make sure output directory exists
        if not os.path.exists(model_obs_lcs_dir):
            os.makedirs(model_obs_lcs_dir)
        
        # Run function on range
        for model_number in tqdm(model_number_range):
            self.apply_mag_uncs(
                model_number,
                model_lcs_dir = model_lcs_dir,
                model_obs_lcs_dir = model_obs_lcs_dir,
            )
        
        return
