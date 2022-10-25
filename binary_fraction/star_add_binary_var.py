# Class to add mock binary light curves to observed light curves
# ---
# Abhimat Gautam

import numpy as np
from astropy.table import Table
from gc_photdata import align_dataset
import os

class star_add_binary_var(object):
    """
    Object for each star to add mock binary light curves to observed data
    """
    
    def __init__(self):
        # Set up defaults
        
        return
    
    def set_model_lcs_dir(self, model_lcs_dir):
        """
        Set model_lcs_dir variable for class, where model light curves for
        mock binaries are located
        
        Parameters
        ----------
        model_lcs_dir : str, default: './mock_binaries/model_light_curves/'
            Directory where the mock binary model light curves are stored
        """
        self.model_lcs_dir = model_lcs_dir
    
    def load_model_lc_params_table(
            self, table_file_path='./binary_model_lc_params.h5'):
        """
        Load the binary model light curve parameters table
        """
        # Load in the table
        model_lc_params_table = Table.read(
            table_file_path, path='data')
        
        model_lc_params_table.add_index(['binary_index'])
        
        # # Clean out stars that don't have any light curves
        # lc_filt = np.where(model_lc_params_table['med_mag_kp'] > 0)
        #
        # model_lc_params_table = model_lc_params_table[lc_filt]
        
        # Save out as a class variable
        self.model_lc_params_table = model_lc_params_table
        
        return
    
    def load_model_sb_params_table(
            self, table_file_path='./stellar_binary_params.h5'):
        """
        Load the binary model light curve parameters table
        """
        # Load in the table
        model_sb_params_table = Table.read(
            table_file_path, path='data')
        
        model_sb_params_table.add_index(['binary_index'])
        
        # Save out as a class variable
        self.model_sb_params_table = model_sb_params_table
        
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
        
        # Read in data from Kp-band align dataset
        self.star_names_kp = self.align_data_kp.star_names
        self.epoch_dates_kp = self.align_data_kp.epoch_dates
        self.epoch_MJDs_kp = self.align_data_kp.epoch_MJDs
        
        self.num_stars_kp = len(self.star_names_kp)
        self.num_epochs_kp = len(self.epoch_MJDs_kp)
        self.num_nights_kp = len(self.epoch_MJDs_kp)
        
        self.mags_kp = self.align_data_kp.star_mags_neighCorr
        self.mag_uncs_kp = self.align_data_kp.star_magErrors_neighCorr
        self.mag_means_kp = self.align_data_kp.star_magMeans_neighCorr
        
        
        # Read in data from H-band align dataset
        self.star_names_h = self.align_data_h.star_names
        self.epoch_dates_h = self.align_data_h.epoch_dates
        self.epoch_MJDs_h = self.align_data_h.epoch_MJDs
        
        self.num_stars_h = len(self.star_names_h)
        self.num_epochs_h = len(self.epoch_MJDs_h)
        self.num_nights_h = len(self.epoch_MJDs_h)
        
        self.mags_h = self.align_data_h.star_mags_neighCorr
        self.mag_uncs_h = self.align_data_h.star_magErrors_neighCorr
        self.mag_means_h = self.align_data_h.star_magMeans_neighCorr
        
        # Set experiment-wide t0 as floor of minimum observation date
        # Used when sampling mock binary light curves
        self.t0 = np.floor(np.min(
            [np.min(self.epoch_MJDs_kp), np.min(self.epoch_MJDs_h)]
        ))
        
        return
    
    def read_epoch_uncs(self,
            uncs_dir = './obs_uncertainties/',
        ):
        """Read in and store mag uncertainties for each epoch
        """
        
        epoch_unc_tables_kp = {}
        epoch_unc_tables_h = {}
        
        # Read in each epoch's uncertainties table
        for cur_epoch_index in range(self.num_nights_kp):
            cur_epoch_table = Table.read(
                f'{uncs_dir}/Kp/epoch_{cur_epoch_index}.txt',
                format='ascii.fixed_width'
            )
            
            epoch_unc_tables_kp[cur_epoch_index] = cur_epoch_table
        
        for cur_epoch_index in range(self.num_nights_h):
            cur_epoch_table = Table.read(
                f'{uncs_dir}/H/epoch_{cur_epoch_index}.txt',
                format='ascii.fixed_width'
            )
            
            epoch_unc_tables_h[cur_epoch_index] = cur_epoch_table
        
        # Save out variables into object variables
        self.epoch_unc_tables_kp = epoch_unc_tables_kp
        self.epoch_unc_tables_h = epoch_unc_tables_h
        
        return
    
    
    def draw_bin_mags(self, model_index, phase_shift,
            print_diagnostics=False,
        ):
        """
        For a model binary of index {model_index}, sample the model light curve
        at this experiment's observation dates.
        """
        # Retrieve relevant information about binary model
        bin_sb_row = self.model_sb_params_table.loc[model_index]
        
        binary_period = bin_sb_row['binary_period']
        
        # Read in the model
        model_mags_kp_table = Table.read(
            f'{self.model_lcs_dir}/binary_{model_index}_mags_Kp.h5',
            path='data')
        
        model_mags_h_table = Table.read(
            f'{self.model_lcs_dir}/binary_{model_index}_mags_H.h5',
            path='data')
        
        # Shift the model phases by the phase shift var
        sel_model_phases_kp = (model_mags_kp_table['model_phases_Kp'] +
                               phase_shift) % 1.0
        
        sel_model_phases_h = (model_mags_h_table['model_phases_H'] +
                              phase_shift) % 1.0
        
        if print_diagnostics:
            print(sel_model_phases_kp)
            print(model_mags_kp_table['model_phases_Kp'])
        
        # Sort the model tables so that the shifted phases are in order
        kp_argsort = np.argsort(sel_model_phases_kp)
        
        sel_model_phases_kp = sel_model_phases_kp[kp_argsort]
        model_mags_kp_table = model_mags_kp_table[kp_argsort]
        
        h_argsort = np.argsort(sel_model_phases_h)
        
        sel_model_phases_h = sel_model_phases_h[h_argsort]
        model_mags_h_table = model_mags_h_table[h_argsort]
        
        if print_diagnostics:
            print(sel_model_phases_kp)
            print(model_mags_kp_table['model_phases_Kp'])
            print(model_mags_kp_table['mags_Kp'])
        
        # Draw model mags at observation times
        
        # Calculate the observation times' phases
        obs_phases_kp = (self.epoch_MJDs_kp - self.t0) % binary_period
        obs_phases_h = (self.epoch_MJDs_h - self.t0) % binary_period
        
        # Interpolate the light curves at the observation's phases
        # from the model
        model_obs_mags_kp = np.interp(
            obs_phases_kp,                  # Experiment observation phases
            sel_model_phases_kp,            # Model phases (shifted)
            model_mags_kp_table['mags_Kp'], # Model mags (shifted)
        )
        
        model_obs_mags_h = np.interp(
            obs_phases_h,                   # Experiment observation phases
            sel_model_phases_h,             # Model phases (shifted)
            model_mags_h_table['mags_H'],   # Model mags (shifted)
        )
        
        # Return interpolated mags at observation phases
        return (model_obs_mags_kp, model_obs_mags_h)
    
    
    def star_lc_add_binary_var(
            self, star, num_lcs_generate=1):
        """For a given star, add mock binary var light curves
        
        Parameters
        ----------
        star : str
            Name of the star, from the Kp align, that the mock binary light
            curves are being added to
        num_lcs_generate : int, default: 1
            Number of mock light curves to generate for the target star, with
            mock binary light curves added
        """
        
        # Read in the star's light curves
        star_mags_kp = self.mags_kp[star]
        star_mag_uncs_kp = self.mag_uncs_kp[star]
        star_mag_mean_kp = self.mag_means_kp[star]
        
        star_mags_h = self.mags_h[star]
        star_mag_uncs_h = self.mag_uncs_h[star]
        star_mag_mean_h = self.mag_means_h[star]
        
        # Calculate median magnitude
        star_det_filt_kp = np.where(star_mags_kp > 0.)
        star_det_filt_h = np.where(star_mags_h > 0.)
        
        star_mag_med_kp = np.median(star_mags_kp[star_det_filt_kp])
        star_mag_med_h = np.median(star_mags_h[star_det_filt_h])
        
        
        # Find all mock binaries that have median magnitudes within +- 0.25 mags
        mag_rad = 0.25
        
        mod_bin_mag_filt = np.where(
            np.logical_and(
                self.model_lc_params_table['med_mag_kp'] >=\
                    star_mag_med_kp - mag_rad,
                self.model_lc_params_table['med_mag_kp'] <=\
                    star_mag_med_kp + mag_rad,
            )
        )
        
        mod_bins_filt_lc_params = self.model_lc_params_table[mod_bin_mag_filt]
        mod_bins_filt_inds = mod_bins_filt_lc_params['binary_index']
        
        while len(mod_bins_filt_inds) < 10:
            mag_rad = mag_rad + 0.25
            print(f'Increasing mag rad size to {mag_rad}')
        
            mod_bin_mag_filt = np.where(
                np.logical_and(
                    self.model_lc_params_table['med_mag_kp'] >=\
                        star_mag_med_kp - mag_rad,
                    self.model_lc_params_table['med_mag_kp'] <=\
                        star_mag_med_kp + mag_rad,
                )
            )
        
            mod_bins_filt_lc_params = self.model_lc_params_table[mod_bin_mag_filt]
            mod_bins_filt_inds = mod_bins_filt_lc_params['binary_index']
        
        
        # Select n random binaries from the selected binaries
        rng = np.random.default_rng()
        
        selected_bin_inds = rng.choice(
            mod_bins_filt_inds,
            size=num_lcs_generate,
            replace=True,
        )
        
        selected_bin_phase_shifts = rng.random(
            size=num_lcs_generate,
        )
        
        selected_lc_params = mod_bins_filt_lc_params[
              np.where(
                  np.isin(
                      mod_bins_filt_lc_params['binary_index'],
                      selected_bin_inds,
                  )
              )
        ]
        
        # For each injected binary: generate a mock light curve at observation
        # dates with a random phase
        sel_mock_light_curves_kp = np.empty(
                                       (num_lcs_generate,
                                        self.num_nights_kp)
                                   )
        sel_mock_mean_mags_kp = np.empty(num_lcs_generate)
        
        
        sel_mock_light_curves_h = np.empty(
                                      (num_lcs_generate,
                                       self.num_nights_h)
                                  )
        sel_mock_mean_mags_h = np.empty(num_lcs_generate)
        
        for (cur_trial_ind, model_ind) in enumerate(selected_bin_inds):
            # Draw light curves with the current trial phase shift
            
            (bin_obs_mags_draw_kp,
             bin_obs_mags_draw_h) = self.draw_bin_mags(
                model_ind, selected_bin_phase_shifts[cur_trial_ind],
            )
            
            # Read in mod_lc row for binary
            mod_lc_params_row = self.model_lc_params_table.loc[model_ind]
            
            # Subtract median from light curves and store
            sel_mock_light_curves_kp[cur_trial_ind] =\
                bin_obs_mags_draw_kp - mod_lc_params_row['med_mag_kp']
            
            sel_mock_light_curves_h[cur_trial_ind] =\
                bin_obs_mags_draw_h - mod_lc_params_row['med_mag_h']
            
        
        # Make mock light curves, with star lc + model mags
        
        # Kp
        star_mock_light_curves_kp = np.empty(
                                       (num_lcs_generate,
                                        self.num_nights_kp)
                                    )
        star_mock_light_curves_kp_uncs = np.empty(
                                            (num_lcs_generate,
                                             self.num_nights_kp)
                                         )
        
        # Step through each mock binary, and then each epoch
        for cur_trial_index in range(num_lcs_generate):
            for cur_epoch_index in range(self.num_nights_kp):
                cur_star_mag = star_mags_kp[cur_epoch_index]
                cur_star_mag_unc = star_mag_uncs_kp[cur_epoch_index]
                
                # If star not detected on this date, continue
                if cur_star_mag < 0:
                    star_mock_light_curves_kp[
                        cur_trial_index,
                        cur_epoch_index] = np.nan
                    star_mock_light_curves_kp_uncs[
                        cur_trial_index,
                        cur_epoch_index] = np.nan
                    continue
                
                # Otherwise read in mock mag and uncs for this observation date
                cur_epoch_model_mag = sel_mock_light_curves_kp[
                                          cur_trial_index,
                                          cur_epoch_index]
                cur_epoch_unc_table = self.epoch_unc_tables_kp[cur_epoch_index]

                # Apply filter for where unc values are defined
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
                star_mock_light_curves_kp[
                    cur_trial_index,
                    cur_epoch_index] = cur_star_mag + cur_epoch_model_obs_mag
                
                star_mock_light_curves_kp_uncs[
                    cur_trial_index,
                    cur_epoch_index] = cur_star_mag_unc
        
        # H
        star_mock_light_curves_h = np.empty(
                                       (num_lcs_generate,
                                        self.num_nights_h)
                                    )
        star_mock_light_curves_h_uncs = np.empty(
                                            (num_lcs_generate,
                                             self.num_nights_h)
                                         )
        
        # Step through each mock binary, and then each epoch
        for cur_trial_index in range(num_lcs_generate):
            for cur_epoch_index in range(self.num_nights_h):
                cur_star_mag = star_mags_h[cur_epoch_index]
                cur_star_mag_unc = star_mag_uncs_h[cur_epoch_index]
                
                # If star not detected on this date, continue
                if cur_star_mag < 0:
                    star_mock_light_curves_h[
                        cur_trial_index,
                        cur_epoch_index] = np.nan
                    star_mock_light_curves_h_uncs[
                        cur_trial_index,
                        cur_epoch_index] = np.nan
                    continue
                
                # Otherwise read in mock mag and uncs for this observation date
                cur_epoch_model_mag = sel_mock_light_curves_h[
                                          cur_trial_index,
                                          cur_epoch_index]
                cur_epoch_unc_table = self.epoch_unc_tables_h[cur_epoch_index]

                # Apply filter for where unc values are defined
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
                star_mock_light_curves_h[
                    cur_trial_index,
                    cur_epoch_index] = cur_star_mag + cur_epoch_model_obs_mag
                
                star_mock_light_curves_h_uncs[
                    cur_trial_index,
                    cur_epoch_index] = cur_star_mag_unc
        
        
        # Return star lc + mock values
        return (selected_bin_inds,
                star_mock_light_curves_kp,
                star_mock_light_curves_kp_uncs,
                star_mock_light_curves_h,
                star_mock_light_curves_h_uncs,
               )
    
    def create_star_binary_var_table(
            self, star, num_lcs_generate=1,
            star_bin_var_out_dir='./star_bin_var/',
            print_diagnostics=False):
        """
        Function to create a table of mock light curves for the target star
        
        Parameters
        ----------
        star : str
            Name of the star, from the Kp align, that the mock binary light
            curves are being added to
        num_lcs_generate : int, default: 1
            Number of mock light curves to generate for the target star, with
            mock binary light curves added
        star_bin_var_out_dir : str, default: './star_bin_var/'
            Directory where the output tables live for each star,
            with mock binary light curves injected
        print_diagnostics : bool, default: False
            Specify if to print diagnostics during run
        """
        
        # Make sure output directory exists
        if not os.path.exists(star_bin_var_out_dir):
            os.makedirs(star_bin_var_out_dir)
        
        # Generate the mock light curves
        star_lc_add_binary_var_out = self.star_lc_add_binary_var(
            star, num_lcs_generate=num_lcs_generate
        )
        
        (selected_bin_ids,
         star_mock_light_curves_kp,
         star_mock_light_curves_kp_uncs,
         star_mock_light_curves_h,
         star_mock_light_curves_h_uncs,
        ) = star_lc_add_binary_var_out
        
        # Create output table
        star_bin_var_ids = list(range(num_lcs_generate))
        
        star_table = Table([star_bin_var_ids,
                            selected_bin_ids,
                            star_mock_light_curves_kp,
                            star_mock_light_curves_kp_uncs,
                            star_mock_light_curves_h,
                            star_mock_light_curves_h_uncs,
                           ],
                           names=('star_bin_var_ids',
                                  'selected_bin_ids',
                                  'mag_kp',
                                  'mag_unc_kp',
                                  'mag_h',
                                  'mag_unc_h',
                                 )
                          )
        star_table.add_index('star_bin_var_ids')
        
        # Write out the output table
        star_table.write(f'{star_bin_var_out_dir}/{star}.h5',
                         path='data', serialize_meta=True, compression=True,
                         overwrite=True)
        
        if print_diagnostics:
            print(star_table)
        
        return star_table
