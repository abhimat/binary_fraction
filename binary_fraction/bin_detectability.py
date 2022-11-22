# Class to determine detection of injected mock binary signal
# with periodicity search
# ---
# Abhimat Gautam

import numpy as np
from astropy.table import Table
from gc_photdata import align_dataset
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
from tqdm import tqdm

class bin_detectability(object):
    """
    Object to determine detection of injected mock binary signals
    with periodicity search
    """
    
    def __init__(self):
        # Set up defaults
        
        return
    
    def set_sbv_dir(self, sbv_dir='../star_bin_var/'):
        """
        Set sbv_dir variable for class, where the light curves
        with injected binary variability live.
        
        Parameters
        ----------
        sbv_dir : str, default: '../star_bin_var/'
            Directory where the injected bin var light curves are stored
        """
        self.sbv_dir = sbv_dir
        
        return
    
    def load_model_sb_params_table(
            self, table_file_path='../stellar_binary_params.h5',
        ):
        """
        Load the binary model light curve parameters table. Assumes table is
        stored in astropy tables hdf5 format
        
        Parameters
        ----------
        table_file_path : str, default: '../stellar_binary_params.h5'
            File path of the stellar binary parameters table
        """
        # Load in the table
        model_sb_params_table = Table.read(
            table_file_path, path='data')
        
        model_sb_params_table.add_index(['binary_index'])
        
        # Save out as a class variable
        self.model_sb_params_table = model_sb_params_table
        
        return
    
    def read_align_data(
            self, align_kp_name, align_h_name,
            align_data_location = '/g/ghez/abhimat/datasets/align_data_py3/'
        ):
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
        
        # Calculate experiment time baseline here
        self.time_baseline_kp = np.max(self.epoch_MJDs_kp) -\
            np.min(self.epoch_MJDs_kp)
        self.time_baseline_h = np.max(self.epoch_MJDs_h) -\
            np.min(self.epoch_MJDs_h)
        self.time_baseline = np.max([
            self.time_baseline_kp, self.time_baseline_h
        ])
        
        return
    
    def compute_detectability(
            self, stars_list,
            num_mock_bins=50,
            out_bin_detect_table_root='./bin_detect',
            print_diagnostics=False,
        ):
        """
        Compute the detectability of the injected light curves for the
        stars provided in stars_list
        
        Parameters
        ----------
        stars_list : list[str]
            List of star names to compute the detectability for
        """
        # Calculate some parameters for search
        longPer_boundary = self.time_baseline / 4.
        longPer_BSSig_boundary = 0.5
        
        # Create empty arrays for storing outputs
        stars_passing_frac = np.zeros(len(stars_list))
        stars_passing_sbvs = np.zeros(
            (len(stars_list), num_mock_bins),
            dtype=bool,
        )
        
        stars_half_bin_per_detected_sbvs = np.zeros(
            (len(stars_list), num_mock_bins),
            dtype=bool,
        )
        stars_full_bin_per_detected_sbvs = np.zeros(
            (len(stars_list), num_mock_bins),
            dtype=bool,
        )
        
        stars_half_bin_per_LS_sig_sbvs = np.zeros(
            (len(stars_list), num_mock_bins),
            dtype=float,
        )
        stars_full_bin_per_LS_sig_sbvs = np.zeros(
            (len(stars_list), num_mock_bins),
            dtype=float,
        )
        
        # Compute detectability for every star in specified sample
        for (star_index, star) in tqdm(enumerate(stars_list)):
            star_table = Table.read(
                f'{self.sbv_dir}/{star}.h5',
                path='data')
            star_table.add_index('star_bin_var_ids')
    
            star_out_LS_table = Table.read(
                f'{self.sbv_dir}/LS_Periodicity_Out/{star}.h5',
                path='data')
    
            # Go through each unique mock index for the given star
            
            if print_diagnostics:
                print(star_table)
                print(star_out_LS_table)
            
            inj_sbv_ids = star_table['star_bin_var_ids']
            LS_sbv_ids = np.unique(star_out_LS_table['bin_var_id']).astype(int)
    
            passing_sbvs = []
            
            for sbv in LS_sbv_ids:
                sbv_filter = np.where(star_out_LS_table['bin_var_id'] == sbv)
                sbv_LS_results = star_out_LS_table[sbv_filter]
        
                mock_bin_id = (star_table.loc[sbv])['selected_bin_ids']
                mock_bin_row = self.model_sb_params_table.loc[mock_bin_id]
        
                mock_true_period = mock_bin_row['binary_period']
        
                # print(mock_LS_results)
        
                # Check for long period getting aliased
                longPer_filt = np.where(sbv_LS_results['LS_periods'] >= longPer_boundary)
                longPer_filt_results = sbv_LS_results[longPer_filt]
        
                if len(longPer_filt_results) > 0:
                    continue
                
                # Check for a signal at binary period and half of binary period
                binPer_filt = np.where(np.logical_and(
                    sbv_LS_results['LS_periods'] >= mock_true_period * 0.99,
                    sbv_LS_results['LS_periods'] <= mock_true_period * 1.01))
            
                binPer_half_filt = np.where(np.logical_and(
                    sbv_LS_results['LS_periods'] >= (0.5*mock_true_period) * 0.99,
                    sbv_LS_results['LS_periods'] <= (0.5*mock_true_period) * 1.01))
        
                binPer_filt_results = sbv_LS_results[binPer_filt]
                binPer_half_filt_results = sbv_LS_results[binPer_half_filt]
        
                if (len(binPer_filt_results) + len(binPer_half_filt_results)) == 0:
                    continue
        
                # Success
                # print(f'\tMock {mock} passed ({mock_true_period:.3f} d period)')
                
                passing_sbvs.append(sbv)
                stars_passing_sbvs[star_index, sbv] = True
                
                if len(binPer_filt_results) > 0:
                    stars_full_bin_per_detected_sbvs[star_index, sbv] = True
                    stars_full_bin_per_LS_sig_sbvs[star_index, sbv] =\
                        np.max(binPer_filt_results['LS_bs_sigs'])
                
                if len(binPer_half_filt_results) > 0:
                    stars_half_bin_per_detected_sbvs[star_index, sbv] = True
                    stars_half_bin_per_LS_sig_sbvs[star_index, sbv] =\
                        np.max(binPer_half_filt_results['LS_bs_sigs'])
            
            if print_diagnostics:
                print('Passing SBVs:')
                print(passing_sbvs)
                
                print('Half binary orb. period detections')
                print(stars_half_bin_per_detected_sbvs)
                print(stars_half_bin_per_LS_sig_sbvs)
                print('---')
                print('Full binary orb. period detections')
                print(stars_full_bin_per_detected_sbvs)
                print(stars_full_bin_per_LS_sig_sbvs)
            
            # Calculate the passing fraction of all SBVs for this star
            passing_frac = len(passing_sbvs) / len(inj_sbv_ids)
            stars_passing_frac[star_index] = passing_frac
            
        # Output a table with binary detections
        
        # ascii tables can't output the multi-dimensional columns,
        # so write an output table without those columns in ascii
        bin_detect_table = Table(
            [stars_list, stars_passing_frac],
            names=['star', 'passing_frac'],
        )

        bin_detect_table['passing_frac'].format = '.2f'

        bin_detect_table.write(out_bin_detect_table_root + '.txt',
                               format='ascii.fixed_width', overwrite=True)
        
        # Can include multi-dimensional columns in HDF5,
        # so including those columns in this output
        bin_detect_table = Table(
            [
                stars_list, stars_passing_frac, stars_passing_sbvs,
                stars_full_bin_per_detected_sbvs,
                stars_half_bin_per_detected_sbvs,
                stars_full_bin_per_LS_sig_sbvs,
                stars_half_bin_per_LS_sig_sbvs,
            ],
            names=[
                'star', 'passing_frac', 'passing_sbvs',
                'full_bin_per_detected_sbvs',
                'half_bin_per_detected_sbvs',
                'full_bin_per_LS_sig_sbvs',
                'half_bin_per_LS_sig_sbvs',
            ],
        )

        bin_detect_table.write(out_bin_detect_table_root + '.h5',
                               format='hdf5', path='data', overwrite=True)
        
        # Return final table
        return bin_detect_table
