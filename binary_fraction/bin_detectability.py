# Class to determine detection of injected mock binary signal
# with periodicity search
# ---
# Abhimat Gautam

import numpy as np
from astropy.table import Table
from gc_photdata import align_dataset
from . import mcmc_fit_trend_sinusoid
from gatspy import periodic
import emcee
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import copy
from tqdm import tqdm
from multiprocessing.pool import Pool
import schwimmbad

class bin_detectability(object):
    """
    Object to determine detection of injected mock binary signals
    with periodicity search
    """
    
    longPer_boundary = 365.25
    longPer_BSSig_boundary = 0.5
    
    num_cores=32
    
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
    
    def load_sbv_sample_table(self, table_file_name='sample_table.txt'):
        """
        Load the SBV sample table
        """
        self.sbv_sample_table = Table.read(
            f'{self.sbv_dir}/{table_file_name}',
            format='ascii.fixed_width',
        )
        
        self.sbv_sample_table.add_index(['star'])
        
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
    
    def fit_trended_sinusoid(
            self,
            poly_trend_order, t0, period,
            mags, mag_errors,
            obs_days, obs_filts,
            mp_pool=None,
            run_initial_LS=True,
            run_initial_polyfit=True,
            show_MCMC_progress=False,
            mcmc_steps=500,
            last_steps=200,
            print_diagnostics=False,
        ):
        """
        Run MCMC fit for trended sinusoid
        """
        os.environ["OMP_NUM_THREADS"] = "1"
        
        kp_obs_filt = np.where(obs_filts == b'kp')
        h_obs_filt = np.where(obs_filts == b'h')
        
        # Setup MCMC fit object
        mcmc_fit_obj = mcmc_fit_trend_sinusoid.mcmc_fitter()
        
        # Set fit parameters in MCMC fit object
        mcmc_fit_obj.set_poly_trend_order_base(poly_trend_order)
        mcmc_fit_obj.set_t0(t0)
        mcmc_fit_obj.set_period(period)
        
        # Set observations in MCMC fit object
        mcmc_fit_obj.set_observation_filts(obs_filts)
        mcmc_fit_obj.set_observation_times(obs_days)
        mcmc_fit_obj.set_observations(mags, mag_errors)
        
        # Compute poly trend Lomb-Scargle and polynomial fit
        # for initial parameter estimates
        if run_initial_LS:
            peak_omega = 2. * np.pi / (period/2.)
            
            min_period_exp = 0.
            min_period = 10. ** min_period_exp
            
            max_period_exp = 4.
            max_period = 10. ** max_period_exp
            
            min_freq = 1./max_period
            max_freq = 1./min_period
            
            samps_per_peak = 5.
            obs_length = np.max(obs_days) - np.min(obs_days)
            
            freq_spacing = 1. / (samps_per_peak * obs_length)
            test_freqs = np.arange(min_freq, max_freq, freq_spacing)
            
            test_periods = 1./test_freqs
            
            LS_multi = periodic.PolyTrend_LombScargleMultiband(
                poly_trend_order_base = poly_trend_order,
                poly_trend_order_band = 1,
                Nterms_base = 1,
                Nterms_band = 0,
            )
            
            LS_multi.fit(obs_days - t0, mags, mag_errors, obs_filts)
            pdgram = LS_multi.periodogram(test_periods)
            
            pdgram_ymean = LS_multi._compute_ymean()
            
            pdgram_max_theta = LS_multi._best_params(peak_omega)
        
        if run_initial_polyfit:
            t_fit = obs_days - t0
            
            mags_polyfit = copy.deepcopy(mags)
            
            poly_coeffs = np.polyfit(
                t_fit[kp_obs_filt],
                mags[kp_obs_filt],
                poly_trend_order,
                w=(1./mag_errors[kp_obs_filt]),
            )
        
        nwalkers = 60
        
        # Set up initial sampler position
        if len(mags[h_obs_filt]) > 0:
            ndim = poly_trend_order + 5
            
            # Initial position, from LS run and polyfit fit
            t0_init = t0
            cos_amp_init = np.hypot(pdgram_max_theta[-5], pdgram_max_theta[-6])
            h_add_init = np.max(mags[h_obs_filt]) - np.max(mags[kp_obs_filt])
        
            theta_init = np.array([t0_init])
            theta_init = np.append(theta_init, poly_coeffs[::-1])
            theta_init = np.append(
                theta_init,
                np.array([cos_amp_init, h_add_init, 0.0]),
            )
        
            scale_mult = np.array([1e-1*period])
            scale_mult = np.append(scale_mult, poly_coeffs[::-1] * 0.1)
            scale_mult = np.append(
                scale_mult,
                np.array([1e-3, 1e-1, 1e-3]),
            )
        else:
            ndim = poly_trend_order + 3
            
            # Initial position, from LS run and polyfit fit
            t0_init = t0
            cos_amp_init = np.hypot(pdgram_max_theta[-3], pdgram_max_theta[-4])
        
            theta_init = np.array([t0_init])
            theta_init = np.append(theta_init, poly_coeffs[::-1])
            theta_init = np.append(
                theta_init,
                np.array([cos_amp_init]),
            )
        
            scale_mult = np.array([1e-1*period])
            scale_mult = np.append(scale_mult, poly_coeffs[::-1] * 0.1)
            scale_mult = np.append(
                scale_mult,
                np.array([1e-3]),
            )
        
        if print_diagnostics:
            print(f'Initial theta: {theta_init}')
            log_prob_init = mcmc_fit_obj.log_probability(theta_init)
            print(f'Initial log prob: {log_prob_init:.5f}')
        
        pos = np.tile(theta_init, (nwalkers,1)) +\
              (scale_mult * np.random.randn(nwalkers, ndim))
        
        # Set up sampler
        if mp_pool == None:
            mp_pool = schwimmbad.MultiPool(self.num_cores)
        
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, mcmc_fit_obj.log_probability,
            pool=mp_pool,
        )
        
        sampler.run_mcmc(pos, mcmc_steps, progress=show_MCMC_progress)
        
        samples = sampler.get_chain(flat=True)
        log_prob_samples = sampler.get_log_prob(flat=True)
        
        samples = samples[(-1 * last_steps * nwalkers):,:]
        log_prob_samples = log_prob_samples[(-1 * last_steps * nwalkers):]
        
        log_prob_argmax_index = np.unravel_index(
            np.argmax(log_prob_samples), log_prob_samples.shape,
        )
        
        # Determine best fit cos amplitude
        if len(mags[h_obs_filt]) > 0:
            maxProb_cos_amp = (samples[:, ndim - 3])[log_prob_argmax_index]
            cos_amp_sig1_nums = np.percentile(
                samples[:, ndim - 3],
                [15.866, 50., 84.134], axis=0,
            )
        else:
            maxProb_cos_amp = (samples[:, -1])[log_prob_argmax_index]
            cos_amp_sig1_nums = np.percentile(
                samples[:, -1],
                [15.866, 50., 84.134], axis=0,
            )
        
        cos_amp_sig1 = 0.5 * (cos_amp_sig1_nums[2] - cos_amp_sig1_nums[0])
        
        cos_amp_sig = maxProb_cos_amp / cos_amp_sig1
        
        return (cos_amp_sig, maxProb_cos_amp, cos_amp_sig1)
    
    def compute_amp_sig(
            self, star,
            num_mock_bins=50,
            low_sig_check = 0.60,
            print_diagnostics=False,
            mp_pool = None,
        ):
        """
        Compute the amplitude significance of possible binary detection signals
        
        Parameters
        ----------
        star : str
            Star name to compute the amplitude significance for
        """
        
        # Make output directory
        amp_sigs_dir = f'{self.sbv_dir}/amp_sigs/'
        
        os.makedirs(amp_sigs_dir, exist_ok=True)
        
        # Compute detectability for every star in specified sample
        # # If cos amp sig calculations already complete for star, return
        # if os.path.isfile(f'{amp_sigs_dir}/{star}.h5'):
        #     print(f'Amp sigs for {star} already computed')
        #     return
        
        print(f'Computing amp sigs for {star}')
        
        # Determine polynomial trend order (minimum 1)
        sbv_sample_table_star_row = self.sbv_sample_table.loc[star]
        star_poly_trend_order = sbv_sample_table_star_row['poly_trend_order']
        
        if star_poly_trend_order < 1:
            star_poly_trend_order = 1
        
        # Read star tables
        star_table = Table.read(
            f'{self.sbv_dir}/{star}.h5',
            path='data')
        star_table.add_index('star_bin_var_ids')
        
        star_out_LS_table = Table.read(
            f'{self.sbv_dir}/LS_Periodicity_Out/{star}.h5',
            path='data',
        )
        
        # Go through each sbv for the given star
        # Compute amplitude only for most significant peak,
        # if most sig peak is consistent or not with binary detection
        
        inj_sbv_ids = star_table['star_bin_var_ids']
        LS_sbv_ids = np.unique(star_out_LS_table['bin_var_id']).astype(int)
        
        # Empty array to store significances
        cos_amp_sigs = np.zeros(len(LS_sbv_ids))
        cos_amps = np.zeros(len(LS_sbv_ids))
        cos_amp_sig1s = np.zeros(len(LS_sbv_ids))
        bin_per_match = np.full(
            len(LS_sbv_ids), False, dtype=bool,
        )
        
        if print_diagnostics:
            print('\n===')
            print(f'Current star: {star}')
            print(star_table)
            print(star_out_LS_table)
            print(cos_amp_sigs)
        
        for (sbv_index, sbv) in tqdm(enumerate(LS_sbv_ids), total=len(LS_sbv_ids)):
            sbv_filter = np.where(star_out_LS_table['bin_var_id'] == sbv)
            sbv_LS_results = star_out_LS_table[sbv_filter]
            
            mock_bin_id = (star_table.loc[sbv])['selected_bin_ids']
            mock_bin_row = self.model_sb_params_table.loc[mock_bin_id]
            mock_bin_lc_row = self.model_lc_params_table.loc[mock_bin_id]
            
            mock_true_period = mock_bin_row['binary_period']
            
            if print_diagnostics:
                print('---')
                print(f'SBV ID: {sbv}')
                print(f'True Binary Period: {mock_true_period:.3f} d')
            
            # Check for long period getting aliased
            longPer_filt = np.where(
                sbv_LS_results['LS_periods'] >= self.longPer_boundary)
            longPer_filt_results = sbv_LS_results[longPer_filt]
            
            if len(longPer_filt_results) > 0:
                if print_diagnostics:
                    print('Long period alias check failing')
                
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
            
            # Perform checks for LS significance and LS significance
            matching_sigs = np.append(
                binPer_filt_results['LS_bs_sigs'],
                binPer_half_filt_results['LS_bs_sigs'],
            )
            matching_powers = np.append(
                binPer_filt_results['LS_powers'],
                binPer_half_filt_results['LS_powers'],
            )
            matching_periods = np.append(
                binPer_filt_results['LS_periods'],
                binPer_half_filt_results['LS_periods'],
            )
            
            peak_sig = np.max(sbv_LS_results['LS_bs_sigs'])
            
            if peak_sig < low_sig_check:
                if print_diagnostics:
                    print(f'Peak significance is lower than low_sig_check: {low_sig_check}')
                
                continue
            
            # Success: possible detected peak, need amplitude check
            if print_diagnostics:
                print(f'\tSBV detected in search')
                print(f'\tNeed amplitude search')
                print('SBV LS Results Table')
                print(sbv_LS_results)
            
            # Determine fit period
            fit_period = (sbv_LS_results['LS_periods'])[
                np.argmax(sbv_LS_results['LS_powers'])
            ]
            
            
            # Check if most significant two peaks are consistent
            # with binary detection
            if len(sbv_LS_results) > 1:
                sorted_powers = np.sort(sbv_LS_results['LS_powers'])
                if (sorted_powers[-1] in matching_powers or
                    sorted_powers[-2] in matching_powers):
                    if print_diagnostics:
                        print('Most significant peaks are from binary period')
                    bin_per_match[sbv_index] = True
            else:
                if fit_period in matching_periods:
                    if print_diagnostics:
                        print('Most significant period is binary period')
                    bin_per_match[sbv_index] = True
                
            
            if print_diagnostics:
                print(f'Fitting for trended sinusoid at period {fit_period:.5f} d')
            
            # Construct dataset for fitting trended sinusoid
            kp_mags = (star_table.loc[sbv])['mag_kp']
            kp_mag_errors = (star_table.loc[sbv])['mag_unc_kp']
            kp_MJDs = self.epoch_MJDs_kp
            
            kp_det_epochs = np.where(kp_mags > 0.)
            
            kp_mags = kp_mags[kp_det_epochs]
            kp_mag_errors = kp_mag_errors[kp_det_epochs]
            kp_MJDs = kp_MJDs[kp_det_epochs]
            
            h_mags = (star_table.loc[sbv])['mag_h']
            h_mag_errors = (star_table.loc[sbv])['mag_unc_h']
            h_MJDs = self.epoch_MJDs_h
            
            h_det_epochs = np.where(h_mags > 0.)
            
            h_mags = h_mags[h_det_epochs]
            h_mag_errors = h_mag_errors[h_det_epochs]
            h_MJDs = h_MJDs[h_det_epochs]
            
            t0 = kp_MJDs[np.argmax(kp_mags)]
            
            mags = np.append(kp_mags, h_mags)
            mag_errors = np.append(kp_mag_errors, h_mag_errors)
            obs_days = np.append(kp_MJDs, h_MJDs)
            obs_filts = np.append(
                np.full_like(kp_MJDs, 'kp', dtype='|S2'),
                np.full_like(h_MJDs, 'h', dtype='|S2'),
            )
            
            (cos_amp_sig,
             cos_amp,
             cos_amp_sig1,) = self.fit_trended_sinusoid(
                star_poly_trend_order, t0, fit_period,
                mags, mag_errors,
                obs_days, obs_filts,
                mp_pool=mp_pool,
                print_diagnostics=print_diagnostics,
            )
            
            if print_diagnostics:
                print(f'Cos Amp Sig = {cos_amp_sig:.5f}')
            
            cos_amp_sigs[sbv_index] = cos_amp_sig
            cos_amps[sbv_index] = cos_amp
            cos_amp_sig1s[sbv_index] = cos_amp_sig1
            
        star_amp_sig_table = Table(
            [
                LS_sbv_ids,
                cos_amp_sigs,
                cos_amps,
                cos_amp_sig1s,
                bin_per_match,
            ],
            names=(
                'star_bin_var_ids',
                'cos_amp_sigs',
                'cos_amps',
                'cos_amp_sig1_uncs',
                'bin_per_match',
            )
        )
        
        star_amp_sig_table.write(
            f'{amp_sigs_dir}/{star}.h5',
            path='data', compression=True,
            overwrite=True,
        )
        try:
            star_amp_sig_table.write(
                f'{amp_sigs_dir}/{star}.txt',
                format='ascii.fixed_width',
                overwrite=True,
            )
        except Exception as ex:
            print(f'Writing .txt table raised exception:\n{ex}')
        
        
        return
    
    
    def compute_detectability(
            self, stars_list,
            num_mock_bins=50,
            low_sig_check = 0.70,
            high_sig_check = 0.95,
            min_amp_check = 3.0,
            amp_check = 20.0,
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
        
        # Create empty arrays for storing outputs
        stars_passing_frac = np.zeros(len(stars_list))
        stars_passing_sbvs = np.zeros(
            (len(stars_list), num_mock_bins),
            dtype=bool,
        )
        
        stars_direct_detection_detected_sbvs = np.zeros(
            (len(stars_list), num_mock_bins),
            dtype=bool,
        )
        stars_alias_detection_detected_sbvs = np.zeros(
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
        
        passing_sbvs_LS_sig_all = np.array([])
        passing_sbvs_sin_amp_all = np.array([])
        
        # Compute detectability for every star in specified sample
        for (star_index, star) in tqdm(enumerate(stars_list), total=len(stars_list)):
            # Read star model, LS, and amp sig tables
            star_table = Table.read(
                f'{self.sbv_dir}/{star}.h5',
                path='data')
            star_table.add_index('star_bin_var_ids')
            
            star_out_LS_table = Table.read(
                f'{self.sbv_dir}/LS_Periodicity_Out/{star}.h5',
                path='data',
            )
            
            star_amp_sig_table = Table.read(
                f'{self.sbv_dir}/amp_sigs/{star}.h5',
                path='data',
            )
            star_amp_sig_table.add_index('star_bin_var_ids')
            
            # Go through each unique mock index for the given star
            
            if print_diagnostics:
                print('\n===')
                print(f'Current star: {star}')
                print(star_table)
                print(star_out_LS_table)
            
            inj_sbv_ids = star_table['star_bin_var_ids']
            LS_sbv_ids = np.unique(star_out_LS_table['bin_var_id']).astype(int)
    
            passing_sbvs = []
            passing_sbvs_LS_sig = []
            passing_sbvs_sin_amp = []
            
            for sbv in LS_sbv_ids:
                sbv_filter = np.where(star_out_LS_table['bin_var_id'] == sbv)
                sbv_LS_results = star_out_LS_table[sbv_filter]
        
                mock_bin_id = (star_table.loc[sbv])['selected_bin_ids']
                mock_bin_row = self.model_sb_params_table.loc[mock_bin_id]
                mock_bin_lc_row = self.model_lc_params_table.loc[mock_bin_id]
                
                mock_true_period = mock_bin_row['binary_period']
        
                if print_diagnostics:
                    print('---')
                    print(f'SBV ID: {sbv}')
                    print(f'True Binary Period: {mock_true_period:.3f} d')
        
                # Check for long period getting aliased
                longPer_filt = np.where(
                    sbv_LS_results['LS_periods'] >= self.longPer_boundary)
                longPer_filt_results = sbv_LS_results[longPer_filt]
                
                if len(longPer_filt_results) > 0:
                    if print_diagnostics:
                        print('Long period alias check failing')
                    
                    continue
                
                # Check for a signal at binary period and half of binary period
                binPer_filt = np.where(np.logical_and(
                    sbv_LS_results['LS_periods'] >= mock_true_period * 0.99,
                    sbv_LS_results['LS_periods'] <= mock_true_period * 1.01
                ))
            
                binPer_half_filt = np.where(np.logical_and(
                    sbv_LS_results['LS_periods'] >= (0.5*mock_true_period) * 0.99,
                    sbv_LS_results['LS_periods'] <= (0.5*mock_true_period) * 1.01
                ))
                
                binPer_filt_results = sbv_LS_results[binPer_filt]
                binPer_half_filt_results = sbv_LS_results[binPer_half_filt]
                
                # Check for signals that are not at either binary period
                # or half binary period
                
                
                
                if (len(binPer_filt_results) + len(binPer_half_filt_results)) == 0:
                    if print_diagnostics:
                        print('No periodic signal found at binary period')
                    
                    continue
                
                # Perform checks for LS significance and LS significance
                matching_sigs = np.append(
                    binPer_filt_results['LS_bs_sigs'],
                    binPer_half_filt_results['LS_bs_sigs'],
                )
                matching_powers = np.append(
                    binPer_filt_results['LS_powers'],
                    binPer_half_filt_results['LS_powers'],
                )
                matching_periods = np.append(
                    binPer_filt_results['LS_periods'],
                    binPer_half_filt_results['LS_periods'],
                )
                
                peak_sig = np.max(sbv_LS_results['LS_bs_sigs'])
                peak_period = (sbv_LS_results['LS_periods'])[
                    np.argmax(sbv_LS_results['LS_powers'])
                ]
                peak_amp = (star_amp_sig_table.loc[sbv])['cos_amp_sigs']
                
                if print_diagnostics:
                    print(f'Peak delta mag / med mag unc = {peak_amp:.3f}')
                    print(f'Peak BS sig = {(peak_sig*100):.3f}%')
                
                if peak_amp < min_amp_check:
                    if print_diagnostics:
                        print('Amplitude < min amp of {min_amp_check}')
                    
                    continue
                
                if peak_sig < low_sig_check:
                    if print_diagnostics:
                        print('No peaks > {low_sig_check*100}% BS significance')
                    
                    continue
                
                if peak_sig < high_sig_check and peak_amp < amp_check:
                    if print_diagnostics:
                        print('Peak < {high_sig_check*100}% significant and amplitude < {amp_check}')
                    
                    continue
                
                # Check if most significant two peaks are consistent
                # with binary detection.
                # Make relevant flags for direct / alias detection and
                # full / half period detection
                
                detection_direct = False
                detection_alias = False
                detection_full_period = False
                detection_half_period = False
                
                sorted_powers = np.sort(sbv_LS_results['LS_powers'])
                
                if sorted_powers[-1] in matching_powers:
                    detection_direct = True
                    
                    if sorted_powers[-1] in binPer_filt_results['LS_powers']:
                        detection_full_period = True
                    elif sorted_powers[-1] in binPer_half_filt_results['LS_powers']:
                        detection_half_period = True
                    
                elif len(sbv_LS_results) > 1 and sorted_powers[-2] in matching_powers:
                    detection_alias = True
                    
                    if sorted_powers[-2] in binPer_filt_results['LS_powers']:
                        detection_full_period = True
                    elif sorted_powers[-2] in binPer_half_filt_results['LS_powers']:
                        detection_half_period = True
                
                # Final check: is most significant peak a detection / alias of
                # binary signal?
                
                if not (detection_direct or detection_alias):
                    if print_diagnostics:
                        print('Most significant peak not a detection / alias of binary signal')
                    
                    continue
                
                
                # Success
                if print_diagnostics:
                    print(f'\tSBV detected in search')
                
                passing_sbvs.append(sbv)
                passing_sbvs_LS_sig.append(peak_sig)
                passing_sbvs_sin_amp.append(peak_amp)
                stars_passing_sbvs[star_index, sbv] = True
                
                stars_direct_detection_detected_sbvs[star_index, sbv] = detection_direct
                stars_alias_detection_detected_sbvs[star_index, sbv] = detection_alias
                
                stars_full_bin_per_detected_sbvs[star_index, sbv] = detection_full_period
                stars_half_bin_per_detected_sbvs[star_index, sbv] = detection_half_period
                
            
            passing_sbvs_LS_sig_all = np.append(
                passing_sbvs_LS_sig_all, np.array(passing_sbvs_LS_sig)
            )
            passing_sbvs_sin_amp_all = np.append(
                passing_sbvs_sin_amp_all, np.array(passing_sbvs_sin_amp)
            )
            
            if print_diagnostics:
                print('Passing SBVs:')
                print(passing_sbvs)
                
                print('Half binary orb. period detections')
                print(stars_half_bin_per_detected_sbvs)
                print('---')
                print('Full binary orb. period detections')
                print(stars_full_bin_per_detected_sbvs)
            
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
                stars_direct_detection_detected_sbvs,
                stars_alias_detection_detected_sbvs,
                stars_full_bin_per_detected_sbvs,
                stars_half_bin_per_detected_sbvs,
            ],
            names=[
                'star', 'passing_frac', 'passing_sbvs',
                'direct_detection_detected_sbvs',
                'alias_detection_detected_sbvs',
                'full_bin_per_detected_sbvs',
                'half_bin_per_detected_sbvs',
            ],
        )
        
        bin_detect_table.write(out_bin_detect_table_root + '.h5',
                               format='hdf5', path='data', overwrite=True)
        
        # Binary detection sig and amp table
        
        # Contstruct output table
        bin_sig_amp_table = Table(
            [
                passing_sbvs_LS_sig_all,
                passing_sbvs_sin_amp_all,
            ],
            names=[
                'LS_FA_sig',
                'sin_amp_sig',
            ],
        )
        
        # Output table
        bin_sig_amp_table.write(
            out_bin_detect_table_root + '_sig_amp.txt',
            format='ascii.fixed_width', overwrite=True)
        
        bin_sig_amp_table.write(
            out_bin_detect_table_root + '_sig_amp.h5',
            format='hdf5', path='data', overwrite=True)
        
        # Return final table
        return bin_detect_table
    
    def costruct_sig_amp_table(
            self,
            skip_stars=['irs16SW', 'S4-258', 'S2-36'],
            bin_detect_table_root='./bin_detect',
            print_diagnostics=False,
        ):
        
        # Read in bin_detect_table
        bin_detect_table = Table.read(
            bin_detect_table_root + '.h5',
            format='hdf5', path='data',
        )
        bin_detect_table.add_index('star')
        
        
        
        passing_sbvs_LS_sig_all = np.array([])
        passing_sbvs_sin_amp_all = np.array([])
        
        binary_detections_col = np.array([], dtype=bool)
        
        LS_sigs_col = np.array([])
        sin_amp_sigs_col = np.array([])
        
        binary_detections_col = np.array([], dtype=bool)
        detections_direct_col = np.array([], dtype=bool)
        detections_alias_col = np.array([], dtype=bool)
        
        detections_full_per_col = np.array([], dtype=bool)
        detections_half_per_col = np.array([], dtype=bool)
        
        for (star_index, star) in tqdm(
                enumerate(bin_detect_table['star']),
                total=len(bin_detect_table),
            ):
            # Skip adding any detections from stars that are in skip_stars
            # i.e., those that already contain a strong known binary signal
            if star in skip_stars:
                continue
            
            # Pull up binary detection row and
            # read SBV, LS sig, and amp sig tables for star
            bin_detect_row = bin_detect_table.loc[star]
            
            star_table = Table.read(
                f'{self.sbv_dir}/{star}.h5',
                path='data')
            star_table.add_index('star_bin_var_ids')
            
            star_out_LS_table = Table.read(
                f'{self.sbv_dir}/LS_Periodicity_Out/{star}.h5',
                path='data',
            )
            
            star_amp_sig_table = Table.read(
                f'{self.sbv_dir}/amp_sigs/{star}.h5',
                path='data',
            )
            star_amp_sig_table.add_index('star_bin_var_ids')
            
            if print_diagnostics:    
                print(bin_detect_row)
                print(star_table)
                print(star_out_LS_table)
                print(star_amp_sig_table)
            
            # Pull out only SBVs that had any peaks detected in LS search
            
            LS_sbv_ids = np.unique(star_out_LS_table['bin_var_id']).astype(int)
            
            
            LS_sigs_star = []
            sin_amp_sigs_star = []
            
            binary_detections_star = []
            detections_direct_star = []
            detections_alias_star = []
            
            detections_full_per_star = []
            detections_half_per_star = []
            
            for sbv in LS_sbv_ids:
                # Determine max LS significance
                sbv_filter = np.where(star_out_LS_table['bin_var_id'] == sbv)
                sbv_LS_results = star_out_LS_table[sbv_filter]
                
                max_LS_power = np.max(sbv_LS_results['LS_powers'])
                max_LS_index = np.argmax(sbv_LS_results['LS_powers'])\
                
                max_LS_sig = (sbv_LS_results['LS_powers'])[max_LS_index]
                
                # Determine sin amp significance
                amp_sig_row = star_amp_sig_table.loc[sbv]
                max_sin_amp_sig = amp_sig_row['cos_amp_sigs']
                
                
                # Determine detection characteristics
                detection_direct = bin_detect_row['direct_detection_detected_sbvs'][sbv]
                detection_alias = bin_detect_row['alias_detection_detected_sbvs'][sbv]
                
                binary_detection = (detection_direct or detection_alias)
                
                detection_full_period = bin_detect_row['full_bin_per_detected_sbvs'][sbv]
                detection_half_period = bin_detect_row['half_bin_per_detected_sbvs'][sbv]
                
                # Store out all quantities into lists for this star
                LS_sigs_star.append(max_LS_sig)
                sin_amp_sigs_star.append(max_sin_amp_sig)
                
                binary_detections_star.append(bool(binary_detection))
                detections_direct_star.append(bool(detection_direct))
                detections_alias_star.append(bool(detection_alias))
                
                detections_full_per_star.append(bool(detection_full_period))
                detections_half_per_star.append(bool(detection_half_period))
            
            # Append star detections onto complete column arrays
            LS_sigs_col = np.append(
                LS_sigs_col,
                np.array(LS_sigs_star),
            )
            sin_amp_sigs_col = np.append(
                sin_amp_sigs_col,
                np.array(sin_amp_sigs_star),
            )
            
            binary_detections_col = np.append(
                binary_detections_col,
                np.array(binary_detections_star, dtype=bool),
            )
            detections_direct_col = np.append(
                detections_direct_col,
                np.array(detections_direct_star, dtype=bool),
            )
            detections_alias_col = np.append(
                detections_alias_col,
                np.array(detections_alias_star, dtype=bool),
            )
            
            detections_full_per_col = np.append(
                detections_full_per_col,
                np.array(detections_full_per_star, dtype=bool),
            )
            detections_half_per_col = np.append(
                detections_half_per_col,
                np.array(detections_half_per_star, dtype=bool),
            )
        
        # Construct output table
        sig_amp_table = Table(
            [
                LS_sigs_col,
                sin_amp_sigs_col,
                binary_detections_col,
                detections_direct_col,
                detections_alias_col,
                detections_full_per_col,
                detections_half_per_col,
            ],
            names=[
                'LS_sigs',
                'sin_amp_sigs',
                'binary_detections',
                'detections_direct',
                'detections_alias',
                'detections_full_per',
                'detections_half_per',
            ],
        )
        
        if print_diagnostics:
            print(f'sig_amp_table:\n{sig_amp_table}')
        
        # Write and return table
        sig_amp_table.write(
            bin_detect_table_root + '_sig_amp.txt',
            format='ascii.fixed_width', overwrite=True)
        
        sig_amp_table.write(
            bin_detect_table_root + '_sig_amp.h5',
            format='hdf5', path='data', overwrite=True)
        
        return sig_amp_table

