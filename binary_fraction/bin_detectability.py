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
            run_initial_LS=True,
            run_initial_polyfit=True,
            show_MCMC_progress=False,
            mcmc_steps=500,
            last_steps=200,
        ):
        """
        Run MCMC fit for trended sinusoid
        """
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
        
        pos = np.tile(theta_init, (nwalkers,1)) +\
              (scale_mult * np.random.randn(nwalkers, ndim))
        
        # Set up sampler
        mp_pool = Pool(self.num_cores)
        
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
            self, stars_list,
            num_mock_bins=50,
            low_sig_check = 0.60,
            print_diagnostics=False,
        ):
        """
        Compute the amplitude significance of possible binary detection signals
        
        Parameters
        ----------
        stars_list : list[str]
            List of star names to compute the amplitude significance for
        """
        
        # Make output directory
        amp_sigs_dir = f'{self.sbv_dir}/amp_sigs/'
        
        os.makedirs(amp_sigs_dir, exist_ok=True)
        
        # Compute detectability for every star in specified sample
        for (star_index, star) in tqdm(enumerate(stars_list), total=len(stars_list)):
            # If cos amp sig calculations already complete for star, continue
            if os.path.isfile(f'{amp_sigs_dir}/{star}.h5'):
                continue
            
            sbv_sample_table_star_row = self.sbv_sample_table.loc[star]
            star_poly_trend_order = sbv_sample_table_star_row['poly_trend_order']
            
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
            # if most sig peak is consistent with binary detection
            
            inj_sbv_ids = star_table['star_bin_var_ids']
            LS_sbv_ids = np.unique(star_out_LS_table['bin_var_id']).astype(int)
            
            # Empty array to store significances
            cos_amp_sigs = np.zeros(len(LS_sbv_ids))
            cos_amps = np.zeros(len(LS_sbv_ids))
            cos_amp_sig1s = np.zeros(len(LS_sbv_ids))
            
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
                
                if (len(binPer_filt_results) + len(binPer_half_filt_results)) == 0:
                    if print_diagnostics:
                        print('No periodic signal found at binary period')
                    
                    continue
                
                # Perform checks for LS significance and LS significance
                matching_sigs = np.append(
                    binPer_filt_results['LS_bs_sigs'],
                    binPer_half_filt_results['LS_bs_sigs'],
                )
                matching_periods = np.append(
                    binPer_filt_results['LS_periods'],
                    binPer_half_filt_results['LS_periods'],
                )
                
                peak_sig = np.max(matching_sigs)
                
                if peak_sig < low_sig_check:
                    if print_diagnostics:
                        print(f'Peak significance is lower than low_sig_check: {low_sig_check}')
                    
                    continue
                
                # Success: possible detected peak, need amplitude check
                if print_diagnostics:
                    print(f'\tSBV detected in search')
                    print(f'\tNeed amplitude search')
                
                # Determine fit period
                fit_period = matching_periods[np.argmax(matching_sigs)]
                
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
                ],
                names=(
                    'star_bin_var_ids',
                    'cos_amp_sigs',
                    'cos_amps',
                    'cos_amp_sig1_uncs',
                )
            )
            
            star_amp_sig_table.write(
                f'{amp_sigs_dir}/{star}.h5',
                path='data', compression=True,
                overwrite=True,
            )
            star_amp_sig_table.write(
                f'{amp_sigs_dir}/{star}.txt',
                format='ascii.fixed_width',
                overwrite=True,
            )
            break
        
    
    def compute_detectability(
            self, stars_list,
            num_mock_bins=50,
            low_sig_check = 0.60,
            high_sig_check = 0.97,
            amp_check = 7.0,
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
        for (star_index, star) in tqdm(enumerate(stars_list), total=len(stars_list)):
            star_table = Table.read(
                f'{self.sbv_dir}/{star}.h5',
                path='data')
            star_table.add_index('star_bin_var_ids')
            
            star_out_LS_table = Table.read(
                f'{self.sbv_dir}/LS_Periodicity_Out/{star}.h5',
                path='data',
            )
            
            # Go through each unique mock index for the given star
            
            if print_diagnostics:
                print('\n===')
                print(f'Current star: {star}')
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
                mock_bin_lc_row = self.model_lc_params_table.loc[mock_bin_id]
                
                mock_true_period = mock_bin_row['binary_period']
        
                if print_diagnostics:
                    print('---')
                    print(f'SBV ID: {sbv}')
                    print(f'True Binary Period: {mock_true_period:.3f} d')
        
                # Check for long period getting aliased
                longPer_filt = np.where(
                    sbv_LS_results['LS_periods'] >= longPer_boundary)
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
                
                if (len(binPer_filt_results) + len(binPer_half_filt_results)) == 0:
                    if print_diagnostics:
                        print('No periodic signal found at binary period')
                    
                    continue
                
                # Perform checks for LS significance and LS significance
                matching_sigs = np.append(
                    binPer_filt_results['LS_bs_sigs'],
                    binPer_half_filt_results['LS_bs_sigs'],
                )
                
                peak_sig = np.max(matching_sigs)
                
                bin_delta_mag_kp = mock_bin_lc_row['delta_mag_kp']
                med_mag_unc_kp = np.median((star_table.loc[sbv])['mag_unc_kp'])
                
                peak_amp = bin_delta_mag_kp / med_mag_unc_kp
                
                if print_diagnostics:
                    print(f'Peak delta mag / med mag unc = {peak_amp:.3f}')
                    print(f'Peak BS sig = {(peak_sig*100):.3f}%')
                
                if peak_sig < low_sig_check:
                    if print_diagnostics:
                        print('No peaks > 60% BS significance')
                    
                    continue
                
                if peak_sig < high_sig_check and peak_amp < amp_check:
                    if print_diagnostics:
                        print('Peak < 97% significant and amplitude < 7')
                    
                    continue
                
                # Success
                if print_diagnostics:
                    print(f'\tSBV detected in search')
                
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
