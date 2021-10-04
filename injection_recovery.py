from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from pathos.multiprocessing import ProcessingPool as Pool
import dill
# import json

from utils import *
from CSMmodel import CSMmodel
from light_curve import LightCurve
from supernova import Supernova

# Default values
SIGMA = [5, 3] # detection certainty
SIGMA_COUNT = [1, 3] # Number of points at corresponding sigma to detect


def main(iterations, tstart_lims, scale_lims, save_dir, model='Chev94', **kwargs):

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
    supernovae = sn_info.sort_values('pref_dist').index

    # Record luminosities of CSM model for scale 1
    base_model = CSMmodel(0, WIDTH, DECAY_RATE, scale=1, model=model)
    base_model_z = 0.04
    scale1 = {'base_model_fuv_lum': base_model(0, base_model_z)['FUV'],
              'base_model_nuv_lum': base_model(0, base_model_z)['NUV'],
              'base_model_hst_lum': base_model(0, base_model_z)['F275W']}
    with open(save_dir / Path('_scale.txt'), 'w') as file:
        file.write(str(scale1))

    run_all(supernovae, iterations, tstart_lims, scale_lims, sn_info=sn_info, 
            model=model, save_dir=save_dir, **kwargs)


def run_all(supernovae, iterations, tstart_lims, scale_lims, sn_info=[], 
        overwrite=False, model='Chev94', save_dir=SAVE_DIR, **kwargs):
    """Run injection recovery trials on all supernovae in given list.
    Inputs:
        supernovae: list of supernova names
        iterations: iterations of injection & recovery for each SN
        tstart_lims: tuple or list of bounds on start time
        scale_lims: tuple or list of bounds on scale factor
        sn_info: supernova info reference DataFrame
        overwrite: bool, whether to overwrite previous save files
        kwargs: keyword arguments for run_trials
    """

    # Remove SNe with previous save files from list
    if not overwrite:
        supernovae = [s for s in supernovae if not check_save(s, iterations, save_dir=save_dir)]
    
    for i, sn_name in enumerate(supernovae):
        print('\n%s [%s/%s]' % (sn_name, i+1, len(supernovae)))
        # Initialize SN object
        sn = Supernova(sn_name, sn_info=sn_info)
        # Import light curves
        lcs = []
        for band in ['FUV', 'NUV']:
            try:
                lc = LightCurve(sn, band, data_dir=DATA_DIR, sed=model)
            except:
                # No data for this channel
                continue

            # Skip if no data after minimum recovery time
            if np.max(lc.data['t_delta_rest']) < RECOV_MIN:
                print('\tno %s data after minimum %s days past discovery, skipping' % (band, RECOV_MIN))
                continue

            lcs.append(lc)

        # If light curve import was unsuccessful
        if len(lcs) == 0:
            print('\tno data available!')
            continue

        run_trials(sn, lcs, iterations, tstart_lims, scale_lims, model=model, 
                save_dir=save_dir, **kwargs)


def run_trials(sn, lcs, iterations, tstart_lims, scale_lims, save=True, 
        save_dir='', **kwargs):
    """Run injection recovery a given number of times on one supernova.
    Inputs:
        sn: Supernova object
        lcs: list of associated LightCurve objects
        iterations: iterations of injection & recovery
        tstart_lims: tuple or list of bounds on start time
        scale_lims: tuple or list of bounds on scale factor
        save: bool, output recovery_df to CSV
        save_dir: directory to place save file
        model: 'Chev94' or 'flat'
        kwargs: keyword arguments for inject_recover
    Outputs:
        recovery_df: DataFrame of injection parameters and recovered times
    """

    # Random injection parameter sample
    params = gen_params(iterations, tstart_lims, scale_lims, log=True)

    # Run injection-recovery trials in parallel
    recovery_df = []
    with Pool() as pool:
        func = partial(inject_recover, sn=sn, lcs=lcs, **kwargs)
        imap = pool.imap(func, params, chunksize=100)
        for recovery in tqdm(imap, total=iterations):
            recovery_df.append(recovery)

    recovery_df = pd.DataFrame(recovery_df, 
            columns=['tstart', 'scale', 'recovered'])

    # Save CSV
    if save:
        fname = sn2fname(sn.name, str(iterations)) # format sn_name-iterations.csv
        recovery_df.to_csv(save_dir / fname, index=False)

    return recovery_df


def inject_recover(params, sn, lcs, sigma=SIGMA, count=SIGMA_COUNT, **kwargs):
    """Perform injection and recovery for given SN and model parameters.
    Inputs:
        params: tuple of (tstart, scale) injection parameters
        sn: Supernova object
        lcs: list of LightCurve objects (typically NUV and FUV data); if a SN is
                excluded in one band, it is considered excluded overall
        sigma: float or list, confidence level required for detection (if 
                multiple, use multi-tier detection)
        count: list, number of points at or above associated sigma to count
                as a detection (same length as sigma)
    Output:
        list with injection parameters and recovery bool
    """

    # Unpack parameters
    tstart, scale = params

    # Inject all light curves
    for lc in lcs:
        inj = Injection(sn, lc, tstart, scale, **kwargs)
        recovered = inj.recover(sigma, count=count)
        # If either band is recovered, SN is considered recovered overall
        if recovered:
            break

    return [tstart, scale, recovered]


class Injection:
    def __init__(self, sn, lc, tstart, scale, twidth=WIDTH, decay_rate=DECAY_RATE,
            model='Chev94'):
        """Generate random model parameters, initialize model and inject into
        data.
        Inputs:
            sn: Supernova object associated with data
            lc: LightCurve object with data to be injected
            tstart: CSM model start time, int
            scale: model scale factor, float
            twidth: model width, int
            decay_rate: model decay rate, float
        """

        # Get data
        self.lc = lc
        self.time_col = 't_delta_rest'
        self.time = lc.data[self.time_col].copy()
        self.data_col = 'luminosity_hostsub'
        self.err_col = 'luminosity_hostsub_err'

        # Inject model
        self.model = CSMmodel(tstart, twidth, decay_rate, scale=scale, model=model)
        self.injection = lc.inject(self.model(self.time, sn.z)[lc.band], self.data_col)


    def __call__(self):
        return self.injection


    @classmethod
    def from_name(self, sn_name, band, tstart, scale, sn_info=[], 
                  model='Chev94', **kwargs):
        """Generate Injection instance from a supernova name and GALEX band,
        also creating a Supernova and LightCurve object in the process."""

        sn = Supernova(sn_name, sn_info=sn_info)
        lc = LightCurve(sn, band, sed=model)
        return Injection(sn, lc, tstart, scale, **kwargs)


    def recover(self, sigma, count=[1], dt_min=RECOV_MIN, detections=None, 
            plot=False):
        """Run detection algorithm on injected data and return points which 
        otherwise would not have been recovered.
        Inputs:
            sigma: float or list, confidence level required for detection (if 
                    multiple, use multi-tier detection)
            count: list, number of points at or above associated sigma to count
                    as a detection (same length as sigma)
            dt_min: minimum time since discovery to include detections
            detections: indices which were detected without injection; pass for
                    faster calcs
        Outputs:
            recovered: bool, whether injected light curve was recovered
        """

        # Convert count to list
        if type(count) == int:
            count = [count]

        # Run detections on original data
        if detections == None:
            detections = self.lc.detect_csm(sigma, count=count, dt_min=dt_min,
                    data_col=self.data_col, err_col=self.err_col)

        # Run detections on injected data
        recovery = self.lc.detect_csm(sigma, count=count, dt_min=dt_min, 
                data_col='%s_injected' % self.data_col, err_col=self.err_col)

        # Remove points that would have been detected either way
        recovery = recovery.drop(detections.index)
        recovered = len(recovery.index) > 0

        # Plot
        if plot:
            self.plot(recovery=recovery, detections=detections)

        return recovered


    def plot(self, recovery=[], detections=[]):
        """Basic plot of original vs injected data."""

        data = self.lc.data[self.data_col].copy()
        err = self.lc.data[self.err_col].copy()

        plt.errorbar(self.time, data, yerr=err, label='Original', 
                linestyle='none', marker='o')
        plt.errorbar(self.time, self.injection, yerr=err, label='Injected',
                linestyle='none', marker='o')
        plt.scatter(self.time[recovery.index], self.injection[recovery.index], 
                label='Newly recovered', marker='x', s=25, c='r', zorder=10)
        plt.scatter(self.time[detections.index], data[detections.index], 
                label='Original detection', marker='D', s=9, c='g', zorder=5)

        plt.xlim((0, None))
        plt.xlabel('Time since discovery [rest frame days]')
        plt.ylabel('Luminosity [erg s$^{-1}$ Å$^{-1}$]')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    dill.settings['recurse']=True

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', '-i', type=int, default=ITERATIONS, 
            help='Iterations')
    parser.add_argument('--tstart', '-s', default=[TSTART_MIN, TSTART_MAX], 
           nargs=2, type=int, help='Limits on CSM model start/end times')
    parser.add_argument('--scale', '-S', type=float, nargs=2, 
            default=[SCALE_MIN, SCALE_MAX], help='Scale factor limits')
    parser.add_argument('--twidth', '-w', help='Plateau width [days]', 
            default=WIDTH, type=float)
    parser.add_argument('--decay-rate', '-D', default=DECAY_RATE, type=float, 
            help='Fractional decay rate per 100 days')
    parser.add_argument('--overwrite', '-o', action='store_true', 
            help='Overwrite previous saves?')
    parser.add_argument('--model', '-m', type=str, default='Chev94', 
            help='CSM spectrum model to use')
    parser.add_argument('--sigma', type=int, nargs='+', default=SIGMA, 
            help='Detection confidence level (multiple for tiered detections)')
    parser.add_argument('--sigcount', type=int, nargs='+', default=SIGMA_COUNT,
            help='Number of points at corresponding sigma to count as detection')
    parser.add_argument('--savedir', type=str, default=SAVE_DIR,
            help='Recovery save directory, default mnt/d/injection_recovery_runs')
    args = parser.parse_args()

    # Save run parameters
    save_dir = run_dir('galex', args.model, args.sigma, parent=args.savedir)
    with open(save_dir / Path('_params.txt'), 'w') as file:
        file.write(str(args))

    # Adjust sigma count length
    sigma_count = args.sigcount[:len(args.sigma)]

    main(args.iterations, args.tstart, args.scale, save_dir, args.model, 
            twidth=args.twidth, decay_rate=args.decay_rate, 
            overwrite=args.overwrite, sigma=args.sigma, count=sigma_count)
