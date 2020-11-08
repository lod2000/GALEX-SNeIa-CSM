from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from pathos.multiprocessing import ProcessingPool as Pool
import dill
import json

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
                lc = LightCurve(sn, band, data_dir=DATA_DIR)
            except:
                # No data for this channel
                continue

            # Skip if no data after minimum recovery time
            if np.max(lc.data['t_delta_rest']) < RECOV_MIN:
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
            columns=['tstart', 'scale', 'recovered_times', 'all_times'])

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
        list with injection parameters, recovered times, and all times
    """

    # Unpack parameters
    tstart, scale = params

    # Inject all light curves
    recovered_times = []
    all_times = []
    for lc in lcs:
        inj = Injection(sn, lc, tstart, scale, **kwargs)
        inj.recover(sigma, count=count)
        recovered_times += inj.recovered_times
        all_times += inj.all_times

    # Remove duplicates
    recovered_times = sorted(list(dict.fromkeys(recovered_times)))
    all_times = sorted(list(dict.fromkeys(all_times)))

    return [tstart, scale, recovered_times, all_times]


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
        self.time = lc.data['t_delta_rest'].copy()
        self.data = lc.data['luminosity_hostsub'].copy()
        self.err = lc.data['luminosity_hostsub_err'].copy()

        # Inject model
        self.model = CSMmodel(tstart, twidth, decay_rate, scale=scale, model=model)
        self.injection = self.data + self.model(self.time, sn.z)[lc.band]


    def __call__(self):
        return self.injection


    @classmethod
    def from_name(self, sn_name, band, tstart, scale, sn_info=[], **kwargs):
        """Generate Injection instance from a supernova name and GALEX band,
        also creating a Supernova and LightCurve object in the process."""

        sn = Supernova(sn_name, sn_info=sn_info)
        lc = LightCurve(sn, band)
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
            recovered: indices of data recovered by detection algorithm
        """

        # Run detections on original data
        if detections == None:
            detections = detect_csm(self.time, self.data, self.err, sigma, 
                    count=count)

        # Run detections on injected data
        recovered = detect_csm(self.time, self.injection, self.err, sigma,
                count=count, dt_min=dt_min)

        # Remove points that would have been detected either way
        self.recovered = [r for r in recovered if r not in detections]
        self.recovered_times = self.time[self.recovered].to_list()

        # List of all times greater than dt_min
        self.all_times = self.time[self.time > dt_min].to_list()

        # Plot
        if plot:
            self.plot(recovered=self.recovered, detections=detections)

        return self.recovered


    def plot(self, recovered=[], detections=[]):
        """Basic plot of original vs injected data."""

        plt.errorbar(self.time, self.data, yerr=self.err, label='Original', 
                linestyle='none', marker='o')
        plt.errorbar(self.time, self.injection, yerr=self.err, label='Injected',
                linestyle='none', marker='o')
        plt.scatter(self.time[recovered], self.injection[recovered], 
                label='Newly recovered', marker='x', s=25, c='r', zorder=10)
        plt.scatter(self.time[detections], self.data[detections], 
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
    args = parser.parse_args()

    # Save run parameters
    sigma_str = ''.join([str(s) for s in args.sigma])
    save_dir = SAVE_DIR / Path('%s_%ssigma' % (args.model, sigma_str))
    if not save_dir.is_dir():
        save_dir.mkdir()
    with open(save_dir / Path('_params.txt'), 'w') as file:
        file.write(str(args))

    # Adjust sigma count length
    sigma_count = args.sigcount[:len(args.sigma)]

    main(args.iterations, args.tstart, args.scale, save_dir, args.model, 
            twidth=args.twidth, decay_rate=args.decay_rate, 
            overwrite=args.overwrite, sigma=args.sigma, count=sigma_count)