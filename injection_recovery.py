from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
# import sys
import pandas as pd
import numpy as np
from pathlib import Path
import random
from pathos.multiprocessing import ProcessingPool as Pool
import dill

from utils import *
from CSMmodel import CSMmodel
from light_curve import LightCurve
from supernova import Supernova

# Default values
TSTART_MIN = 0
TSTART_MAX = 1000
SCALE_MIN = 0.5
SCALE_MAX = 2.
DECAY_RATE = 0.3 # CSM curve decay factor
WIDTH = 250 # days, from PTF11kx
RECOV_MIN = 50 # minimum number of days after discovery to count as recovery
SIGMA = [5, 3] # detection certainty
SIGMA_COUNT = [1, 3] # Number of points at corresponding sigma to detect

SAVE_DIR = Path('save')
DATA_DIR = Path('data')

def main(iterations, overwrite=False):

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')

    supernovae = sn_info.sort_values('pref_dist').index
    run_all(supernovae, iterations, sn_info=sn_info, overwrite=overwrite)


def run_all(supernovae, iterations, sn_info=[], overwrite=False, **kwargs):
    """Run injection recovery trials on all supernovae in given list.
    Inputs:
        supernovae: list of supernova names
        iterations: iterations of injection & recovery for each SN
        sn_info: supernova info reference DataFrame
        overwrite: bool, whether to overwrite previous save files
        kwargs: keyword arguments for run_trials
    """

    combos = get_data_list(supernovae, iterations, overwrite=overwrite)
    
    for i, (sn_name, band) in enumerate(combos):
        # if not overwrite and sn2fname(sn_name, band, suffix='-%s.csv' % iterations)
        print('\n%s - %s [%s/%s]' % (sn_name, band, i+1, len(combos)))
        try:
            # Initialize Supernova and LightCurve objects
            sn = Supernova(sn_name, sn_info=sn_info)
            lc = LightCurve(sn, band, data_dir=DATA_DIR)
        except:
            # If there's some problem with the light curve data, skip
            print('\tproblem with the data!')
            continue

        run_trials(sn, lc, iterations, **kwargs)


def get_data_list(supernovae, iterations, save_dir=SAVE_DIR, data_dir=DATA_DIR, 
        overwrite=False):
    """Returns list of light curve files corresponding to given supernovae, and
    removes SNe from list with previous save files, unless overwrite is True.
    Output:
        combos: list of (sn_name, band) tuples
    """

    # Combine list of bands and list of supernovae
    bands = ['FUV', 'NUV']
    supernovae = [sn_name for sn_name in supernovae for b in bands]
    bands = bands * int(len(supernovae) / len(bands))
    combos = list(zip(supernovae, bands))

    # Remove combinations with previously saved files, unless overwrite
    if not overwrite:
        combos = [c for c in combos if not (save_dir / sn2fname(c[0], c[1], 
                suffix='-%s.csv' % iterations)).is_file()]

    # Remove combinations without data files
    combos = [c for c in combos if (data_dir / sn2fname(c[0], c[1])).is_file()]

    return combos


def run_trials(sn, lc, iterations, save=True, **kwargs):
    """Run injection recovery a given number of times on one supernova.
    Inputs:
        sn_name: supernova name
        band: GALEX band 'FUV' or 'NUV'
        iterations: iterations of injection & recovery
        sn_info: supernova info data frame
        save: bool, output recovery_df to CSV
        kwargs: keyword arguments for inject_recover
    Outputs:
        recovery_df: DataFrame of injection parameters and recovered times
    """

    # Run injection-recovery trials in parallel
    recovery_df = []
    with Pool() as pool:
        func = partial(inject_recover, sn=sn, lc=lc, **kwargs)
        imap = pool.imap(func, list(range(iterations)), chunksize=100)
        for recovery in tqdm(imap, total=iterations):
            recovery_df.append(recovery)

    recovery_df = pd.DataFrame(recovery_df, 
            columns=['tstart', 'scale', 'recovered_times', 'all_times'])

    # Save CSV
    if save:
        fname = sn2fname(sn.name, lc.band, suffix='-%s.csv' % iterations)
        recovery_df.to_csv(SAVE_DIR / fname, index=False)

    return recovery_df


def inject_recover(i, sn, lc, tstart_min=TSTART_MIN, tstart_max=TSTART_MAX, 
        scale_min=SCALE_MIN, scale_max=SCALE_MAX, sigma=SIGMA, count=SIGMA_COUNT):
    """Perform injection and recovery for given SN and model parameters.
    Inputs:
        i: dummy argument for pool.imap
        sn: Supernova object
        lc: LightCurve object
        tstart_min, tstart_max: tstart parameter bounds
        scale_min, scale_max: scale parameter bounds
        sigma: float or list, confidence level required for detection (if 
                multiple, use multi-tier detection)
        count: list, number of points at or above associated sigma to count
                as a detection (same length as sigma)
    Output:
        list with injection parameters, recovered times, and all times
    """

    inj = Injection(sn, lc, tstart_min, tstart_max, scale_min, scale_max)
    inj.recover(sigma, count=count)
    return [inj.tstart, inj.scale, inj.recovered_times, inj.all_times]


class Injection:
    def __init__(self, sn, lc, tstart_min, tstart_max, scale_min, scale_max, 
            width=WIDTH, decay=DECAY_RATE):
        """Generate random model parameters, initialize model and inject into
        data.
        Inputs:
            sn: Supernova object associated with data
            lc: LightCurve object with data to be injected
            tstart_min, tstart_max: tstart parameter bounds
            scale_min, scale_max: scale parameter bounds
            width: model width, int
            decay: model decay rate, float
        """
        
        # Generate random parameters
        self.tstart = random.randint(tstart_min, tstart_max)
        self.scale = random.uniform(scale_min, scale_max)

        # Other parameters
        self.width = width
        self.decay = decay

        # Get data
        self.time = lc.data['t_delta_rest'].copy()
        self.data = lc.data['luminosity_hostsub'].copy()
        self.err = lc.data['luminosity_hostsub_err'].copy()

        # Inject model
        self.model = CSMmodel(self.tstart, self.width, self.decay, 
                scale=self.scale)
        self.injection = self.data + self.model(self.time, sn.z)[lc.band]


    def __call__(self):
        return self.injection


    @classmethod
    def from_name(self, sn_name, band, tstarts, scales, sn_info=[], **kwargs):
        """Generate Injection instance from a supernova name and GALEX band,
        also creating a Supernova and LightCurve object in the process."""

        sn = Supernova(sn_name, sn_info=sn_info)
        lc = LightCurve(sn, band)
        return Injection(sn, lc, tstarts, scales, **kwargs)


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
    parser.add_argument('--iterations', '-i', type=int, default=10000, help='Iterations')
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite saves')
    args = parser.parse_args()

    main(args.iterations, args.overwrite)