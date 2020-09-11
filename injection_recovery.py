from tqdm import tqdm
import itertools
# from multiprocessing import Pool
from functools import partial
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys
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
DECAY_RATE = 0.3
RECOV_MIN = 50 # minimum number of days after discovery to count as recovery
SIGMA = 3
WIDTH = 250 # days, from PTF11kx

SAVE_DIR = Path('save')

def main(iterations, overwrite=False):

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')

    recovery_df = run_trials('SN2007on', 'NUV', 10000, (0, 1000), (0.5, 2), 
            [5, 3], [1, 3], sn_info=sn_info)
    print(recovery_df)


def run_trials(sn_name, band, iterations, tstarts, scales, sigma, count, 
        sn_info=[], save=True):
    """Run injection recovery a given number of times on one supernova.
    Inputs:
        sn_name: supernova name
        band: GALEX band 'FUV' or 'NUV'
        iterations: iterations of injection & recovery
        tstarts: (min, max) tuple of tstart parameter bounds
        scales: (min, max) tuple of scale parameter bounds
        sigma: float or list, confidence level required for detection (if 
                multiple, use multi-tier detection)
        count: list, number of points at or above associated sigma to count
                as a detection (same length as sigma)
        sn_info: supernova info data frame
        save: bool, output recovery_df to CSV
    Outputs:
        recovery_df: DataFrame of injection parameters and recovered times
    """

    # Initialize Supernova and LightCurve objects
    sn = Supernova(sn_name, sn_info=sn_info)
    lc = LightCurve(sn, band)

    # Run injection-recovery trials in parallel
    recovery_df = []
    with Pool() as pool:
        func = partial(inject_recover, sn=sn, lc=lc, tstarts=tstarts, 
                scales=scales, sigma=sigma, count=count)
        imap = pool.imap(func, list(range(iterations)), chunksize=100)
        for recovery in tqdm(imap, total=iterations):
            recovery_df.append(recovery)

    recovery_df = pd.DataFrame(recovery_df, 
            columns=['tstart', 'scale', 'recovered_times', 'all_times'])

    if save:
        fname = sn2fname(sn_name, band, suffix='-%s.csv' % iterations)
        recovery_df.to_csv(SAVE_DIR / fname, index=False)

    return recovery_df


def inject_recover(i, sn, lc, tstarts, scales, sigma, count):
    """Perform injection and recovery for given SN and model parameters.
    Inputs:
        i: dummy argument for pool.imap
        sn: Supernova object
        lc: LightCurve object
        tstarts: (min, max) tuple of tstart parameter bounds
        scales: (min, max) tuple of scale parameter bounds
        sigma: float or list, confidence level required for detection (if 
                multiple, use multi-tier detection)
        count: list, number of points at or above associated sigma to count
                as a detection (same length as sigma)
    Output:
        list with injection parameters and recovered times
    """

    inj = Injection(sn, lc, tstarts, scales)
    inj.recover(sigma, count=count)
    # recover(inj, sigma, count=count)
    return [inj.tstart, inj.scale, inj.recovered_times, inj.all_times]


class Injection:
    def __init__(self, sn, lc, tstarts, scales, width=WIDTH, decay=DECAY_RATE):
        """Generate random model parameters, initialize model and inject into
        data.
        Inputs:
            sn: Supernova object associated with data
            lc: LightCurve object with data to be injected
            tstarts: (min, max) tuple of tstart parameter bounds
            scales: (min, max) tuple of scale parameter bounds
            width: model width, int
            decay: model decay rate, float
        """
        
        # Generate random parameters
        self.tstart = random.randint(tstarts[0], tstarts[1])
        self.scale = random.uniform(scales[0], scales[1])

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


    def recover(self, sigma, count=[1], dt_min=50, detections=None, plot=False):
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
    parser.add_argument('--iterations', '-i', type=int, help='Iterations')
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite saves')
    args = parser.parse_args()

    main(args.iterations, args.overwrite)