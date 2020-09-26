from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.random import default_rng
from pathlib import Path
import random
from pathos.multiprocessing import ProcessingPool as Pool
import dill

from utils import *
from CSMmodel import CSMmodel
from light_curve import LightCurve
from supernova import Supernova

# Default values
DECAY_RATE = 0.3 # CSM curve decay factor
WIDTH = 250 # days, from PTF11kx
SIGMA = [5, 3] # detection certainty
SIGMA_COUNT = [1, 3] # Number of points at corresponding sigma to detect


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

    # Remove SNe with previous save files from list
    if not overwrite:
        supernovae = [s for s in supernovae if not check_save(s, iterations)]
    
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

        run_trials(sn, lcs, iterations, **kwargs)


def check_save(sn_name, iterations, save_dir=SAVE_DIR):
    """Checks if save file exists for given SN and iterations."""

    save_file = sn2fname(sn_name, str(iterations), parent=save_dir)
    return save_file.is_file()


# def get_data_list(supernovae, iterations, save_dir=SAVE_DIR, data_dir=DATA_DIR, 
#         overwrite=False):
#     """Return list of light curve files corresponding to given supernovae, and
#     remove SNe from list with previous save files, unless overwrite is True.
#     Output:
#         combos: list of (sn_name, band) tuples
#     """

#     # Combine list of bands and list of supernovae
#     bands = ['FUV', 'NUV']
#     supernovae = [sn_name for sn_name in supernovae for b in bands]
#     bands = bands * int(len(supernovae) / len(bands))
#     combos = list(zip(supernovae, bands))
#     # combos = [[s, [b for b in bands if sn2fname(s, b, parent=data_dir).is_file()]] 
#     #         for s in supernovae]

#     # Remove combinations with previously saved files, unless overwrite
#     if not overwrite:
#         combos = [c for c in combos if not (Path(save_dir) / sn2fname(c[0], c[1], 
#                 suffix='-%s.csv' % iterations)).is_file()]
#         # combos = [c for c in combos if not sn2fname(c[0][0], c)]

#     # Remove combinations without data files
#     combos = [c for c in combos if (Path(data_dir) / sn2fname(c[0], c[1])).is_file()]

#     return combos


def run_trials(sn, lcs, iterations, save=True, sn_info=[], **kwargs):
    """Run injection recovery a given number of times on one supernova.
    Inputs:
        sn_name: supernova name
        iterations: iterations of injection & recovery
        save: bool, output recovery_df to CSV
        sn_info: supernova info reference DataFrame
        kwargs: keyword arguments for inject_recover
    Outputs:
        recovery_df: DataFrame of injection parameters and recovered times
    """

    # Random injection parameter sample
    params = gen_params(iterations, TSTART_MIN, TSTART_MAX, SCALE_MIN, SCALE_MAX)

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
        recovery_df.to_csv(SAVE_DIR / fname, index=False)

    return recovery_df


def gen_params(iterations, tstart_min, tstart_max, scale_min, scale_max):
    """Generate random injection-recovery parameters."""

    rng = default_rng()
    tstart = rng.integers(tstart_min, tstart_max, iterations, endpoint=True)
    scale = rng.uniform(scale_min, scale_max, iterations)
    params = np.column_stack((tstart, scale))

    return params


def inject_recover(params, sn, lcs, sigma=SIGMA, count=SIGMA_COUNT):
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
        inj = Injection(sn, lc, tstart, scale)
        inj.recover(sigma, count=count)
        recovered_times += inj.recovered_times
        all_times += inj.all_times

    # Remove duplicates
    recovered_times = sorted(list(dict.fromkeys(recovered_times)))
    all_times = sorted(list(dict.fromkeys(all_times)))

    return [tstart, scale, recovered_times, all_times]


class Injection:
    def __init__(self, sn, lc, tstart, scale, width=WIDTH, decay=DECAY_RATE):
        """Generate random model parameters, initialize model and inject into
        data.
        Inputs:
            sn: Supernova object associated with data
            lc: LightCurve object with data to be injected
            tstart: CSM model start time, int
            scale: model scale factor, float
            width: model width, int
            decay: model decay rate, float
        """

        # Get data
        self.time = lc.data['t_delta_rest'].copy()
        self.data = lc.data['luminosity_hostsub'].copy()
        self.err = lc.data['luminosity_hostsub_err'].copy()

        # Inject model
        self.model = CSMmodel(tstart, width, decay, scale=scale)
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
    parser.add_argument('--iterations', '-i', type=int, default=10000, help='Iterations')
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite saves')
    args = parser.parse_args()

    main(args.iterations, args.overwrite)