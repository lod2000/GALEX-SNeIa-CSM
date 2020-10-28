from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from pathos.multiprocessing import ProcessingPool as Pool
import dill
import json
import astropy.units as u

from utils import *
from CSMmodel import CSMmodel
from light_curve import flux2luminosity

# Default values & constants
SIGMA = 3 # detection certainty
z_2015cp = 0.038 # redshift of SN 2015cp
F275W_ZERO_POINT = 1.47713e-8 # erg/cm2/s/A; AB system

# Directories
SAVE_DIR = Path('Graham/save')
OUTPUT_DIR = Path('Graham/out')
DATA_DIR = Path('Graham/data')


def main(iterations, overwrite=False, model='Chev94'):

    # Import Graham data
    data = import_graham_data()
    supernovae = data.index

    # Save run parameters
    base_model = CSMmodel(0, WIDTH, DECAY_RATE, scale=1, model=model)
    base_model_hst_lum = base_model(0, z_2015cp)['F275W'] # NUV luminosity for scale 1
    params = {'iterations': iterations,
              'decay_rate': DECAY_RATE,
              'width': WIDTH,
              'sigma': SIGMA,
              'model': model,
              'base_model_hst_lum': base_model_hst_lum}
    with open(SAVE_DIR / Path(model) / Path('_params.txt'), 'w') as file:
        file.write(json.dumps(params))

    run_all(supernovae, data, iterations, overwrite=overwrite, model=model)


def import_graham_data(fname='limiting_magnitudes.csv', data_dir=DATA_DIR, sigma=SIGMA):
    """Import data from Graham+ 2019 and convert 50% limiting magnitudes to
    (3)-sigma luminosity limits."""

    # Import CSV
    data = pd.read_csv(DATA_DIR / Path(fname), index_col='Target')
    # Convert limiting magnitude & error to n-sigma magnitude limit
    data['Sigma Limit'] = data['Limiting Magnitude'] - sigma * data['Limiting Magnitude Error']
    # Convert magnitude limit to flux limit
    data['Flux Limit'] = 10**(-2/5 * data['Sigma Limit']) * F275W_ZERO_POINT
    # Gather distance and redshift data for targets
    dist = data['Distance [Mpc]'].to_numpy() * u.Mpc
    z = data['Redshift']
    # Convert flux limit to luminosity limit
    data['Luminosity Limit'] = data['Flux Limit'] * (4*np.pi*dist.to('cm')**2) * (1+z)**3
    return data


def run_all(supernovae, data, iterations, overwrite=False, model='Chev94', **kwargs):
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
        supernovae = [s for s in supernovae if not check_save(s, iterations, model, save_dir=SAVE_DIR)]
    
    for i, sn_name in enumerate(supernovae):
        print('\n%s [%s/%s]' % (sn_name, i+1, len(supernovae)))
        run_trials(sn_name, data, iterations, model=model, **kwargs)


def run_trials(sn_name, data, iterations, save=True, model='Chev94', **kwargs):
    """Run injection recovery a given number of times on one supernova.
    Inputs:
        sn_name: supernova name
        data: DataFrame of Graham nondetection limits
        iterations: iterations of injection & recovery
        save: bool, output recovery_df to CSV
        kwargs: keyword arguments for inject_recover
    Outputs:
        recovery_df: DataFrame of injection parameters and recovered times
    """

    nondetection = Nondetection(sn_name, data)

    # Random injection parameter sample
    params = gen_params(iterations, TSTART_MIN, TSTART_MAX, SCALE_MIN, SCALE_MAX)

    # Run injection-recovery trials in parallel
    recovery_df = []
    with Pool() as pool:
        func = partial(inject_recover, nondetection=nondetection, **kwargs)
        imap = pool.imap(func, params, chunksize=100)
        for recovery in tqdm(imap, total=iterations):
            recovery_df.append(recovery)

    recovery_df = pd.DataFrame(recovery_df, 
            columns=['tstart', 'scale', 'recovered_times', 'all_times'])

    # Save CSV
    if save:
        save_dir = SAVE_DIR / Path(model)
        fname = sn2fname(sn_name, str(iterations)) # format sn_name-iterations.csv
        recovery_df.to_csv(save_dir / fname, index=False)

    return recovery_df


def inject_recover(params, nondetection, model='Chev94'):
    """Perform injection and recovery for given SN and model parameters.
    Inputs:
        params: tuple of (tstart, scale) injection parameters
        nondetection: Nondetection object
    Output:
        list with injection parameters and number of recoveries
    """

    # Unpack parameters
    tstart, scale = params
    # Inject & recover
    recovered = nondetection.inject_recover(tstart, scale, model=model)
    recovered_times = [nondetection.rest_phase] if recovered else []
    all_times = [nondetection.rest_phase]

    return [tstart, scale, recovered_times, all_times]


class Nondetection:
    def __init__(self, sn_name, data, sigma=3):
        """Initialize model and inject into data.
        Inputs:
            sn_name: supernova name
            data: DataFrame of Graham nondetection limits
        """

        # Get data
        self.sn_name = sn_name
        self.z = data.loc[sn_name, 'Redshift']
        self.phase = data.loc[sn_name, 'Phase']
        self.rest_phase = 1/(1+self.z) * self.phase
        self.luminosity_limit = data.loc[sn_name, 'Luminosity Limit']
        # limiting_mag = data.loc[sn_name, '50% Limiting Magnitude']
        # luminosity_limit = data.loc[sn_name, '50% Luminosity Limit [erg/s/A]']
        # self.luminosity_limit = self.get_luminosity_limit(luminosity_limit, sigma)


    def inject_recover(self, tstart, scale, width=WIDTH, decay=DECAY_RATE, model='Chev94'):
        """
            tstart: CSM model start time, int
            scale: model scale factor, float
            width: model width, int
            decay: model decay rate, float
        """

        # Inject model
        self.model = CSMmodel(tstart, width, decay, scale=scale, model=model)
        self.injection = self.model(self.rest_phase, self.z)['F275W']

        # Recover
        self.recovered = self.injection > self.luminosity_limit
        return self.recovered


    def get_luminosity_limit(self, limiting_mag, sigma):
        """Convert 50% limiting magnitude to any sigma upper limit for luminosity."""

        # Temporary: use 50% luminosity limit instead
        luminosity_limit = limiting_mag * (sigma / 0.675)

        return luminosity_limit


if __name__ == '__main__':
    dill.settings['recurse']=True

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', '-i', type=int, default=10000, help='Iterations')
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite saves')
    parser.add_argument('--model', '-m', type=str, default='Chev94', help='CSM model spectrum')
    args = parser.parse_args()

    main(args.iterations, args.overwrite, args.model)