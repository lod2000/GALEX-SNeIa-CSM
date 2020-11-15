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


def main(iterations, tstart_lims, scale_lims, save_dir, twidth=WIDTH, 
        decay_rate=DECAY_RATE, overwrite=False, model='Chev94', sigma=SIGMA):

    # Import Graham data
    data = import_graham_data(sigma=sigma)
    supernovae = data.index

    # Record luminosities of CSM model for scale 1
    base_model = CSMmodel(0, WIDTH, DECAY_RATE, scale=1, model=model)
    base_model_z = 0.04
    scale1 = {'base_model_fuv_lum': base_model(0, base_model_z)['FUV'],
              'base_model_nuv_lum': base_model(0, base_model_z)['NUV'],
              'base_model_hst_lum': base_model(0, base_model_z)['F275W']}
    with open(save_dir / Path('_scale.txt'), 'w') as file:
        file.write(str(scale1))

    run_all(supernovae, data, iterations, tstart_lims, scale_lims, 
            overwrite=overwrite, model=model, save_dir=save_dir, twidth=twidth,
            decay_rate=decay_rate)


def import_graham_data(path='ref/Graham_limiting_magnitudes.csv', sigma=SIGMA):
    """Import data from Graham+ 2019 and convert 50% limiting magnitudes to
    (3)-sigma luminosity limits."""

    # Import CSV
    data = pd.read_csv(Path(path), index_col='Target')
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


def run_all(supernovae, data, iterations, tstart_lims, scale_lims, 
        overwrite=False, model='Chev94', save_dir=SAVE_DIR, **kwargs):
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
        supernovae = [s for s in supernovae if not check_save(s, iterations, save_dir)]
    
    for i, sn_name in enumerate(supernovae):
        print('\n%s [%s/%s]' % (sn_name, i+1, len(supernovae)))
        run_trials(sn_name, data, iterations, tstart_lims, scale_lims, 
                model=model, save_dir=save_dir, **kwargs)


def run_trials(sn_name, data, iterations, tstart_lims, scale_lims, save=True, 
        save_dir='', **kwargs):
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
    params = gen_params(iterations, tstart_lims, scale_lims, log=True)

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
        fname = sn2fname(sn_name, str(iterations)) # format sn_name-iterations.csv
        recovery_df.to_csv(save_dir / fname, index=False)

    return recovery_df


def inject_recover(params, nondetection, **kwargs):
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
    recovered = nondetection.inject_recover(tstart, scale, **kwargs)
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
        # Convert observed phase to rest-frame phase
        self.rest_phase = 1/(1+self.z) * self.phase
        self.luminosity_limit = data.loc[sn_name, 'Luminosity Limit']


    def inject_recover(self, tstart, scale, twidth=WIDTH, decay_rate=DECAY_RATE, 
            model='Chev94'):
        """
            tstart: CSM model start time, int
            scale: model scale factor, float
            twidth: model width, int
            decay_rate: model decay rate, float
        """

        # Inject model
        self.model = CSMmodel(tstart, twidth, decay_rate, scale=scale, model=model)
        self.injection = self.model(self.rest_phase, self.z)['F275W']

        # Recover
        self.recovered = self.injection > self.luminosity_limit
        return self.recovered


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
    parser.add_argument('--sigma', type=float, default=SIGMA, help='detection sigma level')
    args = parser.parse_args()

    # Save run parameters
    save_dir = run_dir('Graham', args.model, int(args.sigma))
    with open(save_dir / Path('_params.txt'), 'w') as file:
        file.write(str(args))

    main(args.iterations, args.tstart, args.scale, save_dir, args.twidth, 
            args.decay_rate, args.overwrite, args.model, args.sigma)