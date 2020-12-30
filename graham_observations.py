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
from light_curve import freq2wavelength

# Default values & constants
SIGMA = 3 # detection certainty
F275W_ZERO_POINT = 1.47713e-8 # erg/cm2/s/A; AB system


def main(iterations, tstart_lims, scale_lims, save_dir, twidth=WIDTH, 
        decay_rate=DECAY_RATE, overwrite=False, model='Chev94', sigma=SIGMA,
        detections=False):

    # Import Graham data
    data = pd.read_csv(Path('ref/Graham_observations.csv'), index_col=0)
    # Remove detections and/or nondetections
    if detections:
        data = data[data['Detection']]
    else:
        data = data[~data['Detection']]
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

    # obs = GrahamObservation('iPTF14aqs', data, sigma=3)
    # print(obs.luminosity_limit)


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

    obs = GrahamObservation(sn_name, data, sigma=3)

    # Random injection parameter sample
    params = gen_params(iterations, tstart_lims, scale_lims, log=True)

    # Run injection-recovery trials in parallel
    recovery_df = []
    with Pool() as pool:
        func = partial(inject_recover, obs=obs, **kwargs)
        imap = pool.imap(func, params, chunksize=100)
        for recovery in tqdm(imap, total=iterations):
            recovery_df.append(recovery)

    recovery_df = pd.DataFrame(recovery_df, 
            columns=['tstart', 'scale', 'recovered'])

    # Save CSV
    if save:
        fname = sn2fname(sn_name, str(iterations)) # format sn_name-iterations.csv
        recovery_df.to_csv(save_dir / fname, index=False)

    return recovery_df


def inject_recover(params, obs, **kwargs):
    """Perform injection and recovery for given SN and model parameters.
    Inputs:
        params: tuple of (tstart, scale) injection parameters
        obs: GrahamObservation object
    Output:
        list with injection parameters and number of recoveries
    """

    # Unpack parameters
    tstart, scale = params
    # Inject & recover
    recovered = obs.inject_recover(tstart, scale, **kwargs)

    return [tstart, scale, recovered]


class GrahamObservation:
    def __init__(self, sn_name, data, sigma=3):
        """Initialize observation.
        Inputs:
            sn_name: supernova name
            data: DataFrame of Graham nondetection limits
        """

        # Get data
        self.sn_name = sn_name
        self.info = data.loc[sn_name]

        self.z = self.info['Redshift']
        self.dist = self.info['Distance [Mpc]'] * u.Mpc
        self.phase = self.info['Phase']
        # Convert observed phase to rest-frame phase
        self.rest_phase = 1/(1+self.z) * self.phase
        # 50% limiting mag
        self.mag_lim = self.info['Limiting Magnitude']
        self.mag_lim_err = self.info['Limiting Magnitude Error']
        # Detection or nondetection
        self.detection = self.info['Detection']
        # Luminosity limit
        self.luminosity_limit = self.get_luminosity_limit(sigma)

        # Detections
        if self.detection:
            luminosity_hz = 10 ** self.info['Log Luminosity'] * u.erg
            # luminosity (erg/s/AA)
            self.luminosity = freq2wavelength(luminosity_hz, F275W_LAMBDA_EFF * u.AA)
            self.luminosity_err = self.get_luminosity_limit(1)


    def get_luminosity_limit(self, sigma, zero_point=F275W_ZERO_POINT):
        """Calculate upper limit on luminosity based on limiting magnitude."""

        # Calculate lower magnitude limit
        sigma_lim = self.mag_lim - sigma * self.mag_lim_err
        # Convert magnitude limit to flux limit
        flux_lim = 10 ** (-2/5 * sigma_lim) * zero_point * u.erg/u.cm**2/u.s/u.AA
        # Convert flux limit to luminosity limit
        return flux_lim * (4 * np.pi * self.dist.to('cm')**2) * (1 + self.z)**3


    def inject_recover(self, tstart, scale, twidth=WIDTH, decay_rate=DECAY_RATE, 
            model='Chev94'):
        """
        Inject CSM model light curve and recover detection
        Inputs:
            tstart: CSM model start time, int
            scale: model scale factor, float
            twidth: model width, int
            decay_rate: model decay rate, float
        Output:
            recovered: bool
        """

        # Inject model
        model = CSMmodel(tstart, twidth, decay_rate, scale=scale, model=model)
        model_lum = model(self.rest_phase, self.z)['F275W']

        self.recovered = model_lum > self.luminosity_limit.value

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
    parser.add_argument('-d', '--detections', action='store_true', 
            help='Recover models for G19 detections instead of nondetections')
    args = parser.parse_args()

    # Save run parameters
    study = 'Graham'
    if args.detections:
        study += '_det'
    save_dir = run_dir(study, args.model, int(args.sigma))
    with open(save_dir / Path('_params.txt'), 'w') as file:
        file.write(str(args))

    main(args.iterations, args.tstart, args.scale, save_dir, args.twidth, 
            args.decay_rate, args.overwrite, args.model, args.sigma,
            args.detections)