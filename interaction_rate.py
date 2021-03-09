import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from astropy.stats import binom_conf_interval
from utils import *

# Defaults
TSTART = [100, 1000]
SCALE = [0.9, 1.1]
CONF = 0.9
ITER = 10000 # injection iterations


def main(tstart, scale, model='Chev94', sigma=3, iterations=ITER, conf=CONF):
    """Print binomial confidence interval for CSM interaction rate within
    given parameter bounds.
    Inputs:
        tstart: tuple, CSM model interaction start time bounds
        scale: tuple, CSM model scale factor bounds
        model: 'Chev94' or 'flat', spectral model
    """

    # Initialize DataFrame
    rate_df = pd.DataFrame([], index=['GALEX', 'G19', 'All UV'], 
            columns=['Detections', 'Trials', 'Lower Limit [%]', 'Upper Limit [%]'])

    # Get save directories
    galex_save_dir = run_dir('galex', model, sigma, detections=False)
    graham_save_dir = run_dir('graham', model, sigma, detections=False)
    graham_det_dir = run_dir('graham', model, sigma, detections=True)

    # Successes and trials
    rate_df.loc['GALEX', 'Trials'] = count_recovered_sne(galex_save_dir, tstart, 
            scale, iterations)
    rate_df.loc['GALEX', 'Detections'] = 0
    graham_detections = count_recovered_sne(graham_det_dir, tstart, scale, 
            iterations)
    graham_nondetections = count_recovered_sne(graham_save_dir, tstart, scale, 
            iterations)
    rate_df.loc['G19', 'Detections'] = graham_detections
    rate_df.loc['G19', 'Trials'] = graham_detections + graham_nondetections
    rate_df.loc['All UV'] = np.sum(rate_df.loc[['GALEX', 'G19']])

    # Calculate binomial confidence interval
    # bci = 100 * binom_conf_interval(rate_df['Detections'], rate_df['Trials'], 
    #         confidence_level=conf, interval='jeffreys')
    for study in rate_df.index:
        detections = rate_df.loc[study, 'Detections']
        trials = rate_df.loc[study, 'Trials']
        if trials >= 1:
            bci = 100 * binom_conf_interval(detections, trials, 
                confidence_level=conf, interval='jeffreys')
            rate_df.loc[study, ['Lower Limit [%]', 'Upper Limit [%]']] = bci.T
        else:
            rate_df.loc[study, 'Lower Limit [%]'] = np.nan
            rate_df.loc[study, 'Upper Limit [%]'] = np.nan
    # bci_lower, bci_upper = bci_nan(rate_df[['Detections']], rate_df[['Trials']])
    # rate_df['Lower Limit [%]'] = bci_lower
    # rate_df['Upper Limit [%]'] = bci_upper

    print('\nConfidence intervals for %s < tstart < %s, ' % tstart + 
            '%s < S < %s using the %s model' % (scale + (model,)))
    print(rate_df)


def count_recovered_sne(save_dir, tstart, scale, iterations=ITER):
    """Count recovered SNe from injection-recovery run within given bounds.
    Inputs:
        save_dir: Path, directory containing recovery save files
        tstart: tuple, CSM model interaction start time bounds
        scale: tuple, CSM model scale factor bounds
    Outputs:
        recovered_sne: float, sum of recovery rates, rounded to nearest whole
    """

    print('Importing recovery save files from %s' % save_dir)
    save_files = list(Path(save_dir).glob('*-%s.csv' % iterations))
    recovered_sne = 0
    with Pool() as pool:
        func = partial(get_recovery_rate, tstart=tstart, scale=scale)
        imap = pool.imap(func, save_files, chunksize=10)
        for r in tqdm(imap, total=len(save_files)):
            recovered_sne = np.nansum([r, recovered_sne])

    return np.around(recovered_sne)


def get_recovery_rate(save_file, tstart, scale):
    """Import injection-recovery save file and calculate recovery rate.
    Inputs:
        tstart: tuple, CSM model interaction start time bounds
        scale: tuple, CSM model scale factor bounds
    Output:
        recovery_rate: ratio of N(recoveries) / N(injections)
    """

    # Separate parameter bounds
    tstart_min, tstart_max = tstart
    scale_min, scale_max = scale
    # Import save file
    df = pd.read_csv(save_file)
    # Exclude data outside range
    df = df[(df['tstart'] >= tstart_min) & (df['tstart'] < tstart_max)]
    df = df[(df['scale'] >= scale_min) & (df['scale'] < scale_max)]
    # Calculate recovery rate
    recovered = df[df['recovered']]
    # Allow NaN in case no samples are present in bin
    recovery_rate = np.divide(recovered.shape[0], df.shape[0])

    return recovery_rate


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tstart', '-t', type=int, nargs=2, default=TSTART, 
            help='Range of CSM model interaction start times')
    parser.add_argument('--scale', '-S', type=float, nargs=2, default=SCALE, 
            help='Range of CSM model scale factors')
    parser.add_argument('--model', '-m', type=str, default='Chev94', 
            help='spectral model type ("Chev94" or "flat", default "Chev94")')
    parser.add_argument('--conf', '-c', type=float, default=CONF,
            help='Confidence level of binomial confidence interval, default 0.9')
    parser.add_argument('--iterations', '-i', type=int, default=ITER,
            help='Number of injection iterations in save files, default 10000')
    args = parser.parse_args()

    main(tuple(args.tstart), tuple(args.scale), model=args.model, 
            iterations=args.iterations, conf=args.conf)