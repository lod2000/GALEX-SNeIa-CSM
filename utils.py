from pathlib import Path
import platform
import pandas as pd
import numpy as np

# Default values
RECOV_MIN = 50 # minimum number of days after discovery to count as recovery
TSTART_MIN = 0
TSTART_MAX = 2500
SCALE_MIN = 0.01
SCALE_MAX = 100
DECAY_RATE = 0.3 # CSM curve decay factor
WIDTH = 250 # days, from PTF11kx
ITERATIONS = 10000 # injection-recovery iterations per SN
TSTART_BIN_WIDTH = 100 # horizontal bin width for recovery plot

# Default directories
# SAVE_DIR = Path('/mnt/d/injection_recovery_runs')
SAVE_DIR = Path('save')
DATA_DIR = Path('data')
OUTPUT_DIR = Path('out')

# Reference files
OSC_FILE = Path('ref/osc.csv')

# Plot color palette
COLORS = {'FUV' : '#a37', 'NUV' : '#47a', # GALEX
          'UVW1': '#cb4', 'UVM2': '#283', 'UVW2': '#6ce', # Swift
          'F275W': '#e67' # Hubble
          }

F275W_LAMBDA_EFF = 2714.65
L_2015cp = 7.6e25 # erg/s/Hz (Graham+ 2019)
L_2015cp_cgs = L_2015cp * (3e18) / (F275W_LAMBDA_EFF**2) # luminosity of 2015cp, erg/s/A
Z_2015cp = 0.0413


def fname2sn(fname):
    """Extract SN name and band from a file name."""

    fname = Path(fname)
    # Remove extension(s)
    suffixes = ''.join(fname.suffixes)
    split = fname.name.replace(suffixes, '').split('-')
    sn_name = '-'.join(split[:-1])
    band = split[-1]
    # Windows replaces : with _ in some file names
    if 'CSS' in sn_name or 'MLS' in sn_name:
        sn_name = sn_name.replace('_', ':', 1)
    sn_name = sn_name.replace('_', ' ')
    return sn_name, band


def sn2fname(sn_name, band, suffix='.csv', parent=None):
    """Convert SN name and GALEX band to a file name, e.g. for a light curve CSV."""

    fname = '-'.join((sn_name, band)) + suffix
    fname = fname.replace(' ', '_')
    # Make Windows-friendly
    if (platform.system() == 'Windows') or ('Microsoft' in platform.release()):
        fname = fname.replace(':', '_')
    # Include parent dir
    if parent:
        return Path(parent) / Path(fname)
    return Path(fname)


def check_save(sn_name, iterations, save_dir=SAVE_DIR):
    """Checks if save file exists for given SN and iterations."""

    save_file = sn2fname(sn_name, str(iterations), parent=save_dir)
    return save_file.is_file()


def gen_params(iterations, tstart_lims, scale_lims, log=True):
    """Generate random injection-recovery parameters."""

    from numpy.random import default_rng

    rng = default_rng()
    tstart = rng.integers(tstart_lims[0], tstart_lims[1], iterations, endpoint=True)
    if log:
        scale = rng.uniform(np.log10(scale_lims[0]), np.log10(scale_lims[1]), iterations)
        scale = 10 ** scale
    else:
        scale = rng.uniform(scale_lims[0], scale_lims[1], iterations)
    params = np.column_stack((tstart, scale))

    return params


def run_dir(study, model, sigma, detections=False, parent=SAVE_DIR):
    """Generate run save directory."""

    if type(sigma) == list:
        sigma = ''.join([str(s) for s in sigma])
    if detections:
        study = study + '_det'
    run_name = '%s_%s_%ssigma' % (study, model, sigma)
    run = Path(parent) / Path(run_name)
    if not run.is_dir(): run.mkdir()

    return run


def SN2015cp_scale(model):
    """Get scale factor corresponding to UV luminosity of SN 2015cp (Graham+ 2019)."""

    from CSMmodel import CSMmodel

    # Corrective scale factor: S=1 corresponds to 2015cp
    csm_model = CSMmodel(tstart=0, twidth=100, decay_rate=0.3, scale=1., model=model)
    model_2015cp = csm_model([0], Z_2015cp)
    scale_2015cp = L_2015cp_cgs / model_2015cp['F275W'][0]

    return scale_2015cp


def bci_nan(detections, trials, conf=0.9, interval='jeffreys'):
    """Find binomial confidence interval for DataFrame with NaN values.
    Inputs:
        detections: DataFrame of detections
        trials: DataFrame of trials (incl. detections), same shape as detections
    Returns:
        bci_lower: DataFrame, same shape as trials, with lower BCI limits
        bci_upper: DataFrame, same shape as trials, with upper BCI limits
    """

    from astropy.stats import binom_conf_interval

    if detections.shape != trials.shape:
        raise ValueError('detections and trials must have the same shape.')

    bci_lower = pd.DataFrame([], index=trials.index)
    bci_upper = pd.DataFrame([], index=trials.index)

    # Calculate binomial confidence intervals
    for col in trials.columns:
        # separate bins with no trials
        pos_index = trials[trials[col] >= 1].index
        zero_index = trials[trials[col] < 1].index

        # calculate BCI for cells with positive trials
        bci = binom_conf_interval(detections.loc[pos_index, col], 
                trials.loc[pos_index, col], confidence_level=conf, 
                interval=interval)
        
        # add to dataframes
        bci_lower.loc[pos_index, col] = bci[0].T
        bci_upper.loc[pos_index, col] = bci[1].T

        # in cases with no trials, lower limit 0. and upper limit 1.
        bci_lower.loc[zero_index,col] = 0.
        bci_upper.loc[zero_index,col] = 1.

    return bci_lower, bci_upper