from pathlib import Path
import platform
import pandas as pd
import numpy as np
from numpy.random import default_rng

# Default values
RECOV_MIN = 50 # minimum number of days after discovery to count as recovery
TSTART_MIN = 0
TSTART_MAX = 1000
SCALE_MIN = 0.1
SCALE_MAX = 10.
DECAY_RATE = 0.3 # CSM curve decay factor
WIDTH = 250 # days, from PTF11kx

# Default directories
SAVE_DIR = Path('save')
DATA_DIR = Path('data')
OUTPUT_DIR = Path('out')

# Plot color palette
COLORS = {'FUV' : '#a37', 'NUV' : '#47a', # GALEX
          'UVW1': '#cb4', 'UVM2': '#283', 'UVW2': '#6ce', # Swift
          'F275W': '#e67' # Hubble
          }


def fname2sn(fname):
    """Extract SN name and band from a file name."""

    fname = Path(fname)
    split = fname.stem.split('-')
    sn_name = '-'.join(split[:-1])
    band = split[-1]
    # Windows replaces : with _ in some file names
    if 'CSS' in sn_name or 'MLS' in sn_name:
        sn_name.replace('_', ':', 1)
    sn_name.replace('_', ' ')
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


def check_save(sn_name, iterations, model, save_dir=SAVE_DIR):
    """Checks if save file exists for given SN and iterations."""

    save_file = sn2fname(sn_name, str(iterations), parent=save_dir / Path(model))
    return save_file.is_file()


def gen_params(iterations, tstart_min, tstart_max, scale_min, scale_max):
    """Generate random injection-recovery parameters."""

    rng = default_rng()
    tstart = rng.integers(tstart_min, tstart_max, iterations, endpoint=True)
    scale = rng.uniform(scale_min, scale_max, iterations)
    params = np.column_stack((tstart, scale))

    return params


def detect_csm(time, data, err, sigma, count=[1], dt_min=-30):
    """Detect CSM. For multiple confidence tiers, len(sigma) = len(count).
    Inputs:
        time, data, err: pd.Series
        sigma: detection confidence level, int or list
        count: number of data points above sigma to count as detection, list
        dt_min: minimum number of days post-discovery
    Output:
        detections: list of indices of detected observations
    """

    # separate SN data from host background data
    time = time[time > dt_min]
    data = data[time.index]
    err = err[time.index]
    # confidence level of data being away from 0
    conf = data / err

    if type(sigma) == int:
        sigma = [sigma]

    # Tiered detection: requires N points above X sigma or M points above Y sigma
    detections = []
    for s, c in zip(sigma, count):
        detected = time[conf >= s].index.to_list()
        if len(detected) >= c:
            detections += detected
    
    # Remove duplicates, sort, and return indices of detections
    detections = sorted(list(dict.fromkeys(detections)))
    return detections