from pathlib import Path
import platform
import pandas as pd

# Default values
RECOV_MIN = 50 # minimum number of days after discovery to count as recovery
TSTART_MIN = 0
TSTART_MAX = 1000
SCALE_MIN = 0.1
SCALE_MAX = 10.

# Default directories
SAVE_DIR = Path('save')
DATA_DIR = Path('data')
OUTPUT_DIR = Path('out')


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


def sn2fname(sn_name, band, suffix='.csv'):
    """Convert SN name and GALEX band to a file name, e.g. for a light curve CSV."""

    fname = '-'.join((sn_name, band)) + suffix
    fname = fname.replace(' ', '_')
    # Make Windows-friendly
    if (platform.system() == 'Windows') or ('Microsoft' in platform.release()):
        fname = fname.replace(':', '_')
    return Path(fname)


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