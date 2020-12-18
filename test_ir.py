from galex_injection_recovery import run_trials
from supernova import Supernova
from light_curve import LightCurve
import numpy as np

# Initialize SN object
sn = Supernova('SN2007on')
# Import light curves
lcs = []
for band in ['FUV', 'NUV']:
    try:
        lc = LightCurve(sn, band, data_dir='data')
    except:
        # No data for this channel
        continue

    # Skip if no data after minimum recovery time
    if np.max(lc.data['t_delta_rest']) < 50:
        continue

    lcs.append(lc)

run_trials(sn, lcs, 10000, [0, 2500], [1, 1000])