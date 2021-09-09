import numpy as np
import pandas as pd
import os
from pathlib import Path

from observations import GalexObservation

OSC_FILE = Path('ref/osc.csv')
OBS_FILE = Path('out/observations.csv')
LC_DIR = Path('C:\\Users\\dubay.11\\Documents\\GALEXdata_v10\\LCs\\')

# Import list of SNe from Open Supernova Catalog
osc = pd.read_csv(OSC_FILE, index_col='Name')
# Limit to nearby, z<0.01
nearby = osc[osc['z'] < 0.01]
# Import list of all GALEX SN observations
observations = pd.read_csv(OBS_FILE)
id_col = observations['file'].str.split('.fits.gz', expand=True)[0]
observations.index = pd.Series(id_col, name='id')
# Limit to post-discovery obs only
post_disc = observations[observations['epochs_pre_disc'] == 0]
# Combine above selections
nearby_historical = post_disc[post_disc['sn_name'].isin(nearby.index)]
# Limit to SNe with >=10 epochs
nearby_historical = nearby_historical[nearby_historical['epochs_post_disc'] >= 10]

# Limit to SNe with light curves available
# lc_files = [f.stem for f in LC_DIR.glob('*.csv')]
# nearby_historical = nearby_historical[nearby_historical.index.isin(lc_files)]
nearby_historical.to_csv(Path('out/nearby_historical.csv'))
