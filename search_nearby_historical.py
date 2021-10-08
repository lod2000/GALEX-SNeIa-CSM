import numpy as np
import pandas as pd
from pathlib import Path

OSC_FILE = Path('ref/osc.csv')
OBS_FILE = Path('out/observations.csv')
# LC_DIR = Path('C:\\Users\\dubay.11\\Documents\\GALEXdata_v10\\LCs\\')

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
obs_df = post_disc[post_disc['sn_name'].isin(nearby.index)]
# obs_df.to_csv('out/nearby_historical_obs.csv')
sne = obs_df['sn_name'].drop_duplicates()
cols = ['disc_date', 'epochs_fuv', 'epochs_nuv', 't_delta_first',
        't_delta_last']
nearby_historical = pd.DataFrame([], index=sne, columns=cols)
# Consolidate NUV, FUV info
for sn in sne:
    obs = obs_df[obs_df['sn_name'] == sn]
    nearby_historical.loc[sn, 't_delta_first'] = obs['t_delta_first'].min()
    nearby_historical.loc[sn, 't_delta_last'] = obs['t_delta_last'].max()
    nearby_historical.loc[sn, 'disc_date'] = obs.iloc[0]['disc_date']
    for band in ['FUV', 'NUV']:
        snid = sn + '-' + band
        try:
            nearby_historical.loc[sn, 'epochs_'+band.lower()] = obs.loc[snid, 'epochs_post_disc']
        except KeyError:
            nearby_historical.loc[sn, 'epochs_'+band.lower()] = 0
# Count unique SNe
# unique = nearby_historical.drop_duplicates('sn_name')
print('There are %s nearby historical SNe.' % nearby_historical.shape[0])
# Limit to SNe with >=10 epochs
# nearby_historical = nearby_historical[nearby_historical['epochs_post_disc'] >= 10]
# nearby_historical = unique[cols].reset_index(drop=True).set_index('sn_name')

# Limit to SNe with light curves available
# lc_files = [f.stem for f in LC_DIR.glob('*.csv')]
# nearby_historical = nearby_historical[nearby_historical.index.isin(lc_files)]
nearby_historical.to_csv(Path('out/nearby_historical_all.csv'))
