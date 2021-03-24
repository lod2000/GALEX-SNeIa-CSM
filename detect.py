import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from supernova import Supernova
from light_curve import LightCurve
from light_curve import effective_wavelength

SIGMA = [5, 3]
SIGCOUNT = [1, 3]

L_15cp = 7.6e25 # erg/s/Hz
L_nu = L_15cp * 3e18 / effective_wavelength('F275W').value # erg/s

sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')

detections = []
total_epochs = 0
bg_epochs = 0
post_epochs = 0
below_15cp = 0

for sn_name in tqdm(sn_info.index):
    sn = Supernova(sn_name, sn_info)
    below_15cp_val = 0

    for band in ['FUV', 'NUV']:
        try:
            lc = LightCurve(sn, band)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue

        # Count observation epochs
        total_epochs += lc.data.shape[0]
        bg_epochs += lc.bg_data.shape[0]
        post_epochs += lc.data[lc.data['t_delta_rest'] > 0].shape[0]

        det = lc.detect_csm(SIGMA, SIGCOUNT)

        if len(det.index) > 0:
            det_info = {
                'name': sn_name, 
                'band': band, 
                'ndet': len(det.index),
                'i0': det.index[0], 
                'i1': det.index[-1], 
                't0': det.loc[det.index[0],'t_delta_rest'],
                't1': det.loc[det.index[-1], 't_delta_rest'],
                'lmax': det['luminosity_hostsub'].max(),
                'lmaxerr': det.loc[det['luminosity_hostsub'].idxmax(), 'luminosity_hostsub_err'],
                'tmax': det.loc[det['luminosity_hostsub'].idxmax(), 't_delta_rest'],
                'bg': lc.bg,
                'bgerr': lc.bg_err,
                'bgobs': lc.bg_data.shape[0]
            }
            detections.append(det_info)

        # Count limits below 15cp luminosity
        lc.to_hz()
        nu_eff = 3e18 / effective_wavelength(band).value # Hz
        low_limit = lc.data[nu_eff*lc.data['luminosity_hostsub_err_hz'] < L_nu]
        if len(low_limit.index) > 0:
            below_15cp_val = 1

    below_15cp += below_15cp_val

detections = pd.DataFrame(detections)
print('\nDetections:')
print(detections)
detections.to_csv('out/detections.csv')

print('\nTotal number of epochs: %s' % total_epochs)
print('Total number of post-SN epochs: %s' % post_epochs)
print('Avg pre-SN obs: %s' % (bg_epochs / len(sn_info.index)))

print('\nLuminosity of SN 2015cp: %s erg/s' % L_nu)
print('Number of SNe Ia with limits below 15cp: %s' % below_15cp)
