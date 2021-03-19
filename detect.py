import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from supernova import Supernova
from light_curve import LightCurve

SIGMA = [5, 3]
SIGCOUNT = [1, 3]

sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')

detections = []
total_epochs = 0
bg_epochs = 0
post_epochs = 0

for sn_name in tqdm(sn_info.index):
    sn = Supernova(sn_name, sn_info)

    for band in ['FUV', 'NUV']:
        try:
            lc = LightCurve(sn, band)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue

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

detections = pd.DataFrame(detections)
print('\nDetections:')
print(detections)
detections.to_csv('out/detections.csv')

print('\nTotal number of epochs: %s' % total_epochs)
print('Total number of post-SN epochs: %s' % post_epochs)
print('Avg pre-SN obs: %s' % (bg_epochs / len(sn_info.index)))
