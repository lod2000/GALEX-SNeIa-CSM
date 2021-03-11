import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from light_curve import LightCurve

# sn_name = 'SNLS-04D3df'
# sn_name = 'SDSS-II SN 18647'
sn_name = 'PS1-10k'

lc = LightCurve.from_name(sn_name, 'NUV')

fig, axs = plt.subplots(2, 2, figsize=(8, 5))

fig.suptitle(sn_name)

axs[0,0].scatter(lc('detxs')/600, lc('detys')/600, s=10)
axs[0,0].set_xlabel('detxs [deg]')
axs[0,0].set_ylabel('detys [deg]')

axs[0,1].scatter(lc('detrad')/600, lc('detxs')/600, s=10)
axs[0,1].axvline(0.55, c='r')
axs[0,1].set_xlabel('detrad [deg]')
axs[0,1].set_ylabel('detxs [deg]')

axs[1,0].scatter(lc('detrad')/600, lc('exptime'), s=10)
axs[1,0].axvline(0.55, c='r')
axs[1,0].set_xlabel('detrad [deg]')
axs[1,0].set_ylabel('exptime')

axs[1,1].scatter(lc('detrad')/600, lc('responses'), s=10)
axs[1,1].axvline(0.55, c='r')
axs[1,1].set_xlabel('detrad [deg]')
axs[1,1].set_ylabel('responses')

plt.show()