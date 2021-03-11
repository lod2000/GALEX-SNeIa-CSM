import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from light_curve import LightCurve

sn_name = 'SDSS-II SN 18647'

lc = LightCurve.from_name(sn_name, 'NUV')

fig, axs = plt.subplots(2, 2, figsize=(8, 5))

fig.suptitle(sn_name)

axs[0,0].scatter(lc('detxs'), lc('detys'), s=10)
axs[0,0].set_xlabel('detxs')
axs[0,0].set_ylabel('detys')

axs[0,1].scatter(lc('detrad')/600, lc('detxs'), s=10)
axs[0,1].axvline(0.55, c='r')
axs[0,1].set_xlabel('detrad [deg]')
axs[0,1].set_ylabel('detxs')

axs[1,0].scatter(lc('detrad')/600, lc('exptime'), s=10)
axs[1,0].axvline(0.55, c='r')
axs[1,0].set_xlabel('detrad [deg]')
axs[1,0].set_ylabel('exptime')

axs[1,1].scatter(lc('detrad')/600, lc('responses'), s=10)
axs[1,1].axvline(0.55, c='r')
axs[1,1].set_xlabel('detrad [deg]')
axs[1,1].set_ylabel('responses')

plt.show()