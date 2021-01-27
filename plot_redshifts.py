import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Settings
bin_width = 0.02
z_min = 0.
z_max = 0.5
color = 'gray'
major_tick_space = 20
minor_tick_space = 10

# Import data
sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
z = sn_info['z']
z = z[pd.notna(z)].astype(float)
bins = int((z_max - z_min) / bin_width)

fig, ax = plt.subplots()

hist = ax.hist(z, bins=bins, range=(z_min, z_max), histtype='bar', color=color)

ax.set_xlabel('Redshift')
ax.set_ylabel('Number of SNe Ia')

plt.grid(b=True, which='major', axis='x', color='w', lw=2)
plt.grid(b=True, which='minor', axis='x', color='w', lw=1)
plt.grid(b=True, which='major', axis='y', color='w', lw=1)

# Adjust ticks and frame
ax.set_xlim((z_min - bin_width, z_max))
ax.set_ylim((-minor_tick_space, None))

hist_arr = np.histogram(z, bins=bins, range=(z_min, z_max))
yticks_major = np.arange(0, np.max(hist_arr[0]) + major_tick_space, major_tick_space)
ax.yaxis.set_major_locator(ticker.FixedLocator(yticks_major))
yticks_minor = np.arange(0, np.max(hist_arr[0]) + minor_tick_space, minor_tick_space)
ax.yaxis.set_minor_locator(ticker.FixedLocator(yticks_minor))
ax.tick_params(axis='both', which='both', top=False, right=False)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_bounds(z_min, z_max)
ax.spines['left'].set_bounds(0, yticks_minor[-1])

ax.set_ylabel('Number of SNe Ia', rotation='horizontal', ha='left', y=1.05, labelpad=-1)

plt.tight_layout(pad=0.3)
plt.subplots_adjust(top=0.88)

plt.savefig(Path('out/redshifts.pdf'), bbox_inches='tight', dpi=300)

plt.show()
