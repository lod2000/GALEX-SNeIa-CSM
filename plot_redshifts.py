import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Settings
bin_width = 0.02
z_min = 0.
z_max = 0.5

sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
z = sn_info['z']
z = z[pd.notna(z)].astype(float)
bins = int((z_max - z_min) / bin_width)

fig, ax = plt.subplots(tight_layout=True)

ax.hist(z, bins=bins, range=(z_min, z_max), histtype='bar', color='#e67')

ax.set_xlabel('Redshift')
ax.set_ylabel('Number of SNe Ia')

plt.grid(b=True, which='major', axis='x', color='w', lw=3)
plt.grid(b=True, which='minor', axis='x', color='w', lw=2)
plt.grid(b=True, which='major', axis='y', color='w', lw=1)

plt.savefig(Path('out/redshifts.pdf'), bbox_inches='tight', dpi=300)

# Alternate: no ticks or frame, tick labels every 10
plt.box(False)
ax.set_xlim((z_min, z_max))
hist = np.histogram(z, bins=bins, range=(z_min, z_max))
yticks = np.arange(0, np.max(hist[0]), 20)
ax.set_yticks(yticks)
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, 
        right=False)
ax.tick_params(axis='x', pad=8)
ax.set_ylabel('Number of SNe Ia', rotation='horizontal', ha='left', va='top', y=1.)

plt.savefig(Path('out/redshifts_alt.pdf'), bbox_inches='tight', dpi=300)

plt.show()