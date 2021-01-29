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

ax.grid(b=True, which='major', axis='x', color='w', lw=1)
ax.grid(b=True, which='minor', axis='x', color='w', lw=1)
ax.grid(b=True, which='major', axis='y', color='w', lw=1)

ax.set_xlabel('Redshift')
ax.set_ylabel('Number of SNe Ia')
ax.set_ylim((None, 110))

plt.tight_layout(pad=0.3)

plt.savefig(Path('out/redshifts.pdf'), bbox_inches='tight', dpi=300)

plt.show()
