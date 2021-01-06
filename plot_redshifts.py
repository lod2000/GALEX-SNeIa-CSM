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

fig, ax = plt.subplots()
plt.hist(z, bins=bins, range=(z_min, z_max), histtype='bar', color='#e67', rwidth=1.)
fig.set_tight_layout(True)
plt.xlabel('Redshift')
plt.ylabel('Number of SNe Ia')

plt.grid(b=True, which='major', axis='x', color='w', lw=2)
plt.grid(b=True, which='minor', axis='x', color='w', lw=1)
plt.grid(b=True, which='major', axis='y', color='w', lw=1)

plt.savefig(Path('out/redshifts.pdf'), bbox_inches='tight', dpi=300)
plt.show()