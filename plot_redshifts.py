import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

bin_width = 0.025

sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
z = sn_info['z']
z = z[pd.notna(z)].astype(float)
bins = int((max(z) - min(z)) / bin_width)

fig, ax = plt.subplots()
plt.hist(z, bins=bins, histtype='bar', color='#e67', rwidth=0.95)
fig.set_tight_layout(True)
plt.xlabel('Redshift')
plt.ylabel('Number of SNe Ia')
plt.xlim((0,0.5))

plt.savefig(Path('out/redshifts.pdf'), bbox_inches='tight', dpi=300)
plt.show()