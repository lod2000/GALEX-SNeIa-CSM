import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

fig, ax = plt.subplots(tight_layout=True)

# Import reference info
ref = pd.read_csv(Path('ref/filters.csv'), index_col='name')

for name, f in ref.iterrows():
    label = ' '.join([f.display_name, name])
    # Import response file
    array = np.loadtxt(Path('ref/%s.resp' % name), delimiter=' ')
    freq = array[:,0]
    y = array[:,1]

    # convert effective area to throughput
    if f.type == 'effective area':
        y = y/(np.pi * f.aperture_radius**2)

    ax.plot(freq, y*100, label=label, ls=f.style, alpha=f.alpha, c=f.color)

ax.set_xlabel('Wavelength [Å]')
ax.set_xlim((1200, 3300))
ax.set_ylim((-0.5, None))
ax.set_ylabel('Effective Throughput [%]')
ax.legend()

plt.savefig(Path('out/filters.pdf'), bbox_inches='tight', dpi=300)
plt.show()
