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
    # array = np.loadtxt(Path('ref/%s.resp' % name), delimiter=' ')
    array = pd.read_csv(Path('ref/%s.resp' % name), names=['freq', 'resp'],
            sep=' ', index_col=0)
    # freq = array[:,0]
    # y = array[:,1]

    # convert effective area to throughput
    if f.type == 'effective area':
        array.resp = array.resp/(np.pi * f.aperture_radius**2)

    # Limit tail end of distribution
    ymin = 0.002
    array = array[array.resp > ymin]

    array.resp *= 100

    # Add in-plot labeling
    text_x = {'mean': np.mean(array.index), 'max': array.idxmax()}
    text_y = {'above': array.max() + 0.7, 'below': -0.5, 'middle': array.max()/2}
    if f.instrument == 'GALEX':
        text_size = 20
    else:
        text_size = 16
    ax.text(text_x[f.label_x], text_y[f.label_y], name, ha='center', va='center', 
            size=text_size, c=f.color)

    ax.plot(array.index, array.resp, label=label, ls=f.style, alpha=f.alpha, 
            c=f.color)

ax.set_xlabel('Wavelength [Å]')
ax.set_xlim((1200, 3300))
ax.set_ylim((-1., None))
ax.set_ylabel('Effective Throughput [%]')
# ax.legend()

plt.savefig(Path('out/filters.pdf'), bbox_inches='tight', dpi=300)
plt.show()
