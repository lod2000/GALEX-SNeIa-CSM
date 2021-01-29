import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

fig, ax = plt.subplots()

# Import reference info
ref = pd.read_csv(Path('ref/filters.csv'), index_col='name')

for name, f in ref.iterrows():
    label = ' '.join([f.display_name, name])
    # Import response file
    array = pd.read_csv(Path('ref/%s.resp' % name), names=['freq', 'resp'],
            sep=' ', index_col=0)

    # convert effective area to throughput
    if f.type == 'effective area':
        array.resp = array.resp/(np.pi * f.aperture_radius**2)

    # Limit tail end of distribution
    ymin = 0.002
    array = array[array.resp > ymin]

    array.resp *= 100

    # Add in-plot labeling
    text_x = {'mean': np.mean(array.index), 
              'max': array.idxmax(), 
              'first': array.index[0] + 150}
    text_y = {'above': array.max() + 0.8, 'below': -1.2, 'middle': array.max()/2}
    if f.instrument == 'GALEX':
        text_size = 10
        weight = 'bold'
    else:
        text_size = 10
        weight = 'normal'
    ax.text(text_x[f.label_x], text_y[f.label_y], name, ha='center', va='center', 
            size=text_size, c=f.color, weight=weight)

    # Plot filter response curve
    ax.plot(array.index, array.resp, label=label, ls=f.style, alpha=f.alpha, 
            c=f.color)

ax.set_ylim((-3, None))
ax.set_xlabel('Wavelength [Å]')
ax.set_ylabel('Effective Throughput [%]')

plt.tight_layout(pad=0.3)

plt.savefig(Path('out/filters.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(Path('out/filters.png'), bbox_inches='tight', dpi=300)
plt.show()
