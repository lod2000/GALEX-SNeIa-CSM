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

    # Normalize response curve
    # d_lambda = (array.index[-1] - array.index[0]) / len(array.index)
    # array['resp_norm'] = array.resp / (array.resp.sum() * d_lambda)
    # norm = array / (array.sum() * d_lambda)
    norm = array / array.resp.max()

    # Limit tail end of distribution
    ymin = 0.05
    # if f.instrument == 'Swift':
    # ymax = norm.resp.max()
    # ymin = 0.075 * ymax
    norm = norm[norm.resp > ymin]

    # convert effective area to throughput
    # if f.type == 'effective area':
    #     array.resp = array.resp/(np.pi * f.aperture_radius**2)

    # array.resp *= 100

    # Add in-plot labeling
    pad_y = 0.05
    text_x = {'mean': np.mean(norm.index), 
              'max': norm.idxmax(), 
              'first': norm.index[0] + 150}
    text_y = {'above': norm.max(), 
              'below': -pad_y, 
              'middle': norm.max()/2,
              'bottom': pad_y}
    if f.instrument == 'GALEX':
        text_size = 10
        weight = 'bold'
    else:
        text_size = 10
        weight = 'normal'
    ax.text(text_x[f.label_x] + f.pad_x, text_y[f.label_y], name, ha='center',
            va='bottom', 
            size=text_size, c=f.color, weight=weight)

    # Plot filter response curve
    ax.plot(norm.index, norm.resp, label=label, ls=f.style, alpha=f.alpha, 
            c=f.color)

ax.set_ylim((-pad_y, 1.2))
ax.set_xlabel('Wavelength [Å]')
ax.set_ylabel('Normalized Filter Response')
ax.yaxis.set_ticklabels([])

plt.tight_layout(pad=0.3)

plt.savefig(Path('out/filters.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(Path('out/filters.png'), bbox_inches='tight', dpi=300)
# plt.show()
