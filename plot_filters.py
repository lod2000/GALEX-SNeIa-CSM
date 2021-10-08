import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

fig, axs = plt.subplots(2, figsize=(3.25, 3.25), sharex=True, sharey=True)

# Import reference info
ref = pd.read_csv(Path('ref/filters.csv'), index_col='name')

for name, f in ref.iterrows():
    # Two panels: GALEX/HST and comparison
    if name in ['FUV', 'NUV', 'F275W']:
        ax = axs[0]
    else:
        ax = axs[1]
    
    label = ' '.join([f.display_name, name])
    # Import response file
    array = pd.read_csv(Path('ref/%s.resp' % name), names=['freq', 'resp'],
            sep=' ', index_col=0)

    # Normalize response curve
    norm = array / array.resp.max()

    # Limit tail end of distribution
    ymin = 0.05
    norm = norm[norm.resp > ymin]

    # Add in-plot labeling
    pad_y = 0.05
    text_x = {'mean': np.mean(norm.index), 
              'max': norm.idxmax(), 
              'first': norm.index[0] + 150}
    text_y = {'above': norm.max(), 
              'below': -pad_y, 
              'middle': norm.max()/2,
              'bottom': pad_y}
    # if f.instrument == 'GALEX':
    #     text_size = 10
    #     weight = 'bold'
    # else:
    text_size = 10
    weight = 'normal'
    ax.text(text_x[f.label_x] + f.pad_x, text_y[f.label_y], name, ha='center',
            va='bottom', 
            size=text_size, c=f.color, weight=weight)

    # Plot filter response curve
    ax.plot(norm.index, norm.resp, label=label, ls=f.style, alpha=1, 
            c=f.color)

for ax in axs:
    ax.set_ylim((-pad_y, 1.2))
axs[1].set_xlabel('Wavelength [Å]')

# Add big axis label
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, 
                left=False, right=False)
plt.ylabel('Normalized Filter Response')

plt.tight_layout(pad=0.3)

# plt.savefig(Path('out/filters.pdf'), bbox_inches='tight', dpi=600)
# plt.savefig(Path('out/filters.png'), bbox_inches='tight', dpi=600)
plt.show()
