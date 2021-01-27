import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Plot settings
x_min = 1300
x_max = 3300
y_min = 0
dx_major = 500 # tick spacing
dx_minor = 100
dy_major = 4
dy_minor = 1

fig, ax = plt.subplots()

# Import reference info
ref = pd.read_csv(Path('ref/filters.csv'), index_col='name')

y_max = 0

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
    text_y = {'above': array.max() + 0.7, 'below': -1, 'middle': array.max()/2}
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

    # Plot maxe
    y_max = max(y_max, np.max(array.resp))

x_ticks_major = np.arange(x_min, x_max + dx_major, dx_major)
x_ticks_minor = np.arange(x_min, x_max + dx_minor, dx_minor)
ax.xaxis.set_major_locator(ticker.MultipleLocator(dx_major))
ax.xaxis.set_minor_locator(ticker.FixedLocator(x_ticks_minor))
y_ticks_major = np.arange(y_min, y_max + dy_major, dy_major)
y_ticks_minor = np.arange(y_min, y_max + dy_minor, dy_minor)
ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks_major))
ax.yaxis.set_minor_locator(ticker.FixedLocator(y_ticks_minor))
ax.tick_params(axis='both', which='both', top=False, right=False)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_bounds(x_min, x_max)
ax.spines['left'].set_bounds(y_min, y_ticks_minor[-1])

# Adjust ticks and frame
ax.set_xlim((x_ticks_minor[0] - dx_minor, None))
ax.set_ylim((-2.5 * dy_minor, None))

ax.set_xlabel('Wavelength [Å]')
# ax.set_xlim((1200, 3300))
# ax.set_ylim((-1., None))
ax.set_ylabel('Effective Throughput [%]', rotation='horizontal', ha='left', y=1.02, labelpad=0)

plt.tight_layout(pad=0.2)
plt.subplots_adjust(top=0.9)

plt.savefig(Path('out/filters.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(Path('out/filters.png'), bbox_inches='tight', dpi=300)
plt.show()
