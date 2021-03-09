import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from supernova import Supernova
from light_curve import import_swift, import_cfa, plot_lc, plot_external, flux2luminosity, LightCurve

# Plot color palette
COLORS = {'FUV' : '#a37', 'NUV' : '#47a', # GALEX
          'UVW1': '#cb4', 'UVM2': '#283', 'UVW2': '#6ce', # Swift
          'F275W': '#e67', # Hubble
          'B': '#6ce', 'V': '#283', "r'": '#e67', "i'": '#888' # CfA
          }

# Other plot settings
BG_SPAN_ALPHA = 0.2 # background span transparency
BG_LINE_ALPHA = 0.7 # background line transparency
BG_SIGMA = 1 # background uncertainty
DT_MIN = -30 # Separation between background and SN data (days)
DET_SIGMA = 3 # detection threshold & nondetection upper limit
TMAX = 100 # maximum time to plot
X_MAJOR_TICK = 20 # spacing of x-axis major ticks

sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
supernovae = ['SN2007on', 'SN2008hv', 'SN2009gf', 'SN2010ai']

fig, axs = plt.subplots(2, 2, figsize=(6.5, 5))

all_handles = []
all_labels = []

for i, (sn_name, ax) in enumerate(zip(supernovae, axs.flat)):
    sn = Supernova(sn_name, sn_info=sn_info)

    # Plot Swift data data
    if sn_name in ['SN2007on', 'SN2008hv', 'SN2009gf']:
        lc = import_swift(sn.name, sn.disc_date.mjd, sn.z)
        plot_external(ax, sn, lc, ['UVW1', 'UVM2', 'UVW2'])

    # Plot CfA data
    if sn_name == 'SN2010ai':
        lc = import_cfa(sn)
        plot_external(ax, sn, lc, ['B', 'V', "r'", "i'"], marker='s')

    # Import and plot GALEX data
    ymin = []
    bg_max = 0
    pre_obs = 0
    for band in ['FUV', 'NUV']:
        try:
            lc = LightCurve(sn, band)
            # Pre-SN obs.
            before = lc.data[lc('t_delta_rest') <= DT_MIN]
            pre_obs += len(before.index)
            ymin.append(lc.bg / 1.5)
            bg_max = max(bg_max, lc.bg)

            plot_lc(ax, lc, TMAX)
        except FileNotFoundError:
            # No data for this channel
            continue

    # In-plot labels
    ax.set_title(sn_name.replace('SN', 'SN '), ha='right', va='top', x=0.9, y=0.85, size=14)
    xlim = np.array(ax.get_xlim())
    # ax.text(xlim[0]+3, bg_max * 1.1,'host %sσ (%s obs)' % (BG_SIGMA, pre_obs))

    # Adjust and label axes and ticks
    ax.set_yscale('log')
    ax.tick_params(axis='y', which='both', right=False)
    ax.set_ylim((np.min(ymin), None))
    ax.xaxis.set_major_locator(plt.MultipleLocator(X_MAJOR_TICK))

    # Twin axis with absolute luminosity
    luminosity_ax = ax.twinx()
    ylim_flux = np.array(ax.get_ylim())
    # Assume FUV for extinction; not that big a difference between the two
    ylim_luminosity = flux2luminosity(ylim_flux, 0, sn.dist, sn.dist_err,
            sn.z, sn.z_err, sn.a_v, 'FUV')[0].value
    luminosity_ax.set_yscale('log')
    luminosity_ax.set_ylim(ylim_luminosity)

    if i % 2 == 0:
        ax.set_ylabel('$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')
    else:
        luminosity_ax.set_ylabel('$L_\mathrm{UV}$ [erg s$^{-1}$ Å$^{-1}$]', rotation=270,
                labelpad=18)

    if i in range(2, 4):
        ax.set_xlabel('Time since discovery [days]')

    handles, labels = ax.get_legend_handles_labels()
    all_handles += handles
    all_labels += labels

plt.tight_layout(pad=0.3)
plt.subplots_adjust(top=0.87)

# Remove duplicates
unique = [(h, l) for i, (h, l) in enumerate(zip(all_handles, all_labels)) if l not in all_labels[:i]]
# Re-order with NUV first
handles, labels = zip(*unique)
order = [4, 5, 0, 1, 2, 3, 6, 7, 8, 9]
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
# Add legend
fig.legend(handles, labels, loc='upper right', ncol=5, handletextpad=0.5, 
        handlelength=1., borderaxespad=0.5, borderpad=0.5, columnspacing=1.,
        bbox_to_anchor=(0.915, 1.))

plt.savefig(Path('out/normal_detections.pdf'), dpi=300)

plt.show()