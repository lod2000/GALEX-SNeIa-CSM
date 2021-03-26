import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from astropy.time import Time
from astropy import units as u
from astropy import constants as const

from light_curve import LightCurve
from supernova import Supernova
from graham_observations import GrahamObservation
from light_curve import import_swift, flux2luminosity, wavelength2freq, effective_wavelength
from utils import *

# Settings
DET_SIGMA = [5, 3] # detection significance threshold
DET_COUNT = [1, 3] # number of points above DET_SIGMA to count as detection
PLOT_SIGMA = 1 # multiple of uncertainty to plot as luminosity limit
YLIM = (5e38, 6e41) # erg/s
XLIM = (-50, 2250) # days

# Plot design
COLORS = {  'SN2007on': 'cyan',
            'SN2008hv': 'orange',
            'SN2009gf': 'green',
            'SN2010ai': 'magenta',
            'FUV': '#a37', 
            'NUV' : '#47a',
            'F275W': 'k',
}
MARKERS = { 'SN2007on': 'o',
            'SN2008hv': 's',
            'SN2009gf': 'p',
            'SN2010ai': 'd'
}
MS_DET = 6 # detection marker size
MS_GALEX = 6 # marker size for GALEX nondetections
MS_HST = 5 # marker size for HST nondetections
ALPHA_DET = 0.6
ALPHA_NON = 0.3 # alpha of nondetection limits below cutoff

def main():

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
    det_sne = ['SN2007on', 'SN2008hv', 'SN2009gf', 'SN2010ai']

    x_col = 't_delta_rest'
    y_col = 'luminosity_hostsub'
    yerr_col = 'luminosity_hostsub_err'
    
    fig, ax = plt.subplots(figsize=(6.5, 5), tight_layout=True)

    # Plot Swift SN2011fe from Brown+ 2012
    band = 'UVM2'
    disc_date = Time('2011-08-24', format='iso').mjd
    dist = 6.4 * u.Mpc #from Shappee & Stanek 2011
    dist_err = 0. * u.Mpc
    z = 0 # too close to need correction
    z_err = 0
    a_v = 0 # won't worry about it right now
    a_band = 'NUV' # close enough
    wl_eff = effective_wavelength(band).value # UVM2 effective wavelength
    lc = import_swift('SN2011fe', disc_date, z)
    lc = lc[lc['Filter'] == band]
    lc['luminosity'], lc['luminosity_err'] = flux2luminosity(lc['flux'], 
            lc['flux_err'], dist, dist_err, z, z_err, a_v, a_band)
    ax.plot(lc['t_delta_rest'], wl_eff * lc['luminosity'], color='brown', 
            label='SN2011fe (%s)' % band, zorder=1, lw=2, rasterized=True)

    # Import and plot HST non-detections
    print('Importing HST non-detections...')
    hst_data = pd.read_csv(Path('ref/Graham_observations.csv'), index_col=0)
    wl_eff = effective_wavelength('F275W').value
    nondetections = hst_data[hst_data['Detection'] == False]
    for i, sn_name in enumerate(nondetections.index):
        obs = GrahamObservation(sn_name, hst_data)
        ax.scatter(obs.rest_phase, wl_eff * obs.luminosity_limit, 
                marker='v', s=MS_HST**2, color='w', edgecolors=COLORS['F275W'], 
                alpha=ALPHA_NON, zorder=3, rasterized=True)

    # Plot near-peak SNe Ia
    for sn_name in det_sne:
        sn = Supernova(sn_name, sn_info)
        band = 'NUV'
        lc = LightCurve(sn, band)
        lc.to_hz() # Convert to erg/s/Hz units
        # Effective filter frequency
        # nu_eff = (const.c / effective_wavelength(band)).to('Hz').value
        wl_eff = effective_wavelength(band).value
        # Plot near-peak detections
        detections = lc.detect_csm(DET_SIGMA, count=DET_COUNT)
        # ax.errorbar(detections[x_col], nu_eff * detections[y_col], 
        #         yerr=nu_eff * detections[yerr_col], label='%s (%s)' % (sn.name, band), 
        #         linestyle='none', ms=MS_DET, marker=MARKERS[sn.name], 
        #         color=COLORS[sn.name], mec='k', ecolor='k', elinewidth=1, 
        #         zorder=9, rasterized=True)
        ax.errorbar(detections[x_col], wl_eff * detections[y_col], 
                yerr=wl_eff * detections[yerr_col], label='%s (%s)' % (sn.name, band), 
                linestyle='none', ms=MS_DET, marker=MARKERS[sn.name], 
                color=COLORS[sn.name], mec='k', ecolor='k', elinewidth=1, 
                zorder=9, rasterized=True)

    # Plot non-detection limits
    print('Importing GALEX detections and limits...')
    for sn_name in tqdm(sn_info.index):
        sn = Supernova(sn_name, sn_info)

        for band in ['FUV', 'NUV']:
            try:
                plot_nondetection_limits(ax, sn, band, x_col, yerr_col, ymax=YLIM[1])
            except (FileNotFoundError, pd.errors.EmptyDataError):
                continue

    # Import and plot HST detections
    print('Importing HST detections...')
    wl_eff = effective_wavelength('F275W').value
    detections = hst_data[hst_data['Detection']]
    markers = ['X', '*']
    colors = ['y', 'r']
    sizes = [64, 81]
    for i, sn_name in enumerate(detections.index):
        obs = GrahamObservation(sn_name, hst_data)
        ax.scatter(obs.rest_phase, wl_eff * obs.luminosity, marker=markers[i], 
                color=colors[i], edgecolors='k', label='%s (F275W)' % sn_name, 
                zorder=10,  s=sizes[i])

    print('Plotting...')

    # Format axes
    ax.set_xlabel('Time since discovery [rest-frame days]', size=12)
    ax.set_ylabel('$\\lambda L_\\lambda$ [erg s$^{-1}$]', size=12)
    ax.set_xlim(XLIM)
    ax.set_yscale('log')
    ax.set_ylim(YLIM)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = [
            Line2D([0], [0], marker='v', markerfacecolor='w', 
                    markeredgecolor=COLORS['F275W'], markersize=MS_HST,
                    alpha=ALPHA_NON, label='detection limit (F275W)', lw=0),
            Line2D([0], [0], marker='v', markerfacecolor=COLORS['FUV'], 
                    markeredgecolor='none', markersize=MS_GALEX, alpha=ALPHA_NON,
                    label='detection limit (FUV)', lw=0),
            Line2D([0], [0], marker='v', markerfacecolor=COLORS['NUV'], 
                    markeredgecolor='none', markersize=MS_GALEX, alpha=ALPHA_NON,
                    label='detection limit (NUV)', lw=0)
    ]
    plt.legend(handles=handles + legend_elements, loc='lower center', ncol=3,
            handletextpad=0.5, handlelength=1.0, bbox_to_anchor=(0.5, 1.01))

    plt.savefig(Path('out/limits.pdf'), dpi=300)

    plt.show()


def plot_nondetection_limits(ax, sn, band, x_col, yerr_col, ymax=1e42):
    """Plot UV luminosity limits for GALEX non-detections."""

    lc = LightCurve(sn, band)

    # Effective filter wavelength
    wl_eff = effective_wavelength(band).value

    # Remove detections, inc. spurious
    detections = lc.detect_csm(DET_SIGMA, count=DET_COUNT)
    nondetections = lc(tmin=XLIM[0], tmax=XLIM[1]).drop(detections.index)

    # Plot nondetections below SN 2015cp luminosity
    below_max = nondetections[PLOT_SIGMA * nondetections[yerr_col] <= ymax]
    ax.scatter(below_max[x_col], wl_eff * PLOT_SIGMA * below_max[yerr_col], 
            marker='v', s=MS_GALEX**2, color=COLORS[band], edgecolors='none', 
            alpha=ALPHA_NON, zorder=3, rasterized=True)

    return ax


if __name__ == '__main__':

    main()
