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
YLIM = (None, 6e41) # erg/s
XLIM = (-50, 2250) # days
LIMIT_CUTOFF = 6e41 #10**25.88 # Graham 2015cp detection, erg/s/Hz

# Plot design
COLORS = {  'SN2007on': 'cyan',
            'SN2008hv': 'orange',
            'SN2009gf': 'green',
            'SN2010ai': 'magenta',
            'FUV': '#a37', 
            'NUV' : '#47a'
}
MARKERS = { 'SN2007on': 'o',
            'SN2008hv': 's',
            'SN2009gf': 'p',
            'SN2010ai': 'd'
}
MS_DET = 6 # detection marker size
MS_LOW = 6 # nondetection marker size below cutoff
MS_HI = 4 # nondetection marker size above cutoff
ALPHA_DET = 0.6
ALPHA_LOW = 0.3 # alpha of nondetection limits below cutoff
ALPHA_HI = 0.05 # alpha of nondetection limits above cutoff

def main():

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
    det_sne = ['SN2007on', 'SN2008hv', 'SN2009gf', 'SN2010ai']

    x_col = 't_delta_rest'
    y_col = 'luminosity_hostsub_hz'
    yerr_col = 'luminosity_hostsub_err_hz'
    
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
    nu_m2 = (const.c / effective_wavelength(band)).to('Hz').value # effective freq
    lc = import_swift('SN2011fe', disc_date, z)
    lc = lc[lc['Filter'] == band]
    lc['luminosity'], lc['luminosity_hz'] = flux2luminosity(lc['flux'], 
            lc['flux_err'], dist, dist_err, z, z_err, a_v, a_band)
    lc['luminosity_hz'] = wavelength2freq(lc['luminosity'], 2245.8)
    ax.plot(lc['t_delta_rest'], nu_m2 * lc['luminosity_hz'], color='brown', 
            label='SN2011fe (%s)' % band, zorder=1, lw=2, rasterized=True)

    # Plot near-peak SNe Ia
    for sn_name in det_sne:
        sn = Supernova(sn_name, sn_info)
        band = 'NUV'
        lc = LightCurve(sn, band)
        lc.to_hz() # Convert to erg/s/Hz units
        # Effective filter frequency
        nu_eff = (const.c / effective_wavelength(band)).to('Hz').value
        # Plot near-peak detections
        detections = lc.detect_csm(DET_SIGMA, count=DET_COUNT)
        ax.errorbar(detections[x_col], nu_eff * detections[y_col], 
                yerr=nu_eff * detections[yerr_col], label='%s (%s)' % (sn.name, band), 
                linestyle='none', ms=MS_DET, marker=MARKERS[sn.name], 
                color=COLORS[sn.name], mec='k', ecolor='k', elinewidth=1, 
                zorder=9, rasterized=True)
        
        # Plot nondetection limits
        # nondetections = lc(tmin=XLIM[0], tmax=XLIM[1]).drop(detections.index)
        # ax.scatter(nondetections[x_col], PLOT_SIGMA * nondetections[yerr_col], 
        #         marker='v', s=MS_DET**2, color=COLORS[sn.name], edgecolors='k', 
        #         alpha=ALPHA_DET, zorder=8)

    # Plot non-detection limits
    print('Importing GALEX detections and limits...')
    for sn_name in tqdm(sn_info.index):
        sn = Supernova(sn_name, sn_info)

        for band in ['FUV', 'NUV']:
            try:
                plot_nondetection_limits(ax, sn, band, x_col, yerr_col, ymax=YLIM[1])
            except (FileNotFoundError, pd.errors.EmptyDataError):
                continue

    # Import and plot Graham detections
    print('Importing HST detections...')
    nu_hst = (const.c / effective_wavelength('F275W')).to('Hz').value
    graham_data = pd.read_csv(Path('ref/Graham_observations.csv'), index_col=0)
    detections = graham_data[graham_data['Detection']]
    markers = ['X', '*']
    colors = ['y', 'r']
    for i, sn_name in enumerate(detections.index):
        obs = GrahamObservation(sn_name, graham_data)
        ax.scatter(obs.rest_phase, nu_hst * obs.luminosity_hz, marker=markers[i], 
                color=colors[i], edgecolors='k', label='%s (F275W)' % sn_name, 
                zorder=10,  s=64)

    print('Plotting...')

    # Format axes
    ax.set_xlabel('Time since discovery [rest-frame days]', size=12)
    # ax.set_ylabel('$L_\mathrm{UV}$ [erg s$^{-1}$ Hz$^{-1}$]', size=12)
    ax.set_ylabel('$\\nu L_\\nu$ [erg s$^{-1}$]', size=12)
    ax.set_xlim(XLIM)
    ax.set_yscale('log')
    ax.set_ylim(YLIM)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = [
            Line2D([0], [0], marker='v', markerfacecolor=COLORS['FUV'], 
                    markeredgecolor='none', markersize=MS_LOW, alpha=ALPHA_LOW,
                    label='detection limit (FUV)', lw=0),
            Line2D([0], [0], marker='v', markerfacecolor=COLORS['NUV'], 
                    markeredgecolor='none', markersize=MS_LOW, alpha=ALPHA_LOW,
                    label='detection limit (NUV)', lw=0)
    ]
    plt.legend(handles=handles + legend_elements, loc='lower center', ncol=3,
            handletextpad=0.5, handlelength=1.0, bbox_to_anchor=(0.5, 1.01))

    plt.savefig(Path('out/limits.pdf'), dpi=300)

    plt.show()


def plot_nondetection_limits(ax, sn, band, x_col, yerr_col, ymax=1e28):
    """Plot UV luminosity limits for GALEX non-detections."""

    lc = LightCurve(sn, band)
    lc.to_hz() # Convert to erg/s/Hz units

    # Effective filter frequency
    nu_eff = (const.c / effective_wavelength(band)).to('Hz').value

    detections = lc.detect_csm(DET_SIGMA, count=DET_COUNT)
    nondetections = lc(tmin=XLIM[0], tmax=XLIM[1]).drop(detections.index)

    # Plot nondetections below SN 2015cp luminosity
    below_cut = nondetections[PLOT_SIGMA * nondetections[yerr_col] <= LIMIT_CUTOFF]
    ax.scatter(below_cut[x_col], nu_eff * PLOT_SIGMA * below_cut[yerr_col], 
            marker='v', s=MS_LOW**2, color=COLORS[band], edgecolors='none', 
            alpha=ALPHA_LOW, zorder=3, rasterized=True)

    # Plot nondetections above SN 2015cp luminosity - fainter and smaller
    above_cut = nondetections[PLOT_SIGMA * nondetections[yerr_col] > LIMIT_CUTOFF]
    above_cut = above_cut[PLOT_SIGMA * above_cut[yerr_col] < ymax]
    ax.scatter(above_cut[x_col], nu_eff * PLOT_SIGMA * above_cut[yerr_col], 
            marker='v', s=MS_HI**2, color=COLORS[band], edgecolors='none', 
            alpha=ALPHA_HI, zorder=2, rasterized=True)

    return ax


if __name__ == '__main__':

    main()