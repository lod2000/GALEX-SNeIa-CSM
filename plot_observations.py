#!/usr/bin/env python

import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import astropy.units as u
from astropy.stats import binom_conf_interval
import argparse
from utils import *

DET_SIGMA = 3 # detection threshold
LIMIT_SIGMA = 1 # factor to inflate uncertainty for upper limits


def main():
    parser = argparse.ArgumentParser(description='Plot detection limits.')
    parser.add_argument('-o', '--overwrite', action='store_true',
            help='re-concatenate detection and nondetection data.')
    parser.add_argument('-s', '--systematics', action='store_true',
            help='plot observation and sample systematics.')
    parser.add_argument('--show', action='store_true', help='show plot after saving')
    parser.add_argument('--presentation', action='store_true',
            help='configure plots for presentation')
    args = parser.parse_args()

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
    conf_det = pd.read_csv(Path('out/confirmed_detections.csv'))
    det_sne = list(zip(conf_det['Name'], conf_det['Band']))

    if args.overwrite or not Path('out/nondetections.csv').is_file():
        print('Separating detections from nondetections...')
        detections = aggregate_detections(det_sne, sn_info)
        nondetections = aggregate_nondetections(det_sne, sn_info)
    else:
        detections = pd.read_csv(Path('out/detections.csv'))
        nondetections = pd.read_csv(Path('out/nondetections.csv'))

    if args.systematics:
        print('Plotting systematics...')
        # Look for systematics in observations
        all_detections = nondetections.append(detections)
        all_detections.set_index('name', inplace=True)
        plot_observation_systematics(all_detections, sn_info)
        plot_sample_systematics(sn_info)

    print('Plotting detections & limits...')

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    ebar_alpha = 0.8
    limit_alpha = 0.6
    nondet_alpha = 0.05
    faint_alpha = 0.3
    upper_lim = 1e28
    lower_lim = 1e22 if args.presentation else None
    cutoff = 10**25.88 # Graham 2015cp detection
    det_ms = 6 # detection marker size

    markers = ['o', 's', 'p', 'd', 'P']
    colors = ['cyan', 'orange', 'green', 'magenta']

    # Plot Swift SN2011fe from Brown+ 2012
    if args.presentation:
        band = 'UVW1'
    else:
        band = 'UVM2'
    SN2011fe = pd.read_csv(Path('external/SN2011fe_Brown2012.tsv'), sep='\t',
            comment='#', skiprows=[45, 46])
    SN2011fe = SN2011fe[pd.notna(SN2011fe['mag'])]
    SN2011fe['t_delta'] = SN2011fe['MJD'] - Time('2011-08-24', format='iso').mjd
    lc = SN2011fe[SN2011fe['Filt'] == band.lower()].copy()
    dist = 6.4 # Mpc; from Shappee & Stanek 2011
    z = 0 # too close to need correction
    a_v = 0 # won't worry about it right now
    a_band = 'NUV' # close enough
    lc['FluxDensity'], lc['e_FluxDensity'] = swift_cps2flux(lc['CRate'], 
            lc['e_CRate'], band)
    lc['Luminosity'] = flux2luminosity(lc['FluxDensity'], dist, z, a_v, a_band)
    lc['Luminosity_hz'] = wavelength2freq(lc['Luminosity'], 2245.8)
    ax.plot(lc['t_delta'], lc['Luminosity_hz'], color='brown', 
            label='SN2011fe (%s)' % band, zorder=1)

    # Plot near-peak and CSM detections
    for i, (sn, band) in enumerate(det_sne):
        lc = detections[(detections['name'] == sn) & (detections['band'] == band)]
        lc_det = lc[lc['sigma'] > DET_SIGMA]
        lc_non = lc[lc['sigma'] <= DET_SIGMA]
        if args.presentation:
            # Plot nondetection limits of near-peak SNe
            plot_luminosity_limit(ax, lc_non, s=36, c=COLORS[band], 
                    a=faint_alpha, e='none', z=2)
        else:
            ax.errorbar(lc_det['t_delta_rest'], lc_det['luminosity_hostsub_hz'],
                    yerr=lc_det['luminosity_hostsub_err_hz'], linestyle='none', 
                    label='%s (%s)' % (sn, band), marker=markers[i], ms=det_ms,
                    markeredgecolor='k', color=colors[i], ecolor='k', elinewidth=1, zorder=9)
            # Plot nondetection limits of near-peak SNe
            plot_luminosity_limit(ax, lc_non, s=det_ms**2, c=colors[i], 
                    a=limit_alpha, e='k', z=8)

    # Plot nondetections
    for band in ['FUV', 'NUV']:
        lc = nondetections[nondetections['band'] == band]
        # Make distant (bright) limits smaller
        bright = lc[LIMIT_SIGMA * lc['luminosity_hostsub_err_hz'] >= cutoff]
        bright = bright[LIMIT_SIGMA * bright['luminosity_hostsub_err_hz'] < upper_lim]
        plot_luminosity_limit(ax, bright, s=16, c=COLORS[band], a=nondet_alpha, 
                e='none', z=2, rasterized=True)
        # Make close (faint) limits bigger
        faint = lc[LIMIT_SIGMA * lc['luminosity_hostsub_err_hz'] < cutoff]
        plot_luminosity_limit(ax, faint, s=36, c=COLORS[band], a=faint_alpha, 
                e='none', z=3)

    # Plot Graham detections
    # note: Graham uses days past explosion, not discovery
    if args.presentation:
        ax.axhline(y=cutoff, color='r', label='SN2015cp (F275W)', zorder=10)
    else:
        ax.scatter(686, 10**25.88, marker='*', s=100, color='r', edgecolors='k', 
                label='SN2015cp (F275W)', zorder=10)
        ax.scatter(477, 10**26.06, marker='X', s=64, color='y', edgecolors='k', 
                label='ASASSN-15og (F275W)', zorder=10)

    ax.set_xlabel('Rest frame time since discovery [days]')
    # ax.set_xlabel('Observed time since discovery [days]')
    ax.set_xlim((-50, np.max(faint['t_delta_rest']) + 50))
    ax.set_ylabel('UV Luminosity [erg s$^{-1}$ Hz$^{-1}$]')
    ax.set_yscale('log')
    ax.set_ylim((lower_lim, upper_lim))

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = [
            Line2D([0], [0], marker='v', markerfacecolor=COLORS['FUV'], 
                    markeredgecolor='none', markersize=6, alpha=faint_alpha,
                    label='detection limit (FUV)', lw=0),
            Line2D([0], [0], marker='v', markerfacecolor=COLORS['NUV'], 
                    markeredgecolor='none', markersize=6, alpha=faint_alpha,
                    label='detection limit (NUV)', lw=0)
    ]
    ncol = 2 if args.presentation else 3
    plt.legend(handles=handles + legend_elements, loc='upper right', ncol=ncol,
            handletextpad=0.2, handlelength=1.0)

    plt.savefig(Path('figs/limits.pdf'), dpi=300)
    if args.show:
        plt.show()
    else:
        plt.close()

    # Binomial statistics plot
    fig, ax = plt.subplots()

    conf_level = 0.9
    # Include all nondetections below the luminosity of 2015cp
    below_graham = nondetections[nondetections['luminosity_hostsub_err_hz'] * LIMIT_SIGMA < cutoff]
    # Also include limits from near-peak SNe
    below_graham.append(lc_non[lc_non['luminosity_hostsub_err_hz'] * LIMIT_SIGMA < cutoff])
    # Only those after discovery
    below_graham = below_graham[below_graham['t_delta_rest'] > 0]
    print('Number of SNe with limits fainter than 2015cp: %s' % len(below_graham.drop_duplicates('name').index))
    print('Number of observations with limits fainter than 2015cp: %s' % len(below_graham.index))
    bins = [0, 100, 500, 2500]
    k = []
    n = []
    labels = []
    for i in range(len(bins)-1):
        limits = below_graham[(below_graham['t_delta_rest'] >= bins[i]) & (below_graham['t_delta_rest'] < bins[i+1])]
        discrete_sne = limits.drop_duplicates('name')
        k.append(0)
        n.append(len(discrete_sne.index))
        labels.append('%s - %s' % (bins[i], bins[i+1]))
    print(bins)
    print(n)
    bci = 100 * binom_conf_interval(k, n, confidence_level=conf_level, interval='jeffreys')
    print(bci)
    midpoint = np.mean(bci, axis=0)
    x_pos = np.arange(len(bins)-1)
    ax.errorbar(x_pos, midpoint, yerr=np.abs(bci - midpoint), capsize=10, 
            marker='o', linestyle='none', ms=10, mec='r', c='r', mfc='w',
            label='This study')

    # Confidence interval from Yao 2019
    ztf_bci = 100 * binom_conf_interval(1, 127, confidence_level=conf_level, interval='jeffreys')
    print(ztf_bci)
    ztf_mean = np.mean(ztf_bci)
    ax.errorbar([0.1], [ztf_mean], yerr=([ztf_mean - ztf_bci[0]], [ztf_bci[1] - ztf_mean]),
            marker='o', c='b', linestyle='none', ms=10, capsize=10, mec='b', mfc='w',
            label='ZTF')

    # ASAS-SN interval
    asassn_bci = 100 * binom_conf_interval(3, 460, confidence_level=conf_level, interval='jeffreys')
    print(asassn_bci)
    asassn_mean = np.mean(asassn_bci)
    ax.errorbar([0.2], [asassn_mean], yerr=([asassn_mean - asassn_bci[0]], [asassn_bci[1] - asassn_mean]),
            marker='o', c='orange', linestyle='none', ms=10, capsize=10, mec='orange', mfc='w',
            label='ASAS-SN')

    # Confidence interval & assumed late-onset rate from Graham 2019
    graham_rate = 6
    graham_bci = 100 * binom_conf_interval(1, 64, confidence_level=conf_level, interval='jeffreys')
    print(graham_bci)
    ax.errorbar([2.1], [graham_rate], yerr=([graham_rate - graham_bci[0]], [graham_bci[1] - graham_rate]),
            marker='v', color='g', linestyle='none', ms=15, capsize=10, label='G19')
    # ax.annotate('G19', (2.1, graham_rate), textcoords='offset points', 
    #         xytext=(10, 0), ha='left', va='center', size=18, color='g')

    ax.set_xlim((x_pos[0]-0.5, x_pos[-1]+0.5))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.set_xlabel('Rest frame time since discovery [days]')
    ax.set_ylabel('Rate of CSM interaction [%]')

    # Preliminary!
    if args.presentation:
        fig.text(0.95, 0.05, 'PRELIMINARY', fontsize=72, color='gray', 
                rotation='30', ha='right', va='bottom', alpha=0.5)

    plt.tight_layout()
    plt.legend()
    plt.savefig(Path('figs/rates.png'), dpi=300)
    if args.show:
        plt.show()
    else:
        plt.close()


def plot_luminosity_limit(ax, data, s=36, c='k', a=1.0, e='none', z=1, rasterized=False):
    """
    Plots nondetection limits.
    Optional inputs: size, color, alpha, edge color, zorder
    """
    
    dt = data['t_delta_rest']
    limit = LIMIT_SIGMA * data['luminosity_hostsub_err_hz']
    ax.scatter(dt, limit, marker='v', s=s, color=c, alpha=a, edgecolors=e, zorder=z,
            rasterized=rasterized)


def add_uniform_columns(df, values, col_names):
    new_cols = np.full([len(df.index), len(values)], values)
    new_cols = pd.DataFrame(new_cols, columns=col_names, index=df.index)
    df = df.join(new_cols)
    return df


def aggregate_detections(det_sne, sn_info):
    detections = []
    for sn, band in det_sne:
        lc, bg, bg_err, sys_err = full_import(sn, band, sn_info)
        lc = add_uniform_columns(lc, [sn, band, bg, bg_err, sys_err], 
                ['name', 'band', 'host', 'host_err', 'sys_err'])
        # lc['sigma'] = lc['flux_hostsub_hz'] / lc['flux_hostsub_err_hz']
        detections.append(lc)
    detections = pd.concat(detections, ignore_index=True)
    output_csv(detections, Path('out/detections.csv'), index=False)
    return detections


def aggregate_nondetections(det_sne, sn_info):
    bands = ['FUV', 'NUV'] * len(sn_info.index)
    sne = np.array([[sn] * 2 for sn in sn_info.index]).flatten()
    nondet_sne = zip(sne, bands)
    # Remove detections
    nondet_sne = [loc for loc in nondet_sne if not loc in det_sne]
    with mp.Pool() as pool:
        nondetections = list(tqdm(
            pool.imap(partial(get_nondetection, sn_info=sn_info), 
                nondet_sne, chunksize=10), 
            total=len(nondet_sne)
        ))
    nondetections = [lc for lc in nondetections if len(lc.index) > 0]
    nondetections = pd.concat(nondetections, ignore_index=True)
    output_csv(nondetections, Path('out/nondetections.csv'), index=False)
    return nondetections


def get_nondetection(loc, sn_info):
    sn, band = loc
    try:
        lc, bg, bg_err, sys_err = full_import(sn, band, sn_info)
        lc = add_uniform_columns(lc, [sn, band, bg, bg_err, sys_err], 
                ['name', 'band', 'host', 'host_err', 'sys_err'])
        return lc
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        return pd.DataFrame([])


def plot_observation_systematics(all_detections, sn_info):
    """
    Plots the distribution of GALEX observations over time
    """

    fig, ax = plt.subplots(2,2, figsize=(13, 8))
    fig.set_tight_layout(True)

    # Distribution of GALEX observations of targets (GPS time)
    gps_sub = 740063000
    ax[0, 0].hist(all_detections['t_mean'] - gps_sub, bins=100)
    ax[0, 0].set_xlabel('Observation date [GPS-%s]' % gps_sub)
    ax[0, 0].set_ylabel('# of obs.')

    # Distribution of GALEX observations (MJD)
    ax[0, 1].hist(all_detections['t_mean_mjd'], bins=100)
    ax[0, 1].set_xlabel('Observation date [MJD]')
    ax[0, 1].set_ylabel('# of obs.')

    # Distribution of SN discovery dates among targets
    ax[1, 0].hist(Time(sn_info['disc_date'].to_list(), format='iso').mjd, bins=100)
    ax[1, 0].set_xlabel('Discovery date [MJD]')
    ax[1, 0].set_ylabel('# of SNe')

    # Distribution of GALEX observations of targets (time since discovery)
    ax[1, 1].hist(all_detections['t_delta'], bins=100)
    ax[1, 1].set_xlabel('Observed time since discovery [days]')
    ax[1, 1].set_ylabel('# of obs.')

    plt.savefig(Path('out/obs_dist.png'), dpi=300)
    plt.close()

    all_detections.sort_values('t_mean_mjd', inplace=True)
    gaps = all_detections['t_mean_mjd'].diff()
    print('Longest gap between consecutive observations: %s days' % np.max(gaps))
    all_detections.sort_values('t_delta', inplace=True)
    gaps = all_detections['t_delta'].diff()
    print('Longest gap in phase between consecutive points: %s days' % np.max(gaps))
    gaps = all_detections[all_detections['t_delta'] > 0]['t_delta'].diff()
    print('Longest phase gap post-discovery: %s days' % np.max(gaps))


def plot_sample_systematics(sn_info):
    """
    Plots the distribution of sky positions and discovery dates for SNe in the
    final target sample, as well as from the initial OSC sample
    """

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    fig.set_tight_layout(True)

    osc = pd.read_csv(Path('ref/osc.csv'), index_col='Name')

    ra = [Angle(ang, unit=u.hourangle).hour for ang in osc['R.A.'].to_list()]
    ax[0, 0].hist(ra, bins=100)
    ax[0, 0].set_ylabel('# of SNe in OSC')

    dec = [Angle(ang, unit=u.deg).deg for ang in osc['Dec.'].to_list()]
    ax[0, 1].hist(dec, bins=100)

    disc_date = Time(osc['Disc. Date'].to_list())
    disc_date = [d.mjd for d in disc_date if d.mjd > 40000]
    ax[0, 2].hist(disc_date, bins=100)

    ra = [Angle(ang).hour for ang in sn_info['galex_ra'].to_list()]
    ax[1, 0].hist(ra, bins=100)
    ax[1, 0].set_xlabel('R.A. [hours]')
    ax[1, 0].set_ylabel('# of SNe in sample')

    dec = [Angle(ang).deg for ang in sn_info['galex_dec'].to_list()]
    ax[1, 1].hist(dec, bins=100)
    ax[1, 1].set_xlabel('Dec. [deg]')

    # Distribution of SN discovery dates among targets
    ax[1, 2].hist(Time(sn_info['disc_date'].to_list(), format='iso').mjd, bins=100)
    ax[1, 2].set_xlabel('Discovery date [MJD]')

    plt.savefig(Path('out/target_coord_dist.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    main()