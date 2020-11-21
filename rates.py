from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import binom_conf_interval
from utils import *
from plot import sum_hist

TSTART_BINS = [0, 20, 100, 500, 1000]
CONF = 0.9
MODEL = 'Chev94'
SCALE = 50
GRAHAM_RATE = 6 # percent
SIGMA = 3


def main(tstart_bins=TSTART_BINS, scale=SCALE, iterations=10000, overwrite=False, 
        model=MODEL, sigma=SIGMA):

    # Import non-detections and sum histograms for GALEX and G19
    histograms = []
    for study in ['galex', 'graham']:
        # File names
        save_dir = run_dir(study, model, sigma)
        hist_file = save_dir / Path('out/rates_hist.csv')
        save_files = list(Path(save_dir).glob('*-%s.csv' % iterations))

        # Generate summed histogram
        if overwrite or not hist_file.is_file():
            print('\nImporting and summing saves...')
            hist = sum_hist(save_files, x_edges, y_edges, output_file=hist_file)
        else:
            print('\nImporting histograms...')
            hist = pd.read_csv(hist_file, index_col=0)
        
        histograms.append(hist.iloc[0].to_numpy())

    [galex_hist, graham_hist] = histograms
    uv_hist = galex_hist + graham_hist

    # Import detections
    graham_detections = pd.read_csv('ref/Graham_detections.csv')
    graham_det_hist = np.histogram(graham_detections['Rest Phase'], tstart_bins)[0]
    
    # Bin edges
    x_edges = np.array(tstart_bins)
    y_edges = np.array([scale, 1000])
    nbins = len(tstart_bins)-1
    x_pos = np.arange(nbins)
    x_step = 0.12

    # Row labels
    rows = []
    for i in range(nbins):
        rows.append('%s - %s' % (tstart_bins[i], tstart_bins[i+1]))

    # DataFrame for number of trials per tstart bin and data source
    sources = ['GALEX', 'G19', 'ASAS-SN', 'ZTF']
    trials = pd.DataFrame([], index=pd.Series(rows))
    trials['GALEX'] = galex_hist.T
    trials['G19'] = (graham_hist + graham_det_hist).T
    trials['UV'] = trials['GALEX'] + trials['G19']
    trials['ASAS-SN'] = np.array([464] + [0]*(nbins-1)).T
    trials['ZTF'] = np.array([127] + [0]*(nbins-1)).T
    trials['All'] = trials[sources].sum(axis=1)
    print(trials)

    # DataFrame for number of detections per tstart bin and data source
    detections = pd.DataFrame([], index=pd.Series(rows))
    detections['GALEX'] = np.zeros((nbins, 1))
    detections['G19'] = graham_det_hist.T
    detections['UV'] = detections['GALEX'] + detections['G19']
    detections['ASAS-SN'] = np.array([3] + [0]*(nbins-1)).T
    detections['ZTF'] = np.array([1] + [0]*(nbins-1)).T
    detections['All'] = detections[sources].sum(axis=1)
    print(detections)

    # Calculate binomial confidence intervals
    bci_lower = bci_upper = pd.DataFrame([], index=pd.Series(rows))
    for col in trials.columns:
        # separate bins with no trials
        pos_index = trials[trials[col] > 0].index
        zero_index = trials[trials[col] == 0].index

        bci = 100 * binom_conf_interval(detections.loc[pos_index, col], 
                trials.loc[pos_index, col], confidence_level=CONF, 
                interval='jeffreys')
        
        # add to dataframes (nan for no trials)
        bci_lower.loc[pos_index, col] = bci[0].T
        bci_lower.loc[zero_index,col] = np.nan
        bci_upper.loc[pos_index, col] = bci[1].T
        bci_upper.loc[zero_index,col] = np.nan

    print(bci_lower)
    print(bci_upper)

    fig, ax = plt.subplots()

    # Add Graham 2019
    x_adjust = np.array([-0.3] + [-x_step] * (nbins-1))
    graham_trials = graham_hist + graham_det_hist
    ax = plot_bci(ax, graham_det_hist, graham_trials, x_pos+x_adjust, label='G19',
            color='#004d40')[0]

    # Add our binomial confidence interval
    x_adjust += x_step
    ax = plot_bci(ax, 0, galex_hist, x_pos+x_adjust, color='#d81b60', 
            label='$\it{GALEX}$')[0]

    # UV combined rates
    x_adjust += x_step
    uv_trials = uv_hist + graham_det_hist
    ax = plot_bci(ax, graham_det_hist, uv_trials, x_pos+x_adjust, 
            label='UV combined', color='k', elinewidth=3.5)[0]

    # ASASSN
    asassn_det = 3
    asassn_trials = 464
    x_adjust = x_adjust[0] + x_step
    ax = plot_bci(ax, asassn_det, [asassn_trials], x_adjust, label='ASAS-SN', 
            color='#ffc107', marker='s')[0]

    # Zwicky Transient Facility
    ztf_det = 1
    ztf_trials = 127
    x_adjust += x_step
    ax = plot_bci(ax, ztf_det, [ztf_trials], x_adjust, color='#1e88e5', 
            label='ZTF', marker='s')[0]

    # All combined
    all_trials = uv_trials[0] + ztf_trials + asassn_trials
    all_det = graham_det_hist[0] + ztf_det + asassn_det
    x_adjust += x_step
    ax = plot_bci(ax, [all_det], [all_trials], x_adjust, label='All combined', 
            color='k', marker='*', ms=16)[0]

    # Format axis
    ax.set_xlim((x_pos[0]-0.7, x_pos[-1]+1.8))
    ax.set_xticks(np.append(x_pos, nbins)-0.5)
    ax.set_xticklabels(tstart_bins)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.set_xlabel('CSM interaction start time [rest frame days post-discovery]')
    ax.set_ylabel('Rate of CSM interaction [%]')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # remove errorbars
    handles = [h[0] for h in handles]
    plt.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(Path('out/rates_%s.pdf' % model), dpi=300)
    plt.show()


def table():
    pass


def plot():
    pass


def plot_bci(ax, detections, trials, x_pos, color='r', label='', conf_level=CONF, 
        elinewidth=2, marker='o', ms=10):

    bci = 100 * binom_conf_interval(detections, trials, 
            confidence_level=conf_level, interval='jeffreys')
    midpoint = np.mean(bci, axis=0)
    ax.errorbar(x_pos, midpoint, yerr=np.abs(bci - midpoint), 
            capsize=6, marker=marker, linestyle='none', ms=ms, mec=color, c=color, 
            mfc='w', label=label, elinewidth=elinewidth, mew=elinewidth)

    return ax, bci


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite histograms')
    parser.add_argument('--model', '-m', type=str, default='Chev94', help='CSM model spectrum')
    parser.add_argument('--scale', '-S', type=float, default=SCALE)
    parser.add_argument('--sigma', type=int, nargs='+', default=[SIGMA], 
            help='Detection confidence level (multiple for tiered detections)')
    args = parser.parse_args()

    main(overwrite=args.overwrite, model=args.model, scale=args.scale, 
            sigma=args.sigma)