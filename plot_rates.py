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

    fig, ax = plt.subplots()
    
    # Bin edges
    x_edges = np.array(tstart_bins)
    y_edges = np.array([scale, 1000])
    nbins = len(tstart_bins)-1
    x_pos = np.arange(nbins)

    # Import non-detections and sum histograms
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

    # Add our binomial confidence interval
    ax = plot_bci(ax, 0, galex_hist, x_pos, color='r', label='$\it{GALEX}$')

    # Add Graham 2019
    graham_trials = graham_hist + graham_det_hist
    ax = plot_bci(ax, graham_det_hist, graham_trials, x_pos, 
            color='g', label='G19', x_adjust=0.1)

    # UV combined rates
    uv_trials = uv_hist + graham_det_hist
    ax = plot_bci(ax, graham_det_hist, uv_trials, x_pos, 
            label='UV combined', x_adjust=0.2, color='k')

    # ASASSN
    asassn_det = 3
    asassn_trials = 464
    ax = plot_bci(ax, asassn_det, [asassn_trials], 0, color='y', x_adjust=-0.1, 
            label='ASAS-SN')

    # Zwicky Transient Facility
    ztf_det = 1
    ztf_trials = 127
    ax = plot_bci(ax, ztf_det, [ztf_trials], 0, color='b', x_adjust=-0.2, 
            label='ZTF')

    # All combined
    all_trials = uv_trials[0] + ztf_trials + asassn_trials
    all_det = graham_det_hist[0] + ztf_det + asassn_det
    ax = plot_bci(ax, [all_det], [all_trials], 0, label='All combined', 
            x_adjust=0.3, color='k')

    # x axis labels
    labels = []
    for i in range(nbins):
        labels.append('%s - %s' % (tstart_bins[i], tstart_bins[i+1]))

    # Format axis
    ax.set_xlim((x_pos[0]-0.5, x_pos[-1]+1.5))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.set_xlabel('CSM interaction start time [rest frame days post-discovery]')
    ax.set_ylabel('Rate of CSM interaction [%]')

    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(Path('out/rates_%s.pdf' % model), dpi=300)
    plt.show()


def plot_bci(ax, detections, trials, x_pos, color='r', label='', x_adjust=0., 
        conf_level=CONF):

    bci = 100 * binom_conf_interval(detections, trials, 
            confidence_level=conf_level, interval='jeffreys')
    midpoint = np.mean(bci, axis=0)
    ax.errorbar(x_pos+x_adjust, midpoint, yerr=np.abs(bci - midpoint), 
            capsize=10, marker='o', linestyle='none', ms=10, mec=color, c=color, 
            mfc='w', label=label)

    return ax


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('galex_dir', type=str)
    # parser.add_argument('graham_dir', type=str)
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite histograms')
    parser.add_argument('--model', '-m', type=str, default='Chev94', help='CSM model spectrum')
    parser.add_argument('--scale', '-S', type=float, default=SCALE)
    parser.add_argument('--sigma', type=int, nargs='+', default=[SIGMA], 
            help='Detection confidence level (multiple for tiered detections)')
    args = parser.parse_args()

    main(overwrite=args.overwrite, model=args.model, scale=args.scale, 
            sigma=args.sigma)