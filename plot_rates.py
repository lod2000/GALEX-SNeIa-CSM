from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import binom_conf_interval
from utils import *
from plot import sum_hist

TSTART_BINS = [0, 100, 500, 1000]
CONF = 0.9
MODEL = 'Chev94'
SCALE = 50
GRAHAM_RATE = 6 # percent


def main(galex_run_dir, graham_run_dir, tstart_bins=TSTART_BINS, scale=SCALE, 
        iterations=10000, overwrite=False, model=MODEL):

    fig, ax = plt.subplots()
    
    # Bin edges
    x_edges = np.array(tstart_bins)
    y_edges = np.array([scale, 1000])
    nbins = len(tstart_bins)-1
    x_pos = np.arange(nbins)

    # Sum histograms
    for run_dir in [galex_run_dir, graham_run_dir]:
        # File names
        hist_file = run_dir / Path('out/rates_hist.csv')
        save_files = list(Path(run_dir).glob('*-%s.csv' % iterations))

        # Generate summed histogram
        if overwrite or not hist_file.is_file():
            print('\nImporting and summing saves...')
            hist = sum_hist(save_files, x_edges, y_edges, output_file=hist_file)
    
    print('\nImporting histograms...')
    galex_hist = pd.read_csv(galex_run_dir/Path('out/rates_hist.csv'), index_col=0)
    graham_hist = pd.read_csv(graham_run_dir/Path('out/rates_hist.csv'), index_col=0)

    # Add our binomial confidence interval
    ax = plot_bci(ax, 0, galex_hist.iloc[0], x_pos, color='r', label='This study')

    # Add Graham 2019
    ax = plot_bci(ax, [0,1,1], graham_hist.iloc[0], x_pos, color='g', label='G19',
            x_adjust=-0.1)
    ax.scatter([1.9], [GRAHAM_RATE], marker='v', color='g', s=100, label='G19 reported')

    # ASASSN
    ax = plot_bci(ax, 3, [460], 0, color='y', x_adjust=0.1, label='ASAS-SN')

    # Zwicky Transient Facility
    ax = plot_bci(ax, 1, [127], 0, color='b', x_adjust=0.2, label='ZTF')

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
    plt.savefig(Path('out/rates_%s.png' % model), dpi=300)
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
    parser.add_argument('galex_dir', type=str)
    parser.add_argument('graham_dir', type=str)
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite histograms')
    parser.add_argument('--model', '-m', type=str, default='Chev94', help='CSM model spectrum')
    parser.add_argument('--scale', '-S', type=float, default=SCALE)
    args = parser.parse_args()

    main(args.galex_dir, args.graham_dir, overwrite=args.overwrite, 
            model=args.model, scale=args.scale)