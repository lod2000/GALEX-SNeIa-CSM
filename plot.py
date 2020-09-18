from functools import reduce
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from utils import *


def main(iterations, t_min=TSTART_MIN, t_max=TSTART_MAX, scale_min=SCALE_MIN,
        scale_max=SCALE_MAX, bin_width=50, bin_height=0.1, overwrite=False):

    # Bin edges
    x_edges = np.arange(t_min, t_max+bin_width, bin_width)
    y_edges = np.arange(scale_min, scale_max+bin_height, bin_height)

    # Separate plot for each band
    for band in ['FUV', 'NUV']:
        # List of files in save dir
        save_files = get_save_files(iterations, band)
        # Generate summed histogram
        hist_file = Path('hist-%s.csv' % band)
        if overwrite or not (OUTPUT_DIR / hist_file).is_file():
            print('\nImporting and summing %s saves...' % band)
            hist = sum_hist(save_files, x_edges, y_edges, output_file=hist_file)
        else:
            print('\nImporting %s histogram...' % band)
            hist = pd.read_csv(OUTPUT_DIR / hist_file, index_col=0)

        # Plot histogram
        print('Plotting...')
        plot(hist, output_file='recovery-%s.png' % band)


def plot(hist, show=False, output_file='recovery.png'):
    """Plot 2D histogram of recovery rate by time since discovery and scale factor."""

    # Flip y-axis
    hist.sort_index(ascending=True, inplace=True)

    # Calculate data range
    x_bins = hist.columns.to_numpy(dtype=float)
    y_bins = hist.index.to_numpy(dtype=float)
    bin_width = x_bins[1] - x_bins[0]
    bin_height = y_bins[1] - y_bins[0]
    extent = (x_bins[0], x_bins[-1]+bin_width, y_bins[0], y_bins[-1]+bin_height)

    # Plot
    fig, ax = plt.subplots()
    im = ax.imshow(hist, aspect='auto', origin='lower', extent=extent)
    ax.xaxis.set_minor_locator(MultipleLocator(bin_width))
    ax.yaxis.set_minor_locator(MultipleLocator(bin_height))
    ax.set_xlabel('CSM interaction start time [rest frame days post-discovery]')
    ax.set_ylabel('Scale factor')
    plt.colorbar(im, label='No. of excluded SNe Ia')
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / Path(output_file), dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def sum_hist(save_files, x_edges, y_edges, save=True, output_file='hist.csv'):
    """Generate histograms for each save file and sum together.
    Inputs:
        save_files: list of recovery output CSVs
        x_edges: list of x-axis bin edges
        y_edges: list of y-axis bin edges
        save: save summed histogram as CSV
        output_file: output CSV file name
    Output:
        hist: summed histogram from all given save data
    """

    hist = []
    with Pool() as pool:
        func = partial(get_hist, x_edges=x_edges, y_edges=y_edges)
        imap = pool.imap(func, save_files, chunksize=10)
        for h in tqdm(imap, total=len(save_files)):
            hist.append(h)

    # Sum 2D histograms
    hist = reduce(lambda x, y: x.add(y, fill_value=0), hist)

    if save:
        hist.to_csv(OUTPUT_DIR / Path(output_file))

    return hist


def get_save_files(iterations, band='*', save_dir=SAVE_DIR):
    """Return list of files in save directory for given iterations"""

    save_files = list(Path(save_dir).glob('*-%s-%s.csv' % (band, iterations)))
    return save_files


def get_hist(fname, x_edges, y_edges):
    """Import recovery save data and return histogram.
    Inputs:
        x_edges: list of x-axis bin edges
        y_edges: list of y-axis bin edges
    """

    rd = RecoveryData(fname)
    return rd.hist(x_edges, y_edges)


class RecoveryData:
    def __init__(self, fname):
        """Import recovery save data.
        Input:
            fname: path to CSV
        """

        # Import save file; convert columns from strings to lists
        split_list = lambda x: x[1:-1].split(', ')
        data = pd.read_csv(fname, 
                converters={'recovered_times': split_list, 
                            'all_times': split_list},
        )
        
        # Join lists of recovered times and all times
        join_list = lambda x: [float(t) for l in x for t in l if t != '']
        self.recovered_times = join_list(data.recovered_times)
        self.all_times = join_list(data.all_times)

        # Join lists of recovered scales and all scales
        count_param = lambda x, y: [data.loc[i,y] for i, l in enumerate(x) 
                for t in l if t != '']
        self.recovered_scales = count_param(data.recovered_times, 'scale')
        self.all_scales = count_param(data.all_times, 'scale')

        # Join lists of CSM interaction start times
        self.recovered_tstarts = count_param(data.recovered_times, 'tstart')
        self.all_tstarts = count_param(data.all_times, 'tstart')


    def hist(self, x_edges, y_edges):
        """Generate 2D histogram of recovery rate according to given bin edges.
        Inputs:
            x_edges: list of x-axis bin edges
            y_edges: list of y-axis bin edges
        """

        # 2D histograms for recovered data and total data
        recovered = np.histogram2d(self.recovered_tstarts, self.recovered_scales, 
                [x_edges, y_edges])[0]
        total = np.histogram2d(self.all_tstarts, self.all_scales, 
                [x_edges, y_edges])[0]

        # Calculate recovery rate
        rate_hist = recovered / total

        # Transpose and convert to DataFrame with time increasing along the rows
        # and scale height increasing down the columns. Column and index labels
        # are the lower bound of each bin
        rate_hist = pd.DataFrame(rate_hist.T, index=y_edges[:-1], columns=x_edges[:-1])
        return rate_hist


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', '-i', type=int, default=10000, help='Iterations')
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite histograms')
    args = parser.parse_args()

    main(args.iterations, t_min=0, overwrite=args.overwrite)