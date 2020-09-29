from functools import reduce
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from utils import *


def main(iterations, t_min=TSTART_MIN, t_max=TSTART_MAX, scale_min=SCALE_MIN,
        scale_max=SCALE_MAX, bin_width=50, y_bins=20, overwrite=False,
        show_plot=False):

    # Bin edges
    x_edges = np.arange(t_min, t_max+bin_width, bin_width)
    y_edges = np.logspace(np.log10(scale_min), np.log10(scale_max), num=y_bins)

    # List of files in save dir
    save_files = list(Path(SAVE_DIR).glob('*-%s.csv' % iterations))
    # Generate summed histogram
    hist_file = Path('hist.csv')
    if overwrite or not (OUTPUT_DIR / hist_file).is_file():
        print('\nImporting and summing saves...')
        hist = sum_hist(save_files, x_edges, y_edges, output_file=hist_file)
    else:
        print('\nImporting histogram...')
        hist = pd.read_csv(OUTPUT_DIR / hist_file, index_col=0)

    # Plot histogram
    print('Plotting...')
    plot(x_edges, y_edges, hist, show=show_plot)


def plot(x_edges, y_edges, hist, show=False, output_file='recovery.png'):
    """Plot 2D histogram of recovery rate by time since discovery and scale factor."""

    # Flip y-axis
    hist.sort_index(ascending=True, inplace=True)

    # Plot
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(x_edges, y_edges, hist)
    ax.set_yscale('log')
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    # formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel('CSM interaction start time [rest frame days post-discovery]')
    ax.set_ylabel('Scale factor')
    plt.colorbar(pcm, label='No. of excluded SNe Ia')
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