from functools import reduce
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter, MultipleLocator
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import dill
from utils import *
from CSMmodel import CSMmodel

SIGMA = 3

def main(iterations, t_min=TSTART_MIN, t_max=TSTART_MAX, scale_min=SCALE_MIN,
        scale_max=SCALE_MAX, bin_width=TSTART_BIN_WIDTH, y_bins=20,
        show_plot=True, model='Chev94', study='galex', cmax=None,
        sigma=SIGMA, plot_rates=False, overwrite=False, detections=False):
    
    # Bin edges
    x_edges = np.arange(t_min, t_max+bin_width, bin_width)
    y_edges = np.logspace(np.log10(scale_min), np.log10(scale_max), num=y_bins)

    # Define folder structure
    save_dir = run_dir(study, model, sigma)
    if detections:
        save_dir = run_dir(study + '_det', model, sigma)
    fname = 'recovery_%s_%s' % (study, model)
    if detections:
        fname = 'recovery_%s_det_%s' % (study, model)
    hist_file = OUTPUT_DIR / Path(fname + '.csv')
    plot_file = OUTPUT_DIR / Path(fname + '.pdf')

    # List of files in save dir
    save_files = list(Path(save_dir).glob('*-%s.csv' % iterations))
    # Generate summed histogram
    if overwrite or not hist_file.is_file():
        print('\nImporting and summing saves...')
        hist = sum_hist(save_files, x_edges, y_edges, output_file=hist_file)
    else:
        print('\nImporting histogram...')
        hist = pd.read_csv(hist_file, index_col=0)

    # Plot histogram
    print('Plotting recovery histogram...')
    plot(x_edges, y_edges, hist, show=show_plot, output_file=plot_file, cmax=cmax,
            detections=detections)


def plot(x_edges, y_edges, hist, show=True, output_file='recovery.pdf', cmax=None,
        detections=False):
    """Plot 2D histogram of recovery rate by time since discovery and scale factor.
    Inputs:
        x_edges: x-axis bin edges
        y_edges: y-axis bin edges
        hist: 2D histogram
        show: whether to display plot
        output_file: output png plot file
        cbin_width: width of colormap bins
    """

    # Flip y-axis
    hist.sort_index(ascending=True, inplace=True)

    # Colormap
    cmap = plt.cm.jet # colormap of choice
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist = [(0, 0, 0, 1)] + cmaplist # add black for 0-1
    cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    hist_max = int(np.max(hist.to_numpy()))+1 # max value of histogram
    hist_max = hist_max if cmax == None else cmax
    if hist_max > 12:
        cbin_width = int(hist_max / 12)
        cmap_bounds = np.arange(1, hist_max, cbin_width)
    else:
        hist_max = np.max(hist.to_numpy())
        cbin_width = hist_max / 10
        cmap_bounds = np.arange(0.01, hist_max, cbin_width)
    # include lower, uppper bounds
    cmap_bounds = np.concatenate(([0], cmap_bounds, [hist_max]))
    norm = BoundaryNorm(cmap_bounds, cmap.N)

    # Plot
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(x_edges, y_edges, hist, cmap=cmap, norm=norm, vmin=0,
            edgecolor='k', linewidth=0.3, antialiased=True)
    ax.set_yscale('log')
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel('$t_{start}$ [rest frame days post-discovery]')
    ax.set_ylabel('Scale factor')

    # Color bar
    if detections:
        cbar_label = 'Fraction of possible models'
    else:
        cbar_label = 'No. of excluded SNe Ia'
    cbar = plt.colorbar(pcm, label=cbar_label, spacing='proportional')
    if hist_max > 24:
        cbar.ax.yaxis.set_minor_locator(MultipleLocator(int(hist_max/24)))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def sum_hist(save_files, x_edges, y_edges, save=True, output_file='recovery.csv'):
    """Generate histograms for each save file and sum together.
    Inputs:
        save_files: list of recovery output CSVs
        model: 'Chev94' or 'flat'
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
        hist.to_csv(output_file)

    return hist


def get_hist(fname, x_edges, y_edges):
    """Import recovery save data and return histogram. Also scale to SN 2015cp.
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

        self.fname = fname
        # Import save file; convert columns from strings to lists
        data = pd.read_csv(fname)

        self.recovered_scales = data[data['recovered']]['scale'].to_numpy()
        self.all_scales = data['scale'].to_numpy()
        self.recovered_tstarts = data[data['recovered']]['tstart'].to_numpy()
        self.all_tstarts = data['tstart'].to_numpy()


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
    dill.settings['recurse']=True

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', '-i', type=int, default=10000, help='Iterations')
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite histograms')
    parser.add_argument('--model', '-m', type=str, default='Chev94', help='CSM model spectrum')
    parser.add_argument('--study', '-s', type=str, default='galex', help='Study from which to pull data')
    parser.add_argument('--sigma', type=int, nargs='+', default=[SIGMA], 
            help='Detection confidence level (multiple for tiered detections)')
    parser.add_argument('--max', type=float, help='Max colorbar value')
    parser.add_argument('--tmax', default=TSTART_MAX, type=int, help='x-axis upper limit')
    parser.add_argument('--twidth', default=TSTART_BIN_WIDTH, type=int, help='x-axis bin width')
    parser.add_argument('-d', '--detections', action='store_true', 
            help='Recover models for G19 detections instead of nondetections')
    parser.add_argument('--smax', default=SCALE_MAX, type=int, help='y-axis upper limit')
    args = parser.parse_args()

    main(args.iterations, t_min=0, t_max=args.tmax, overwrite=args.overwrite, model=args.model, 
            study=args.study.lower(), sigma=args.sigma, cmax=args.max, scale_max=args.smax,
            bin_width=args.twidth, detections=args.detections)