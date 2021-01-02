from functools import reduce
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter, MultipleLocator
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, SymLogNorm
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import dill
from utils import *
from CSMmodel import CSMmodel

SIGMA = 3
CONF = 0.9

def main(iterations, t_min=TSTART_MIN, t_max=TSTART_MAX, scale_min=SCALE_MIN,
        scale_max=SCALE_MAX, bin_width=TSTART_BIN_WIDTH, y_bins=20,
        show_plot=True, model='Chev94', study='galex', cmax=None,
        sigma=SIGMA, plot_rates=False, overwrite=False, detections=False,
        upper_lim=False, cmin=None):
    
    # Bin edges
    x_edges = np.arange(t_min, t_max+bin_width, bin_width)
    y_edges = np.logspace(np.log10(scale_min), np.log10(scale_max), num=y_bins)

    # Define folder structure
    save_dir = run_dir(study, model, sigma)
    det_save_dir = run_dir(study + '_det', model, sigma)
    if detections:
        save_dir = det_save_dir
    fname = 'recovery_%s_%s' % (study, model)
    det_fname = 'recovery_%s_det_%s' % (study, model)
    if detections:
        fname = det_fname
    hist_file = OUTPUT_DIR / Path(fname + '.csv')
    det_hist_file = OUTPUT_DIR / Path(det_fname + '.csv')
    if upper_lim:
        plot_file = OUTPUT_DIR / Path(fname + '_upperlim.pdf')
    else:
        plot_file = OUTPUT_DIR / Path(fname + '.pdf')

    # List of files in save dir
    save_files = list(Path(save_dir).glob('*-%s.csv' % iterations))
    # Generate summed histogram
    if overwrite or not hist_file.is_file():
        print('\nImporting and summing saves...')
        hist = sum_hist(save_files, x_edges, y_edges, output_file=hist_file)
        if upper_lim and study == 'graham':
            det_save_files = list(Path(det_save_dir).glob('*-%s.csv' % iterations))
            det_hist = sum_hist(det_save_files, x_edges, y_edges, 
                    output_file=det_hist_file)
            hist = hist + det_hist
    else:
        print('\nImporting histogram...')
        hist = pd.read_csv(hist_file, index_col=0)
        if upper_lim and study == 'graham':
            det_hist = pd.read_csv(det_hist_file, index_col=0)
            hist = hist + det_hist

    # Calculate 90% binom conf interval upper limits
    if upper_lim:
        # Zero detections for GALEX
        if study == 'galex':
            zeros = np.zeros(hist.shape)
            det_hist = pd.DataFrame(zeros, index=hist.index, columns=hist.columns)

        # Binomial confidence interval
        hist = 100 * bci_nan(det_hist, hist)[1]

    # Plot histogram
    print('Plotting recovery histogram...')
    plot(x_edges, y_edges, hist, show=show_plot, output_file=plot_file, cmax=cmax,
            cmin=cmin, upper_lim=upper_lim)


def plot(x_edges, y_edges, hist, show=True, output_file='recovery.pdf', 
        cmax=None, cmin=None, upper_lim=False):
    """Plot 2D histogram of recovery rate by time since discovery and scale factor.
    Inputs:
        x_edges: x-axis bin edges
        y_edges: y-axis bin edges
        hist: 2D histogram
        show: whether to display plot
        output_file: output png plot file
        cmax: optional manual maximum value of colorbar
        cmin: optional manual minimum value of colorbar (must be >0)
        upper_lim: if True, plot upper 90% C.I. instead of number of excluded SNe
    """

    # Flip y-axis
    hist.sort_index(ascending=True, inplace=True)

    # Define colormap
    n_colors = 10 # number of distinct colors
    cmap_name = 'jet'
    if upper_lim:
        cmap_name += '_r' # reversed
    cmap = plt.cm.get_cmap(cmap_name)
    cmap.set_under('k') # set color for values under minimum
    cmap.set_over('k') # set color for values over maximum

    # colorbar limits
    if cmin <= 0:
        raise ValueError('cmin must be positive')
    if upper_lim:
        # set limit to highest value not at 100%
        hist_max = int(np.max(hist[hist < 100].to_numpy())) + 1
        hist_min = int(np.min(hist.to_numpy()))
    else:
        hist_max = int(np.max(hist.to_numpy())) + 1
        hist_min = 1
    if not cmax:
        cmax = hist_max
    if not cmin:
        cmin = hist_min

    # colormap bounds: (rounded) logarithmic scale
    cmap_bounds = np.logspace(np.log10(cmin), np.log10(cmax), num=n_colors)
    cmap_bounds[0] = int(cmap_bounds[0]) # first bound must round down
    cmap_bounds = np.round(cmap_bounds)
    norm = BoundaryNorm(cmap_bounds, cmap.N) # map boundaries onto colorbar

    # Plot
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(x_edges, y_edges, hist, cmap=cmap, norm=norm,
            edgecolor='k', linewidth=0.3, antialiased=True)

    # Format axes
    ax.set_yscale('log')
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel('$t_{start}$ [rest frame days post-discovery]')
    ax.set_ylabel('Scale factor')

    # Adjust colorbar: add extension below lower limit
    if upper_lim:
        cbar_label = 'Upper 90\% confidence [%]'
        bounds = list(cmap_bounds) + [100]
        extend = 'max'
    else:
        cbar_label = 'No. of excluded SNe Ia'
        bounds = [0] + list(cmap_bounds)
        extend = 'min'
    cbar = fig.colorbar(pcm, label=cbar_label, spacing='uniform', extend=extend, 
            boundaries=bounds, ticks=cmap_bounds, extendfrac='auto')
    cbar.ax.minorticks_off()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def sum_hist(save_files, x_edges, y_edges, save=True, output_file='recovery.csv',
        binary=False):
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
        func = partial(get_hist, x_edges=x_edges, y_edges=y_edges, binary=binary)
        imap = pool.imap(func, save_files, chunksize=10)
        for h in tqdm(imap, total=len(save_files)):
            hist.append(h)

    # Sum 2D histograms
    hist = reduce(lambda x, y: x.add(y, fill_value=0), hist)

    if save:
        hist.to_csv(output_file)

    return hist


def get_hist(fname, x_edges, y_edges, binary=False):
    """Import recovery save data and return histogram.
    Inputs:
        x_edges: list of x-axis bin edges
        y_edges: list of y-axis bin edges
        binary: if true, returns ceiling of recovery rate
    """

    rd = RecoveryData(fname)
    hist = rd.hist(x_edges, y_edges)
    if binary:
        hist = np.ceil(hist)
    return hist


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
    parser.add_argument('--cmax', type=float, help='Max colorbar value')
    parser.add_argument('--cmin', type=float, help='Minimum colorbar value')
    parser.add_argument('--tmax', default=TSTART_MAX, type=int, help='x-axis upper limit')
    parser.add_argument('--twidth', default=TSTART_BIN_WIDTH, type=int, help='x-axis bin width')
    parser.add_argument('-d', '--detections', action='store_true', 
            help='Recover models for G19 detections instead of nondetections')
    parser.add_argument('-u', '--upperlim', action='store_true', 
            help='plot upper limit of binomial conf intervals instead of excluded SNe')
    parser.add_argument('--smax', default=SCALE_MAX, type=int, help='y-axis upper limit')
    args = parser.parse_args()

    main(args.iterations, t_min=0, t_max=args.tmax, overwrite=args.overwrite, model=args.model, 
            study=args.study.lower(), sigma=args.sigma, cmax=args.cmax, 
            cmin=args.cmin, scale_max=args.smax, bin_width=args.twidth, 
            detections=args.detections, upper_lim=args.upperlim)