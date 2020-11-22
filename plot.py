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
        scale_max=SCALE_MAX, bin_width=50, y_bins=20, overwrite=False,
        show_plot=True, model='Chev94', study='galex', cmax=None,
        sigma=SIGMA, plot_rates=False):

    # Scale to UV luminosity of SN 2015cp
    reduced_scale = SN2015cp_scale(model)
    
    # Bin edges
    x_edges = np.arange(t_min, t_max+bin_width, bin_width)
    y_edges = np.logspace(np.log10(scale_min), np.log10(scale_max), num=y_bins) 
    y_edges /= reduced_scale

    # Define folder structure
    save_dir = run_dir(study, model, sigma)
    run_out_dir = save_dir / Path('out')
    if not run_out_dir.is_dir(): run_out_dir.mkdir()
    hist_file = run_out_dir / Path('hist.csv')
    plot_file = OUTPUT_DIR / Path('recovery_%s_%s.pdf' % (study, model))

    # List of files in save dir
    save_files = list(Path(save_dir).glob('*-%s.csv' % iterations))
    # Generate summed histogram
    if overwrite or not hist_file.is_file():
        print('\nImporting and summing saves...')
        hist = sum_hist(save_files, x_edges, y_edges, output_file=hist_file,
                reduced_scale=reduced_scale)
    else:
        print('\nImporting histogram...')
        hist = pd.read_csv(hist_file, index_col=0)

    # Plot histogram
    print('Plotting recovery histogram...')
    plot(x_edges, y_edges, hist, show=show_plot, output_file=plot_file, cmax=cmax)


def plot(x_edges, y_edges, hist, show=True, output_file='recovery.pdf', cmax=None):
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
    cbin_width = int(hist_max / 12)
    cmap_bounds = np.arange(1, hist_max, cbin_width)
    # include lower, uppper bounds
    cmap_bounds = np.concatenate(([0], cmap_bounds, [hist_max]))
    norm = BoundaryNorm(cmap_bounds, cmap.N)

    # Plot
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(x_edges, y_edges, hist, cmap=cmap, norm=norm, vmin=0)
    ax.set_yscale('log')
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel('CSM interaction start time [rest frame days post-discovery]')
    ax.set_ylabel('Relative scale factor')

    # Color bar
    cbar = plt.colorbar(pcm, label='No. of excluded SNe Ia', spacing='proportional')
    cbar.ax.yaxis.set_minor_locator(MultipleLocator(int(hist_max/24)))

    fig.tight_layout()
    plt.savefig(output_file, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def sum_hist(save_files, x_edges, y_edges, save=True, output_file='hist.csv', **kwargs):
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
        func = partial(get_hist, x_edges=x_edges, y_edges=y_edges, **kwargs)
        imap = pool.imap(func, save_files, chunksize=10)
        for h in tqdm(imap, total=len(save_files)):
            hist.append(h)

    # Sum 2D histograms
    hist = reduce(lambda x, y: x.add(y, fill_value=0), hist)

    if save:
        hist.to_csv(output_file)

    return hist


def get_hist(fname, x_edges, y_edges, **kwargs):
    """Import recovery save data and return histogram. Also scale to SN 2015cp.
    Inputs:
        x_edges: list of x-axis bin edges
        y_edges: list of y-axis bin edges
    """

    rd = RecoveryData(fname, **kwargs)
    return rd.hist(x_edges, y_edges)


class RecoveryData:
    def __init__(self, fname, reduced_scale=1):
        """Import recovery save data.
        Input:
            fname: path to CSV
        """

        self.fname = fname
        # Import save file; convert columns from strings to lists
        data = pd.read_csv(fname)
        data['recovered'] = data['recovered_times'].str.len() > 2

        self.recovered_scales = data[data['recovered']]['scale'].to_numpy() / reduced_scale
        self.all_scales = data['scale'].to_numpy() / reduced_scale
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
    args = parser.parse_args()

    main(args.iterations, t_min=0, overwrite=args.overwrite, model=args.model, 
            study=args.study.lower(), sigma=args.sigma, cmax=args.max)