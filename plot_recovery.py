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
        sigma=SIGMA, overwrite=False, detections=False,
        upper_lim=False, cmin=None):
    
    # Bin edges
    x_edges = np.arange(t_min, t_max+bin_width, bin_width)
    y_edges = np.logspace(np.log10(scale_min), np.log10(scale_max), num=y_bins)

    # Define folder structure
    save_dir = run_dir(study, model, sigma)
    det_save_dir = run_dir(study + '_det', model, sigma)
    # if detections:
    #     save_dir = det_save_dir
    fname = 'recovery_%s_%s' % (study, model)
    det_fname = 'recovery_%s_det_%s' % (study, model)
    # if detections:
    #     fname = det_fname
    hist_file = OUTPUT_DIR / Path(fname + '.csv')
    det_hist_file = OUTPUT_DIR / Path(det_fname + '.csv')
    if upper_lim:
        plot_file = OUTPUT_DIR / Path(fname + '_upperlim.pdf')
    elif detections:
        plot_file = OUTPUT_DIR / Path(fname + '_det.pdf')        
    else:
        plot_file = OUTPUT_DIR / Path(fname + '.pdf')

    # List of files in save dir
    save_files = list(Path(save_dir).glob('*-%s.csv' % iterations))
    # Generate summed histogram
    if overwrite or not hist_file.is_file():
        print('\nImporting and summing saves...')
        hist = sum_hist(save_files, x_edges, y_edges, output_file=hist_file)
        if (detections or upper_lim) and study == 'graham':
            det_save_files = list(Path(det_save_dir).glob('*-%s.csv' % iterations))
            det_hist = sum_hist(det_save_files, x_edges, y_edges, 
                    output_file=det_hist_file)
        if upper_lim and study == 'graham':
            hist = hist + det_hist
    else:
        print('\nImporting histogram...')
        hist = pd.read_csv(hist_file, index_col=0)
        if (detections or upper_lim) and study == 'graham':
            det_hist = pd.read_csv(det_hist_file, index_col=0)
        if upper_lim and study == 'graham':
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
    if not detections:
        det_hist = []
    plot(x_edges, y_edges, hist, show=show_plot, output_file=plot_file, cmax=cmax,
            cmin=cmin, upper_lim=upper_lim, det_hist=det_hist)


def plot(x_edges, y_edges, hist, show=True, output_file='recovery.pdf', 
        cmax=None, cmin=None, upper_lim=False, det_hist=[]):
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
        det_hist: histogram of detections
    """

    # Flip y-axis
    hist.sort_index(ascending=True, inplace=True)

    # Define colormap
    n_colors = 9 # number of distinct colors, including below/above bounds
    cmap_name = 'plasma'
    if upper_lim:
        cmap_name += '_r' # reversed
    cmap = plt.cm.get_cmap(cmap_name)
    cmap.set_under('k') # set color for values under minimum
    cmap.set_over('k') # set color for values over maximum

    # colorbar limits
    if upper_lim:
        # set limit to highest value not at 100%
        hist_max = int(np.nanmax(hist[hist < 100].to_numpy())) + 1
        hist_min = np.min(hist.to_numpy())
        print(hist_min)
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
    fig, ax = plt.subplots(tight_layout=True)
    pcm = ax.pcolormesh(x_edges, y_edges, hist, cmap=cmap, norm=norm,
            edgecolor='k', linewidth=0.3, antialiased=True)

    # Outline detections
    if len(det_hist) > 0:
        ls = ['--', ':', '-.']
        lw = [3, 4, 5]

        for n in range(int(det_hist.max().max())):
            # Mask detections
            det_mask = det_hist[det_hist >= n+1]
            det_mask[pd.notna(det_mask)] = -1

            # Outline area
            det_mask.reset_index(inplace=True, drop=True)
            x_lower = []
            x_upper = []
            y_lower = []
            y_upper = []
            for i, x_edge in enumerate(x_edges):
                col = det_mask[x_edge]
                # Continuous range of values above limit
                cont = col[pd.notna(col)]
                if len(cont) == 0:
                    x_upper.insert(0, x_lower[0])
                    y_upper.insert(0, y_lower[0])
                    break

                x_lower += [x_edge, x_edges[i+1]]
                x_upper += [x_edge, x_edges[i+1]]

                y_lower += [y_edges[cont.index[0]]] * 2
                y_upper += [y_edges[cont.index[-1]+1]] * 2

            x_upper.reverse()
            x = x_lower + x_upper[0:1] # don't outline top

            y_upper.reverse()
            y = y_lower + y_upper[:1]

            line, = ax.plot(x, y, color='k', linestyle=ls[n], linewidth=lw[n],
                    label='%s det.' % (n+1))
            line.set_clip_on(False) # allow line to bleed over spines

        # Legend for detections
        plt.legend(loc='upper left', ncol=2, handletextpad=0.8, handlelength=2.,
                borderpad=0.4, bbox_to_anchor=(0., 1.12), borderaxespad=0.)

    # Format axes
    ax.set_yscale('log')
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.tick_params(which='both', direction='out', top=False, right=False)
    ax.set_xlabel('$t_{start}$ [days]')
    # ax.set_ylabel('Scale factor')
    ax.set_ylabel('$S$', rotation='horizontal')

    # Adjust colorbar: add extension below lower limit
    if upper_lim:
        cbar_label = 'Upper 90% confidence limit [%]'
        bounds = list(cmap_bounds) + [100]
        extend = 'max'
    else:
        cbar_label = 'Excluded SNe Ia'
        bounds = [0] + list(cmap_bounds)
        extend = 'min'
    cbar = fig.colorbar(pcm, spacing='uniform', extend=extend, 
            boundaries=bounds, ticks=cmap_bounds, extendfrac='auto', 
            fraction=0.1, aspect=16, pad=0.04)
    cbar.ax.tick_params(which='both', right=False)
    # cbar.ax.set_ylabel(cbar_label)
    cbar.ax.set_ylabel(cbar_label, rotation='horizontal', 
                ha='right', va='top', y=1.12, labelpad=0)

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
    """Import recovery save data and return histogram.
    Inputs:
        x_edges: list of x-axis bin edges
        y_edges: list of y-axis bin edges
    """

    rd = RecoveryData(fname)
    hist = rd.hist(x_edges, y_edges)
    return hist


class RecoveryHistogram:
    def __init__(self, save_dir, iterations=ITERATIONS):
        """Import recovery save data for all targets sampled.
        Input:
            save_dir: directory of injection-recovery save files
            iterations: number of iterations of injection per SN Ia
        """

        self.save_dir = Path(save_dir)
        # Find save files
        save_files = list(Path(save_dir).glob('*-%s.csv' % iterations))


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
            help='Include recovery of G19 detections as hatch overlay')
    parser.add_argument('-u', '--upperlim', action='store_true', 
            help='plot upper limit of binomial conf intervals instead of excluded SNe')
    parser.add_argument('--smax', default=SCALE_MAX, type=int, help='y-axis upper limit')
    args = parser.parse_args()

    main(args.iterations, t_min=0, t_max=args.tmax, overwrite=args.overwrite, model=args.model, 
            study=args.study.lower(), sigma=args.sigma, cmax=args.cmax, 
            cmin=args.cmin, scale_max=args.smax, bin_width=args.twidth, 
            detections=args.detections, upper_lim=args.upperlim)