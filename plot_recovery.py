from functools import reduce
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter, MultipleLocator
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, SymLogNorm
from matplotlib import cm
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import dill
import copy
from utils import *
from CSMmodel import CSMmodel

SIGMA = 3
CONF = 0.9 # confidence level of binomial intervals
OUTPUT_DIR = Path('recovery_plots/')
SCALE_BINS = 20 # number of scale factor bins
ITERATIONS = 10000 # default number of injection iterations

def main(model, study, t_min=TSTART_MIN, t_max=TSTART_MAX, scale_min=SCALE_MIN,
        scale_max=SCALE_MAX, bin_width=TSTART_BIN_WIDTH, y_bins=SCALE_BINS,
        show=True, cmax=None, overwrite=False, detections=False,
        reference=False, upper_lim=False, cmin=None, quad=False, extension='.pdf'):

    # Import reference SN labels
    if reference:
        sn_label = pd.read_csv(Path('ref/sn_scale_factors.csv'), index_col=0)
    else:
        sn_label = []
    
    # Bin edges
    x_edges = np.arange(t_min, t_max+bin_width, bin_width)
    y_edges = np.logspace(np.log10(scale_min), np.log10(scale_max), num=y_bins+1)

    # Quad plot: four main plots in one figure
    if quad:
        plot_quad(x_edges, y_edges, sn_label=sn_label, detections=detections,
                upper_lim=upper_lim, extension=extension, show=show, 
                overwrite=overwrite, cmin=cmin, cmax=cmax, bin_width=bin_width)

    # Single plot
    else:
        # Plot file name
        fname = 'recovery_%s_%s' % (study, model)
        if upper_lim:
            fname += '_upperlim'
        elif detections:
            fname += '_det'

        print('Plotting recovery histogram...')
        fig, ax = plt.subplots()
    
        hist, det_hist = get_histograms(study, model, x_edges, y_edges, 
                detections=detections, upper_lim=upper_lim, 
                conf=CONF, overwrite=overwrite)

        # Define colormap and index and plot
        cmap, bounds = get_cmap(hist, upper_lim=upper_lim, cmin=cmin, cmax=cmax)
        norm = BoundaryNorm(bounds, cmap.N) # colormap index
        pcm = plot_hist(ax, x_edges, y_edges, hist, cmap, norm, bin_width, sn_label)

        # Outline detections
        if len(det_hist) > 0:
            plot_detections(ax, x_edges, y_edges, det_hist, label=True, c='k')
            # Legend for detections (actually upper left)
            plt.legend(loc='lower left', ncol=2, handletextpad=0.8, handlelength=2.,
                    borderpad=0.3, bbox_to_anchor=(0., 1.05), borderaxespad=0.,
                    fontsize=9)

        # Adjust colorbar: add extension below lower limit
        if upper_lim:
            cbar_label = 'Upper 90% confidence limit [%]'
            bounds = list(bounds) + [100]
            extend = 'max'
        else:
            cbar_label = 'Excluded SNe Ia'
            bounds = [0] + list(bounds)
            extend = 'min'
        # Add vertical colorbar to the side
        if reference:
            cbar_pad = 0.16
        else:
            cbar_pad = 0.04
        cbar = fig.colorbar(pcm, spacing='uniform', extend=extend, 
                boundaries=bounds, ticks=bounds, extendfrac='auto', 
                fraction=0.1, aspect=16, pad=cbar_pad)
        cbar.ax.tick_params(which='both', right=False)
        # horizontal label above plot
        cbar.ax.set_ylabel(cbar_label, rotation='horizontal', 
                    ha='right', va='top', y=1.16, labelpad=0)

        plt.tight_layout(pad=0.3)

        # Save hist
        plot_file = OUTPUT_DIR / Path(fname + extension)
        plt.savefig(plot_file, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()


def plot_quad(x_edges, y_edges, sn_label=[], detections=False, upper_lim=False, 
        extension='.pdf', show=True, overwrite=False, cmin=None, cmax=None,
        bin_width=TSTART_BIN_WIDTH):
    """Plot four injection-recovery plots with shared axes and color bar.
    Inputs:
        x_edges, y_edges: bin edges, np arrays
        sn_label: DataFrame of reference SN labels
        detections: if True, outline detections on HST plots
        upper_lim: if True, plot upper 90% conf interval instead of excluded sne
        extension: output file name extension
        show: if True, show plot before saving
        overwrite: if True, overwrite hist csv files
        cmin: optional manual minimum value of colorbar (must be >0)
        cmax: optional manual maximum value of colorbar
        bin_width: width of tstart bins
    """

    # Plot file name
    fname = 'recovery_quad'
    if upper_lim:
        fname += '_upperlim'

    if len(sn_label) > 0:
        wspace = 0.17 # widen spacing between plots
    else:
        wspace = 0.08

    # Define subplots
    print('Plotting recovery histograms...')
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 6.5))

    study = ['galex', 'galex', 'graham', 'graham']
    model = ['Chev94', 'flat', 'Chev94', 'flat']
    label = ['$\it{GALEX}$ observations, line-emission model',
             '$\it{GALEX}$ observations, flat spectrum model',
             '$\it{HST}$ observations, line-emission model',
             '$\it{HST}$ observations, flat spectrum model']

    # Determine max and min of all four histograms; widest range -> cmap
    hist_max = 0
    hist_min = 1e6 # arbitrary large number
    for i, ax in enumerate(axs.flat):
        hist, det_hist = get_histograms(study[i], model[i], x_edges, y_edges, 
                detections=detections, upper_lim=upper_lim, 
                conf=CONF, overwrite=overwrite)
        if np.nanmax(hist) > hist_max and np.nanmin(hist) < hist_min:
            cmap, bounds = get_cmap(hist, upper_lim=upper_lim, 
                    cmin=cmin, cmax=cmax)
            hist_max = np.nanmax(hist)
            hist_min = np.nanmin(hist)

    # Add subplots
    add_label = True
    for i, ax in enumerate(axs.flat):
        # Import histograms again
        hist, det_hist = get_histograms(study[i], model[i], x_edges, y_edges, 
                detections=detections, upper_lim=upper_lim, 
                conf=CONF, overwrite=False, verb=False)
        norm = BoundaryNorm(bounds, cmap.N) # colormap index
        # Right-hand plots: y-ticks on right side and no reference labels
        if i % 2 == 1:
            pcm = plot_hist(ax, x_edges, y_edges, hist, cmap, norm, bin_width,
                    sn_label=sn_label, markers_only=True)
            ax.tick_params(which='major', left=False, right=True)
        # Left-hand plots: y-ticks on left side, reference labels on right
        else:
            pcm = plot_hist(ax, x_edges, y_edges, hist, cmap, norm, bin_width,
                    sn_label=sn_label)
        # Outline detections
        if len(det_hist) > 0 and not upper_lim:
            plot_detections(ax, x_edges, y_edges, det_hist,
                    label=add_label, c='k')
            add_label = False # only include one set of legend handles
        # Format
        ax.set_title(label[i], weight='normal')
        ax.label_outer() # hide labels for inner plots

    # Legend for detections
    if len(det_hist) > 0 and not upper_lim:
        fig.legend(loc='upper left', ncol=2, handletextpad=0.8, handlelength=2.,
                borderpad=0.5)

    # Adjust colorbar bounds: add extension below lower limit
    if upper_lim:
        cbar_label = 'Upper 90% confidence limit [%]'
        bounds = list(bounds) + [100]
        extend = 'max'
    else:
        cbar_label = 'Excluded SNe Ia'
        bounds = [0] + list(bounds)
        extend = None
    # Adjust subplots to make room for colorbar axis
    plt.subplots_adjust(bottom=0.08, top=0.87, left=0.08, right=0.97, 
            wspace=wspace, hspace=0.15)
    # Define cbar axis
    if len(det_hist) > 0 and not upper_lim:
        cax = plt.axes([0.3, 0.92, 0.5, 0.03]) # left, bottom, width, height
    else:
        cax = plt.axes([0.08, 0.92, 0.59, 0.03]) # left, bottom, width, height
    # Add horizontal colorbar above subplots
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, 
            ax=axs, orientation='horizontal', fraction=0.1,
            spacing='uniform', extend=extend, boundaries=bounds, 
            ticks=bounds, extendfrac='auto')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(which='both', bottom=False, top=False, pad=2)
    # horizontal label next to cbar
    cbar.set_label(cbar_label, rotation='horizontal', va='bottom', 
            ha='left', x=1.03, labelpad=0)

    # Save hist
    plot_file = OUTPUT_DIR / Path(fname + extension)
    plt.savefig(plot_file, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def get_histograms(study, model, x_edges, y_edges, sigma=SIGMA, detections=False, 
        upper_lim=False, conf=0.9, overwrite=False, iterations=ITERATIONS, verb=True):
    """Import histogram files if they exist, or generate if they don't.
    Inputs:
        study: 'galex' or 'graham'
        model: 'Chev94' or 'flat'
        x_edges: x-axis bin edges
        y_edges: y-axis bin edges
        sigma: recovery significance threshold
        detections: whether to outline detections
        upper_lim: if True, plot upper limit of binomial conf interval
        conf: confidence level of binomial interval
        overwrite: if True, re-bin and overwrite histograms
        iterations: number of iterations used in injection runs
        verb: if True, print status updates as files are generated/imported
    Outputs:
        hist: 2D histogram of excluded SNe Ia or upper conf interval
        det_hist: list of 2D histograms of individual detections if needed, else []
    """
        
    fname = 'recovery_%s_%s' % (study, model)
    hist_file = OUTPUT_DIR / Path(fname + '.csv')
    det_hist_file = OUTPUT_DIR / Path(fname + '_det.csv')

    # Import list of individual detections
    if verb:
        print('Importing individual %s %s detections...' % (model, study))
    if (detections or upper_lim) and study == 'graham':
        det_save_dir = run_dir(study + '_det', model, sigma)
        det_save_files = list(Path(det_save_dir).glob('*-%s.csv' % iterations))
        det_hist = {fname2sn(f)[0]: sum_hist([f], x_edges, y_edges, save=False) for f in det_save_files}

    # Generate histogram(s)
    if overwrite or not hist_file.is_file():
        if verb:
            print('Generating %s %s histogram(s)...' % (study, model))

        # List of files in save dir
        save_dir = run_dir(study, model, sigma)
        save_files = list(Path(save_dir).glob('*-%s.csv' % iterations))

        hist = sum_hist(save_files, x_edges, y_edges, output_file=hist_file)

        # Also sum detections, if applicable
        if (detections or upper_lim) and study == 'graham':
            det_sum = sum_hist(det_save_files, x_edges, y_edges, 
                    output_file=det_hist_file)
        else:
            det_sum = []
            det_hist = []

        # Include detections in nondetections for binomial upper limits
        if upper_lim and study == 'graham':
            hist = hist + det_sum

    # Import histogram(s)
    else:
        hist = pd.read_csv(hist_file, index_col=0)
        hist.columns = hist.columns.astype(int)
        if verb:
            print('Imported %s' % hist_file)

        # Also import detections, if applicable
        if (detections or upper_lim) and study == 'graham':
            det_sum = pd.read_csv(det_hist_file, index_col=0)
            det_sum.columns = det_sum.columns.astype(int)
            if verb:
                print('Imported %s' % det_hist_file)
        else:
            det_sum = []
            det_hist = []

        # Include detections in nondetections for binomial upper limits
        if upper_lim and study == 'graham':
            hist = hist + det_sum

    # Calculate 90% binom conf interval upper limits
    if upper_lim:
        # Zero detections for GALEX
        if study == 'galex':
            zeros = np.zeros(hist.shape)
            det_sum = pd.DataFrame(zeros, index=hist.index, columns=hist.columns)

        # Binomial confidence interval
        hist = 100 * bci_nan(det_sum, hist, conf=conf)[1]
    
    if not detections:
        det_hist = []

    return hist, det_hist


def get_cmap(hist, upper_lim=False, cmin=None, cmax=None, n_colors=9, 
        name='plasma', under='k', over='k'):
    """Return colormap and bounds for 2D histogram.
    Inputs:
        hist: 2D histogram
        cmin: optional manual minimum value of colorbar (must be >0)
        cmax: optional manual maximum value of colorbar
        n_colors: number of distinct colors, including above/below bounds
        name: pre-defined colormap name
    Outputs:
        cmap: matplotlib colormap
        bounds: boundaries between discrete colors
    """

    # Get colormap
    if upper_lim:
        name += '_r' # reversed
    cmap = copy.copy(cm.get_cmap(name))
    cmap.set_under(under) # set color for values under minimum
    cmap.set_over(over) # set color for values over maximum

    # colorbar limits
    if upper_lim:
        # set limit to highest value not at 100%
        hist_max = int(np.nanmax(hist[hist < 100].to_numpy())) + 1
        hist_min = np.min(hist.to_numpy())
    else:
        hist_max = int(np.max(hist.to_numpy())) + 1
        hist_min = 1
    if not cmax:
        cmax = hist_max
    if not cmin:
        cmin = hist_min

    # colormap bounds: (rounded) logarithmic scale
    bounds = np.logspace(np.log10(cmin), np.log10(cmax), num=n_colors)
    bounds[0] = int(bounds[0]) # first bound must round down
    bounds = np.round(bounds)

    return cmap, bounds


def plot_hist(ax, x_edges, y_edges, hist, cmap, norm, bin_width=100, sn_label=[],
        markers_only=False):
    """Add 2D histogram plot to axis.
    Inputs:
        ax: matplotlib axis
        x_edges: x-axis bin edges
        y_edges: y-axis bin edges
        hist: Pandas 2D histogram
        cmap: colormap
        norm: colormap index
        bin_width: x-axis bin width in days
        sn_label: DataFrame of SNe to label as reference on right side of plot
        markers_only: if True, only add SN label markers to left side
    Output:
        pcm: pcolormesh object
    """

    # Flip y-axis and plot hist
    hist.sort_index(ascending=True, inplace=True)
    pcm = ax.pcolormesh(x_edges, y_edges, hist, cmap=cmap, norm=norm,
            edgecolor='k', linewidth=0.3, antialiased=True)

    # Format axes
    ax.set_yscale('log')
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    ax.yaxis.set_major_formatter(formatter)
    # ax.yaxis.set_minor_formatter(formatter)
    # ax.xaxis.set_minor_locator(MultipleLocator(bin_width))
    ax.xaxis.set_major_locator(MultipleLocator(5 * bin_width))
    # ax.xaxis.set_minor_locator(MultipleLocator(5 * bin_width))
    ax.tick_params(which='major', direction='out', top=False, right=False, width=1)
    # ax.tick_params(which='minor', axis='both', left=False)
    ax.minorticks_off()
    ax.set_xlabel('$t_\mathrm{start}$ [days]')
    ax.set_ylabel('$S$', rotation='horizontal')

    # Add reference SN labels
    if len(sn_label) > 0:
        for sn, info in sn_label.iterrows():
            if markers_only:
                ax.text(x_edges[0], info.scale, '-', c='r', va='center', ha='right')
            else:
                ax.text(x_edges[-1], info.scale, '- '+info.label, c='r', va='center')

    return pcm


def plot_detections(ax, x_edges, y_edges, det_hist, label=True, c='k'):
    """Add outline of detections to histogram.
    Inputs:
        ax: matplotlib axis
        x_edges: x-axis bin edges
        y_edges: y-axis bin edges
        det_hist: dict of 2D histograms of individual detections, all same shape
        label: if true, add handles and labels to legend
    """

    # Outline styles
    lw = [2, 2.5]
    dashes = [(4, 1), (1, 1)]

    # for n in range(int(det_hist.max().max())):
    for n, sn_name in enumerate(det_hist):
        # Mask detections
        hist = det_hist[sn_name]
        # det_mask = det_hist[det_hist >= n+1]
        det_mask = hist[hist > 0]
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

        line, = ax.plot(x, y, color=c, linestyle='--', linewidth=lw[n], 
                dashes=dashes[n])
        line.set_clip_on(False) # allow line to bleed over spines
        if label:
            # line.set_label('%s det.' % (n+1))
            line.set_label(sn_name.split('-')[-1].split(' 20')[-1])


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
    parser.add_argument('--overwrite', '-o', action='store_true', 
            help='Overwrite histograms (takes longer)')
    parser.add_argument('--model', '-m', type=str, default='Chev94', 
            help='CSM spectral model: "flat" or "Chev94"')
    parser.add_argument('--study', '-s', type=str, default='galex', 
            help='Study from which to pull data: "galex" or "graham"')
    parser.add_argument('--cmax', type=float, help='Max colorbar value')
    parser.add_argument('--cmin', type=float, help='Minimum colorbar value')
    parser.add_argument('--tmax', default=TSTART_MAX, type=int, 
            help='t_start upper limit')
    parser.add_argument('--twidth', default=TSTART_BIN_WIDTH, type=int, 
            help='t_start bin width')
    parser.add_argument('-d', '--detections', action='store_true', 
            help='Include recovery of HST detections as hatch overlay')
    parser.add_argument('-u', '--upperlim', action='store_true', 
            help='Plot upper limit of binomial conf intervals instead of excluded SNe')
    parser.add_argument('--smax', default=SCALE_MAX, type=int, 
            help='Scale factor upper limit')
    parser.add_argument('--sbins', default=SCALE_BINS, type=int, 
            help='Number of scale factor bins')
    parser.add_argument('--quad', '-q', action='store_true', 
            help='Grid of four recovery plots')
    parser.add_argument('--reference', '-r', action='store_true', 
            help='Label reference SNe along vertical axis')
    parser.add_argument('--extension', '-e', type=str, default='.pdf', 
            help='Plot file extension')
    args = parser.parse_args()

    main(args.model, args.study.lower(), t_min=0, t_max=args.tmax, 
            scale_max=args.smax, overwrite=args.overwrite,
            cmin=args.cmin, cmax=args.cmax, bin_width=args.twidth, 
            detections=args.detections, upper_lim=args.upperlim, quad=args.quad,
            extension=args.extension, reference=args.reference)
