from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from astropy.stats import binom_conf_interval
from utils import *
from plot_recovery import sum_hist

# Defaults
CONF = 0.9 # binomial confidence level
MODEL = 'Chev94' # default spectral model
SCALE = [0.9, 1.1] # default model scale
SIGMA = 3 # default confidence for excluded SNe
TSTART_MAX = 2000
YMAX = None
PAD = 0.1 # percent, y-axis padding on log scale

# Plot settings
ARROW_LENGTH = 2
COLORS = {  'GALEX': '#47a',
            'HST': '#e67',
            'This study': 'k',
            'ASAS-SN': 'k',
            'ZTF': 'k'
}
MARKERS = { 'ASAS-SN': 'o', 
            'ZTF': 's'
}
ALPHAS = {  'GALEX': 0.1,
            'HST': 0.1,
            'This study': 0.3
}
HATCHES = { 'GALEX': '|',
            'HST': '-',
            'This study': 'x'
}
STYLE = {   'GALEX': '--',
            'HST': ':',
            'This study': '-'}


def main(bin_width=TSTART_BIN_WIDTH, scale=SCALE, iterations=10000, y_max=YMAX,
        model=MODEL, sigma=SIGMA, t_min=TSTART_MIN, t_max=TSTART_MAX, log=False,
        pad=False):
    
    # Bin edges
    x_edges = np.arange(t_min, t_max+bin_width, bin_width)
    y_edges = np.array(scale)
    nbins = len(x_edges)-1

    # Import sum histograms for GALEX and HST non-detections
    galex_hist = import_recovery('galex', model, sigma, x_edges, y_edges)
    graham_hist = import_recovery('graham', model, sigma, x_edges, y_edges)
    # and HST detections
    graham_det_hist = import_recovery('graham', model, sigma, x_edges, y_edges,
            detections=True)

    # DataFrame for number of trials per tstart bin and data source
    tstart_bins = pd.Series(x_edges[:-1])
    trials = pd.DataFrame([], index=tstart_bins)
    trials['GALEX'] = galex_hist.T
    trials['HST'] = (graham_hist + graham_det_hist).T
    trials['This study'] = trials['GALEX'] + trials['HST']

    # DataFrame for number of detections per tstart bin and data source
    detections = pd.DataFrame([], index=tstart_bins)
    detections['GALEX'] = np.zeros((nbins, 1))
    detections['HST'] = graham_det_hist.T
    detections['This study'] = detections['GALEX'] + detections['HST']

    # Import ASAS-SN and ZTF SNe
    asassn_det, asassn_all = count_asassn_sne()
    ztf_det, ztf_all = count_ztf_sne()

    # Calculate binomial confidence intervals
    bci_lower, bci_upper = bci_nan(detections, trials, conf=CONF)
    # Convert to percentages
    bci_lower *= 100
    bci_upper *= 100

    # Calculate binomial confidence intervals for external data
    print('\nExternal measures of f_CSM:')
    asassn_bci = 100 * binom_conf_interval(asassn_det, asassn_all, 
            confidence_level=CONF, interval='jeffreys')
    print('ASAS-SN')
    print(asassn_bci)
    ztf_bci = 100 * binom_conf_interval(ztf_det, ztf_all, confidence_level=CONF, 
            interval='jeffreys')
    print('ZTF')
    print(ztf_bci)
    external_bci = pd.DataFrame([asassn_bci, ztf_bci], index=['ASAS-SN', 'ZTF'],
            columns=['bci_lower', 'bci_upper'])

    scale_mean = int(np.mean(y_edges))
    plot_single(bci_lower, bci_upper, external_bci, show=True, y_max=y_max, log=log, pad=pad,
            output_file=Path('out/rates_%s_scale%s.pdf' % (model, scale_mean))) 


def plot_single(lower, upper, external, output_file='out/rates.pdf', show=True, 
        y_max=YMAX, log=False, pad=False):
    """Plot binomial confidence limits for CSM interaction rate.
    Inputs:
        lower: DataFrame of lower 90% CI for GALEX, HST data
        upper: DataFrame of upper 90% CI for GALEX, HST data
        external: DataFrame of BCI for ASAS-SN and ZTF data
        output_file: file name for plot
        show: if True, display plot before saving
        y_max: maximum y-axis value (if log==False)
        log: if True, plot y-axis on log scale from 0.1 - 100
        pad: if True, pad all data by 0.1% so lower bound is displayed on log
    """

    fig, ax = plt.subplots(figsize=(3.25, 2.25))

    x = plot_bounds(ax, lower, upper, y_max, log, pad)
    
    # Axes labels
    ax.set_xlabel('$t_{start}$ [days]')
    if log and pad:
        ylabel = '$f_{CSM} + ' + str(PAD) + '\%$'
    else:
        ylabel = '$f_{CSM}$ [%]'
    ax.set_ylabel(ylabel)

    # Axes limits
    if log:
        y_min = 0.05
        y_max = 150
        ax.set_yscale('log')
    else:
        y_min = None
    ax.set_ylim((y_min, y_max))
    ylim = ax.get_ylim()
        
    # x-axis ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(100))
    ax.xaxis.set_major_locator(plt.MultipleLocator(500))

    # y-axis ticks
    if log:
        formatter = ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y))
        ax.yaxis.set_major_formatter(formatter)

    # Plot external study estimates as bars
    x_frac = 1/30 # fraction of x-axis to plot ranges left of x=0
    x_bar = -int(x[-1] * x_frac) # x-val of error bar
    plot_external(ax, external, x_bar, log=log, pad=pad)

    # Legend (actually upper right)
    plt.legend(loc='lower right', ncol=3, bbox_to_anchor=(1.05, 0.95), 
            handletextpad=0.5, handlelength=1., borderpad=0.3, fontsize=9)

    plt.tight_layout(pad=0.3)

    # Save & exit
    plt.savefig(output_file, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_bounds(ax, lower, upper, y_max=YMAX, log=False, pad=False):
    """Plot lower & upper bounds on fCSM for GALEX and HST data.
    Inputs:
        ax: matplotlib axis
        lower: DataFrame of lower 90% CI for GALEX, HST data
        upper: DataFrame of upper 90% CI for GALEX, HST data
        y_max: maximum y-axis value (if log==False)
        log: if True, plot y-axis on log scale from 0.1 - 100
        pad: if True, pad all data by 0.1% so lower bound is displayed on log
    Output:
        x: x values of fCSM bounds
    """

    # x-values of fCSM bounds
    x = lower.index.to_numpy()
    x_end = x[-1] + (x[1] - x[0])
    x = np.append(x, [x_end])

    for col in lower.columns:
        # Find midpoint and errors for plotting
        y1 = lower[col].to_numpy()
        y2 = upper[col].to_numpy()
        # Add endpoints
        y1 = np.append(y1, y1[-1])
        y2 = np.append(y2, y2[-1])

        if log and pad:
            y1 += PAD
            y2 += PAD
        
        color = COLORS[col]
        alpha = ALPHAS[col]
        hatch = HATCHES[col]
        ls = STYLE[col]
        lw = 1

        # Italicize GALEX
        if col == 'GALEX' or col == 'HST':
            label = '$\it{%s}$' % col
        else:
            label = col

        if col == 'This study':
            ax.fill_between(x, y1, y2, facecolor=color, edgecolor='None', 
                    alpha=alpha, step='post', ls=ls, lw=lw, label=label, zorder=1)
            # In-plot label
            # ax.text(20, (y2[0]-y1[0])/2, col, ha='left', va='center', 
            #         size=TEXT_SIZE)
            # Add arrows if line exits plot
            if y_max is not None:
                x_mid = (x[1:] + x[:-1]) / 2
                above_lim = x_mid[y2[:-1] > y_max]
                for val in above_lim:
                    ax.arrow(val, y_max-ARROW_LENGTH, 0, ARROW_LENGTH, color=color,
                            head_width=30, zorder=10, length_includes_head=True, 
                            head_starts_at_zero=False, head_length=ARROW_LENGTH,
                    )
        
        else:
            # Upper bound
            line2, = ax.step(x, y2, c=color, ls=ls, lw=lw, where='post', label=label)
            # Lower bound
            line1, = ax.step(x, y1, c=color, ls=ls, lw=lw, where='post')
            if (log and pad) or not log:
                line1.set_clip_on(False) # stop bottom line from clipping on axis
            # In-plot label
            # ax.text(0, y2[0], col, ha='left', va='bottom', size=TEXT_SIZE, 
            #         color=color)
            # Add arrow if line exits plot
            if (y_max is not None) and (np.max(y2) > y_max):
                exit_x = x[y2 > y_max][0]
                ax.arrow(exit_x, y_max-ARROW_LENGTH, 0, ARROW_LENGTH, color=color, 
                        head_width=30, zorder=10, length_includes_head=True, 
                        head_starts_at_zero=False, head_length=ARROW_LENGTH,
                )

    # Grid lines
    ax.grid(b=True, which='minor', axis='x', color='w', lw=0.5)
    ax.grid(b=True, which='major', axis='x', color='w', lw=1)
    if log:
        ax.grid(b=True, which='major', axis='y', color='w', lw=0.5)
    else:
        ax.grid(b=True, which='both', axis='y', color='w', lw=0.5)

    return x


def plot_external(ax, external, x_bar, y_text=-0.2, log=False, pad=False):
    """Plot estimates from ASAS-SN and ZTF studies as bars to left of x-axis.
    Inputs:
        ax: matplotlib axis
        external: DataFrame of external fCSM bounds
        x_bar: x-value of right-most error bar
        y_text: y-value of annotation
        log: if True, plot y-axis on log scale from 0.1 - 100
        pad: if True, pad all data by 0.1% so lower bound is displayed on log
    """

    for i, study in enumerate(external.index):
        # Calculate error bar
        row = external.loc[study]
        midpoint = row.mean()
        err = row['bci_upper'] - midpoint
        if log and pad:
            midpoint += PAD
        # Plot error bar
        ax.errorbar(x_bar, midpoint, yerr=err, marker=MARKERS[study], 
                c=COLORS[study], mec=COLORS[study], mfc='w', ms=0, 
                linestyle='none', elinewidth=3, mew=2, capsize=0)#, label=study)
        # Annotate label under bar
        ax.annotate(study, xy=(x_bar, midpoint - err), xytext=(x_bar, y_text), 
                textcoords=('data', 'axes fraction'),
                ha='right', va='top', size=9, #size=TEXT_SIZE-2,
                arrowprops={'color': 'k', 'shrink': 0.1, 'width': 0.2,
                        'headwidth': 0.2, 'headlength': 0.01})
        # Adjust label coordinates
        x_bar += x_bar
        y_text += 0.1


def import_recovery(study, model, sigma, x_edges, y_edges, detections=False,
        iterations=ITERATIONS):
    """Import recovery save files and sum histograms with given bounds."""

    # File names
    save_dir = run_dir(study, model, sigma, detections)
    save_files = list(Path(save_dir).glob('*-%s.csv' % iterations))
    # Generate summed histogram
    print('Importing and summing %s saves from %s' % (study, save_dir))
    hist = sum_hist(save_files, x_edges, y_edges, save=False)
    count = np.nan_to_num(hist.iloc[0].to_numpy())

    return count


def count_asassn_sne():
    """Count SNe Ia-CSM in four ASAS-SN data sets.
    Returns:
        number of Ia-CSM, total number of Ia
    """

    data = []
    col_names = ['sn_name', 'iau_name', 'disc_date', 'z', 'type', 'disc_age']
    files = ['ASASSN_I.txt', 'ASASSN_II.txt', 'ASASSN_III.txt', 'ASASSN_IV.txt']
    skip_rows = [31, 29, 28, 30]
    for file, skip_row in zip(files, skip_rows):
        if file == 'ASASSN_IV.txt':
            cols = [0, 1, 2, 5, 10, 11]
        else:
            cols = [0, 1, 2, 5, 9, 10]
        file_path = Path('ref/%s' % file)
        df = read_pub_data(file_path, cols, col_names, skip_row)
        df['disc_age'] = df['disc_age'].astype(float)
        data.append(df)

    data = pd.concat(data)

    all_Ia = data[data['type'].str.contains('Ia')]
    Ia_CSM = data[data['type'] == 'Ia+CSM']

    return len(Ia_CSM.index), len(all_Ia.index)


def count_ztf_sne():
    """Count SNe Ia-CSM in ZTF I data set.
    Returns:
        number of Ia-CSM, total number of Ia
    """

    cols = [0, 3, 7, 8]
    col_names = ['sn_name', 'iau_name', 'subtype', 'z']
    file_path = Path('ref/ZTF_I.txt')
    data = read_pub_data(file_path, cols, col_names, 53)

    # Count number of Ia-CSM
    Ia_CSM = data[data['subtype'] == 'Ia-CSM']

    return len(Ia_CSM.index), len(data.index)


def read_pub_data(path, cols, col_names, skip_rows):
    """Import astronomical data tables for, e.g., ZTF or ASAS-SN.
    Inputs:
        path: file path
        cols: list of column indices to import
        col_names: list of column names, same length as cols
        skip_rows: number of rows to skip at beginning of file
    """

    df = pd.read_csv(path, sep='\s+', index_col=0, na_values='---', 
            skiprows=skip_rows, usecols=cols, names=col_names)
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='Chev94', 
            help='CSM spectrum model')
    parser.add_argument('--scale', '-S', nargs=2, type=float, default=SCALE,
            help='lower and upper scale factor bounds')
    parser.add_argument('--sigma', type=int, nargs='+', default=[SIGMA], 
            help='Detection confidence level (multiple for tiered detections)')
    parser.add_argument('--tmax', type=int, default=TSTART_MAX, 
            help='Maximum CSM interaction start time')
    parser.add_argument('--twidth', type=int, default=TSTART_BIN_WIDTH, 
            help='t_start bin width')
    parser.add_argument('--ymax', type=float, default=YMAX, 
            help='y-axis upper limit')
    parser.add_argument('--log', action='store_true', help='y-axis log scale')
    parser.add_argument('--pad', action='store_true', help='pad y-axis by 0.1% on log scale')
    args = parser.parse_args()

    main(model=args.model, scale=args.scale, bin_width=args.twidth, pad=args.pad,
            sigma=args.sigma, t_max=args.tmax, y_max=args.ymax, log=args.log)