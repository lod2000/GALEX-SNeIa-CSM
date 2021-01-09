from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from astropy.stats import binom_conf_interval
from utils import *
from plot_recovery import sum_hist

# TSTART_BINS = [0, 20, 100, 500, 1000, 2500]
# Defaults
CONF = 0.9 # binomial confidence level
MODEL = 'Chev94' # default spectral model
SCALE = [0.9, 1.1] # default model scale
# DS = 0.1 # range above and below scale to include
SIGMA = 3 # default confidence for excluded SNe
TSTART_MAX = 2000
YMAX = None

# Plot settings
TEXT_SIZE = 16
ARROW_LENGTH = 2
COLORS = {  'GALEX': '#47a',
            'HST': '#e67',
            'This study': 'k',
            # 'ASAS-SN': '#cb4',
            # 'ZTF': '#283'
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
        model=MODEL, sigma=SIGMA, t_min=TSTART_MIN, t_max=TSTART_MAX):
    
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
    asassn_bci = 100 * binom_conf_interval(asassn_det, asassn_all, 
            confidence_level=CONF, interval='jeffreys')
    ztf_bci = 100 * binom_conf_interval(ztf_det, ztf_all, confidence_level=CONF, 
            interval='jeffreys')
    external_bci = pd.DataFrame([asassn_bci, ztf_bci], index=['ASAS-SN', 'ZTF'],
            columns=['bci_lower', 'bci_upper'])

    # table(detections, trials, bci_upper, tstart_bins=TSTART_BINS, 
    #         output_file=Path('out/rates_%s.tex' % model))

    scale_mean = int(np.mean(y_edges))
    plot(bci_lower, bci_upper, external_bci, show=True, y_max=y_max,
            output_file=Path('out/rates_%s_scale%s.pdf' % (model, scale_mean))) 


def plot(lower, upper, external, output_file='out/rates.pdf', show=True, y_max=YMAX):
    """Plot binomial confidence limits for CSM interaction rate.
    Inputs:
        lower: DataFrame of lower 90% CI for GALEX, HST data
        upper: DataFrame of upper 90% CI for GALEX, HST data
        external: DataFrame of BCI for ASAS-SN and ZTF data
    """

    fig, ax = plt.subplots(tight_layout=True)

    x = lower.index.to_numpy()
    x_end = x[-1] + (x[1] - x[0])
    x = np.append(x, [x_end])

    # Plot confidence intervals
    for col in lower.columns:
        # Find midpoint and errors for plotting
        y1 = lower[col].to_numpy()
        y2 = upper[col].to_numpy()
        # Add endpoints
        y1 = np.append(y1, y1[-1])
        y2 = np.append(y2, y2[-1])
        
        color = COLORS[col]
        alpha = ALPHAS[col]
        hatch = HATCHES[col]
        ls = STYLE[col]
        lw = 2

        # Italicize GALEX
        if col == 'GALEX':
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
        
        # if col != 'This study':
        else:
            # Upper bound
            ax.step(x, y2, c=color, ls=ls, lw=lw, where='post', label=label)
            # Lower bound
            ax.step(x, y1, c=color, ls=ls, lw=lw, where='post')
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

    # Format axes
    ax.set_xlabel('$t_{start}$ [days]')
    ax.set_ylabel('Rate of CSM interaction [%]', rotation='horizontal', 
                ha='left', va='top', y=1.12, labelpad=-2)
    ax.set_ylim((None, y_max))
    ylim = ax.get_ylim()

    # Set spine extent
    ax.spines['bottom'].set_bounds(x[0], x[-1])
    ax.spines['left'].set_bounds(0, ylim[1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['right'].set_bounds(0, ylim[1])
        
    # Set ticks
    x_minor_ticks = np.arange(x[0], x[-1]+0.1, 100)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(x_minor_ticks))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    dy = int((ylim[1] - 0) / 10)
    y_minor_ticks = np.arange(0, ylim[1]+0.1, dy)
    ax.yaxis.set_minor_locator(ticker.FixedLocator(y_minor_ticks))
    y_major_ticks = np.arange(0, ylim[1]+0.1, 2*dy)
    ax.yaxis.set_major_locator(ticker.FixedLocator(y_major_ticks))
    ax.tick_params(which='both', top=False, right=False)

    # Plot external study estimates as points
    pt_x = -int(x[-1] / 40)
    text_y = -2.2*dy
    for i, study in enumerate(external.index):
        row = external.loc[study]
        midpoint = row.mean()
        err = row['bci_upper'] - midpoint
        ax.errorbar(pt_x, midpoint, yerr=err, marker=MARKERS[study], 
                c=COLORS[study], mec=COLORS[study], mfc='w', ms=0, 
                linestyle='none', elinewidth=5, mew=2, capsize=0)#, label=study)
        # In-plot label
        # if i % 2 == 1:
        #     text_y = midpoint + err + 1
        #     va = 'bottom'
        # else:
        #     text_y = -1
        #     va = 'top'
        # ax.text(pt_x * 1.3, text_y, study, color=COLORS[study], size=TEXT_SIZE-2,
        #         ha='left', va=va)
        ax.annotate(study, xy=(pt_x, 0), xytext=(pt_x, text_y), ha='right',
                size=TEXT_SIZE-2, va='center', 
                arrowprops={'color': 'k', 'shrink': 0.08, 'width': 0.5,
                        'headwidth': 0.5, 'headlength': 0.5})
        pt_x += pt_x
        text_y += dy/1.2

    # White grid
    ax.grid(b=True, which='both', axis='x', color='w', lw=2, zorder=2)
    ax.grid(b=True, which='both', axis='y', color='w', lw=1, zorder=2)

    ax.set_ylim((-dy, None))

    # ax.set_ylim((0.1, None))
    # ax.set_yscale('log')
    # ax.set_xlim((50, None))
    # ax.set_xscale('log')

    # Legend (actually upper right)
    plt.legend(loc='lower right', ncol=3, bbox_to_anchor=(1., 1.), 
            handletextpad=0.8, handlelength=2., borderpad=0.4)

    plt.savefig(output_file, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


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
        # df = pd.read_csv(file_path, sep='\s+', index_col=0, na_values='---', 
        #         skiprows=skip_row, usecols=cols, names=)
        df['disc_age'] = df['disc_age'].astype(float)
        data.append(df)

    data = pd.concat(data)
    # print(data)
    # print(np.min(data['disc_age']))
    # print(np.mean(data['disc_age']))
    # print(np.max(data['disc_age']))

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


# def table(detections, trials, bci_upper, tstart_bins=TSTART_BINS, 
#         output_file='out/rates.tex'):
#     """Generate LATEX table to go along with plot."""
    
#     # Combine to single DataFrame
#     df = pd.melt(detections.reset_index(), id_vars='index', var_name='Source', 
#             value_name='Detections')
#     df['Trials'] = pd.melt(trials.reset_index(), id_vars='index')['value']
#     df['Upper BCI'] = pd.melt(bci_upper.reset_index(), id_vars='index')['value']
#     # Remove empty rows & rename epochs col
#     df = df[pd.notna(df['Upper BCI'])].rename(columns={'index': 'Epoch'})
#     # Replace index and sort
#     df = df.sort_values('Epoch').set_index('Epoch')
#     # df = df.astype({'Detections': int, 'Trials': int})
#     # Rearrange columns
#     df = df[['Detections', 'Trials', 'Upper BCI', 'Source']]

#     # Row labels
#     rows = {}
#     for i in range(len(tstart_bins)-1):
#         rows[tstart_bins[i]] = '%s - %s' % (tstart_bins[i], tstart_bins[i+1])
#     new_index = pd.Series(np.vectorize(rows.get)(df.index), name='Epoch')
#     df.set_index(new_index, inplace=True)

#     table = df.to_latex(formatters={'Source': source_fmt, 'Upper BCI': '{:.1f}'.format,
#               'Detections': '{:.1f}'.format, 'Trials': '{:.1f}'.format}, escape=False)
#     # Replace table header and footer with template
#     # Edit this file if you need to change the number of columns or description
#     with open(Path('ref/deluxetable_template.tex'), 'r') as file:
#         dt_file = file.read()
#         header = dt_file.split('===')[0]
#         footer = dt_file.split('===')[1]
#     table = header + '\n'.join(table.split('\n')[5:-3]) + footer
#     # Write table
#     with open(Path(output_file), 'w') as file:
#         file.write(table)


# def source_fmt(source):
#     """Format source string in LATEX table."""

#     fmt = { 'GALEX': '$\it{%s}$' % source, 
#             'HST': '\citetalias{Graham2019-SN2015cp}',
#             'ASAS-SN': '%s\\tablenotemark{a}' % source,
#             'ZTF': '%s\\tablenotemark{b}' % source,
#             'This study': 'This study', 'All': 'All'}

#     return fmt[source]


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
    args = parser.parse_args()

    main(model=args.model, scale=args.scale, bin_width=args.twidth,
            sigma=args.sigma, t_max=args.tmax, y_max=args.ymax)