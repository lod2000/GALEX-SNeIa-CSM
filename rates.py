from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import binom_conf_interval
from utils import *
from plot_recovery import sum_hist

# TSTART_BINS = [0, 20, 100, 500, 1000, 2500]
# Defaults
CONF = 0.9 # binomial confidence level
MODEL = 'Chev94' # default spectral model
SCALE = 1 # default model scale
SIGMA = 3 # default confidence for excluded SNe
TSTART_MAX = 2000
YMAX = 20

# Plot settings
COLORS = {  'GALEX': '#d81b60',
            'G19': '#004d40',
            'This study': 'k',
            'ASAS-SN': '#ffc107',
            'ZTF': '#1e88e5',
            'All': 'k'
}
MARKERS = { 'GALEX': 'o', 
            'G19': 'o', 
            'This study': 'o', 
            'ASAS-SN': 's', 
            'ZTF': 's', 
            'All': '*'
}


def main(bin_width=TSTART_BIN_WIDTH, scale=SCALE, iterations=10000, 
        model=MODEL, sigma=SIGMA, t_min=TSTART_MIN, t_max=TSTART_MAX):
    
    # Bin edges
    x_edges = np.arange(t_min, t_max+bin_width, bin_width)
    x_midpoints = (x_edges[1:] + x_edges[:-1]) / 2
    y_edges = np.array([scale, SCALE_MAX])
    nbins = len(x_edges)-1

    # Import non-detections and sum histograms for GALEX and G19
    histograms = []
    for study in ['galex', 'graham']:
        # File names
        save_dir = run_dir(study, model, sigma)
        save_files = list(Path(save_dir).glob('*-%s.csv' % iterations))
        # Generate summed histogram
        print('Importing and summing %s saves...' % study)
        hist = sum_hist(save_files, x_edges, y_edges, save=False)
        histograms.append(hist.iloc[0].to_numpy())

    [galex_hist, graham_hist] = histograms
    uv_hist = galex_hist + graham_hist

    # Import detections
    save_dir = run_dir('graham_det', model, sigma)
    save_files = list(Path(save_dir).glob('*-%s.csv' % iterations))
    # Generate summed histogram
    print('Importing and summing G19 detections...')
    hist = sum_hist(save_files, x_edges, y_edges, save=False)
    graham_det_hist = np.nan_to_num(hist.iloc[0].to_numpy())
    # graham_detections = pd.read_csv('ref/Graham_detections.csv')
    # graham_det_hist = np.histogram(graham_detections['Rest Phase'], x_edges)[0]

    # DataFrame for number of trials per tstart bin and data source
    # sources = ['G19', 'GALEX', 'ASAS-SN', 'ZTF']
    sources = ['G19', 'GALEX']
    index = pd.Series(x_edges[:-1])
    trials = pd.DataFrame([], index=index)
    trials['G19'] = (graham_hist + graham_det_hist).T
    trials['GALEX'] = galex_hist.T
    trials['This study'] = trials['GALEX'] + trials['G19']
    # trials['ASAS-SN'] = np.array([464] + [0]*(nbins-1)).T
    # trials['ZTF'] = np.array([127] + [0]*(nbins-1)).T
    # trials['All'] = trials[sources].sum(axis=1)
    # trials.loc[trials['All'] == trials['This study'], 'All'] = 0

    # DataFrame for number of detections per tstart bin and data source
    detections = pd.DataFrame([], index=index)
    detections['G19'] = graham_det_hist.T
    detections['GALEX'] = np.zeros((nbins, 1))
    detections['This study'] = detections['GALEX'] + detections['G19']
    # detections['ASAS-SN'] = np.array([3] + [0]*(nbins-1)).T
    # detections['ZTF'] = np.array([1] + [0]*(nbins-1)).T
    # detections['All'] = detections[sources].sum(axis=1)

    bci_lower = pd.DataFrame([], index=index)
    bci_upper = pd.DataFrame([], index=index)

    # Calculate binomial confidence intervals
    for col in trials.columns:
        # separate bins with no trials
        pos_index = trials[trials[col] > 0].index
        zero_index = trials[trials[col] == 0].index

        bci = 100 * binom_conf_interval(detections.loc[pos_index, col], 
                trials.loc[pos_index, col], confidence_level=CONF, 
                interval='jeffreys')
        print(col)
        print(bci)
        
        # add to dataframes (nan for no trials)
        bci_lower.loc[pos_index, col] = bci[0].T
        bci_lower.loc[zero_index,col] = np.nan
        bci_upper.loc[pos_index, col] = bci[1].T
        bci_upper.loc[zero_index,col] = np.nan

    print(bci_lower)

    # table(detections, trials, bci_upper, tstart_bins=TSTART_BINS, 
    #         output_file=Path('out/rates_%s.tex' % model))

    plot(x_midpoints, bci_lower, bci_upper, show=True,)
            # output_file=Path('out/rates_%s.pdf' % model))    


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


def source_fmt(source):
    """Format source string in LATEX table."""

    fmt = { 'GALEX': '$\it{%s}$' % source, 
            'G19': '\citetalias{Graham2019-SN2015cp}',
            'ASAS-SN': '%s\\tablenotemark{a}' % source,
            'ZTF': '%s\\tablenotemark{b}' % source,
            'This study': 'This study', 'All': 'All'}

    return fmt[source]


def plot(x, bci_lower, bci_upper, output_file='out/rates.pdf', show=True):
    """Plot binomial confidence limits for CSM interaction rate."""

    fig, ax = plt.subplots()

    for col in bci_lower.columns:
        # Find midpoint and errors for plotting
        y1 = bci_lower[col]
        y2 = bci_upper[col]
        midpoint = np.mean([y1, y2], axis=0)
        color = COLORS[col]

        if col == 'This study':
            ax.fill_between(x, y1, y2, color=color, alpha=0.1)
        else:
            ax.plot(x, y1, c=color, ls='--')
            ax.plot(x, y2, c=color, ls='--')

        ax.plot(x, midpoint, color=COLORS[col], lw=2, label=col)
        # err = bci_upper[col].to_numpy() - midpoint
        # line width
        # lw = 3.5 if col == 'This study' else 2
        # italicize GALEX
        # label = '$\it{%s}$' % col if col == 'GALEX' else col
        # marker size
        # ms = 16 if col == 'All' else 10
        # plot
        # ax.errorbar(x_pos + x_adjust, midpoint, yerr=err, label=label, 
        #         marker=MARKERS[col], c=COLORS[col], mec=COLORS[col], mfc='w', 
        #         ms=ms, linestyle='none', elinewidth=lw, mew=lw, capsize=6)

    # Format axis
    # ax.set_xlim((x_pos[0]-0.7, x_pos[-1]+1.8))
    # ax.set_xticks(np.append(x_pos, nbins)-0.5)
    # ax.set_xticklabels(tstart_bins)
    # ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.set_xlabel('$t_{start}$ [rest frame days post-discovery]')
    ax.set_ylabel('Rate of CSM interaction [%]')

    ax.set_ylim((0, YMAX))

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # remove errorbars
    # handles = [h[0] for h in handles]
    plt.legend(handles, labels, loc='upper left')

    plt.tight_layout()
    # plt.savefig(output_file, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='Chev94', 
            help='CSM spectrum model')
    parser.add_argument('--scale', '-S', type=float, default=SCALE)
    parser.add_argument('--sigma', type=int, nargs='+', default=[SIGMA], 
            help='Detection confidence level (multiple for tiered detections)')
    parser.add_argument('--tmax', type=int, default=TSTART_MAX, 
            help='Maximum CSM interaction start time')
    args = parser.parse_args()

    main(model=args.model, scale=args.scale, 
            sigma=args.sigma, t_max=args.tmax)