from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import binom_conf_interval
from utils import *
from plot import sum_hist

TSTART_BINS = [0, 20, 100, 500, 1000]
CONF = 0.9
MODEL = 'Chev94'
SCALE = 1
SIGMA = 3

COLORS = {'GALEX': '#d81b60',
          'G19': '#004d40',
          'UV': 'k',
          'ASAS-SN': '#ffc107',
          'ZTF': '#1e88e5',
          'All': 'k'}
MARKERS = {'GALEX': 'o', 'G19': 'o', 'UV': 'o', 
           'ASAS-SN': 's', 'ZTF': 's', 'All': '*'}


def main(tstart_bins=TSTART_BINS, scale=SCALE, iterations=10000, overwrite=False, 
        model=MODEL, sigma=SIGMA):

    # Scale to UV luminosity of SN 2015cp
    reduced_scale = SN2015cp_scale(model)
    
    # Bin edges
    x_edges = np.array(tstart_bins)
    y_edges = np.array([scale, 1000 / reduced_scale])
    nbins = len(tstart_bins)-1

    # Import non-detections and sum histograms for GALEX and G19
    histograms = []
    for study in ['galex', 'graham']:
        # File names
        save_dir = run_dir(study, model, sigma)
        hist_file = save_dir / Path('out/rates_hist.csv')
        save_files = list(Path(save_dir).glob('*-%s.csv' % iterations))

        # Generate summed histogram
        if overwrite or not hist_file.is_file():
            print('Importing and summing %s saves...' % study)
            hist = sum_hist(save_files, x_edges, y_edges, output_file=hist_file,
                    reduced_scale=reduced_scale)
        else:
            print('Importing %s histograms...' % study)
            hist = pd.read_csv(hist_file, index_col=0)
        
        histograms.append(hist.iloc[0].to_numpy())

    [galex_hist, graham_hist] = histograms
    uv_hist = galex_hist + graham_hist

    # Import detections
    graham_detections = pd.read_csv('ref/Graham_detections.csv')
    graham_det_hist = np.histogram(graham_detections['Rest Phase'], tstart_bins)[0]

    # DataFrame for number of trials per tstart bin and data source
    sources = ['G19', 'GALEX', 'ASAS-SN', 'ZTF']
    index = pd.Series(tstart_bins[:-1])
    trials = pd.DataFrame([], index=index)
    trials['G19'] = (graham_hist + graham_det_hist).T.astype(int)
    trials['GALEX'] = galex_hist.T.astype(int)
    trials['UV'] = trials['GALEX'] + trials['G19']
    trials['ASAS-SN'] = np.array([464] + [0]*(nbins-1)).T
    trials['ZTF'] = np.array([127] + [0]*(nbins-1)).T
    trials['All'] = trials[sources].sum(axis=1)
    trials.loc[trials['All'] == trials['UV'], 'All'] = 0

    # DataFrame for number of detections per tstart bin and data source
    detections = pd.DataFrame([], index=index)
    detections['G19'] = graham_det_hist.T
    detections['GALEX'] = np.zeros((nbins, 1))
    detections['UV'] = detections['GALEX'] + detections['G19']
    detections['ASAS-SN'] = np.array([3] + [0]*(nbins-1)).T
    detections['ZTF'] = np.array([1] + [0]*(nbins-1)).T
    detections['All'] = detections[sources].sum(axis=1)

    # Calculate binomial confidence intervals
    bci_lower = pd.DataFrame([], index=index)
    bci_upper = pd.DataFrame([], index=index)
    for col in trials.columns:
        # separate bins with no trials
        pos_index = trials[trials[col] > 0].index
        zero_index = trials[trials[col] == 0].index

        bci = 100 * binom_conf_interval(detections.loc[pos_index, col], 
                trials.loc[pos_index, col], confidence_level=CONF, 
                interval='jeffreys')
        
        # add to dataframes (nan for no trials)
        bci_lower.loc[pos_index, col] = bci[0].T
        bci_lower.loc[zero_index,col] = np.nan
        bci_upper.loc[pos_index, col] = bci[1].T
        bci_upper.loc[zero_index,col] = np.nan

    table(detections, trials, bci_upper, tstart_bins=TSTART_BINS, 
            output_file=Path('out/rates_%s.tex' % model))

    plot(bci_lower, bci_upper, tstart_bins=tstart_bins, show=True,
            output_file=Path('out/rates_%s.pdf' % model))    


def table(detections, trials, bci_upper, tstart_bins=TSTART_BINS, 
        output_file='out/rates.tex'):
    """Generate LATEX table to go along with plot."""
    
    # Combine to single DataFrame
    df = pd.melt(detections.reset_index(), id_vars='index', var_name='Source', 
            value_name='Detections')
    df['Trials'] = pd.melt(trials.reset_index(), id_vars='index')['value']
    df['Upper BCI'] = pd.melt(bci_upper.reset_index(), id_vars='index')['value']
    # Remove empty rows & rename epochs col
    df = df[pd.notna(df['Upper BCI'])].rename(columns={'index': 'Epoch'})
    # Replace index and sort
    df = df.sort_values('Epoch').set_index('Epoch')
    df = df.astype({'Detections': int, 'Trials': int})
    # Rearrange columns
    df = df[['Detections', 'Trials', 'Upper BCI', 'Source']]

    # Row labels
    rows = {}
    for i in range(len(tstart_bins)-1):
        rows[tstart_bins[i]] = '%s - %s' % (tstart_bins[i], tstart_bins[i+1])
    new_index = pd.Series(np.vectorize(rows.get)(df.index), name='Epoch')
    df.set_index(new_index, inplace=True)

    table = df.to_latex(formatters={'Source': source_fmt, 'Upper BCI': '{:.2f}'.format}, escape=False)
    # Replace table header and footer with template
    # Edit this file if you need to change the number of columns or description
    with open(Path('ref/deluxetable_template.tex'), 'r') as file:
        dt_file = file.read()
        header = dt_file.split('===')[0]
        footer = dt_file.split('===')[1]
    table = header + '\n'.join(table.split('\n')[5:-3]) + footer
    # Write table
    with open(Path(output_file), 'w') as file:
        file.write(table)


def source_fmt(source):
    """Format source string in LATEX table."""

    fmt = { 'GALEX': '$\it{%s}$' % source, 
            'G19': '\citetalias{Graham2019-SN2015cp}',
            'ASAS-SN': '%s\\tablenotemark{a}' % source,
            'ZTF': '%s\\tablenotemark{b}' % source,
            'UV': 'Both UV', 'All': 'All'}

    return fmt[source]


def plot(bci_lower, bci_upper, tstart_bins=TSTART_BINS, 
        output_file='out/rates.pdf', show=True):
    """Plot binomial confidence limits for CSM interaction rate at multiple epochs."""

    fig, ax = plt.subplots()

    # x-axis position of bounds
    nbins = len(tstart_bins)-1
    x_pos = np.arange(nbins)
    x_step = 0.12
    # horizontal position adjustment
    x_adjust = np.array([-0.3] + [-x_step] * (nbins-1))

    for col in bci_lower.columns:
        # Find midpoint and errors for plotting
        midpoint = np.mean([bci_lower[col], bci_upper[col]], axis=0)
        err = bci_upper[col].to_numpy() - midpoint
        # line width
        lw = 3.5 if col == 'UV' else 2
        # italicize GALEX
        label = '$\it{%s}$' % col if col == 'GALEX' else col
        # marker size
        ms = 16 if col == 'All' else 10
        # plot
        ax.errorbar(x_pos + x_adjust, midpoint, yerr=err, label=label, 
                marker=MARKERS[col], c=COLORS[col], mec=COLORS[col], mfc='w', 
                ms=ms, linestyle='none', elinewidth=lw, mew=lw, capsize=6)
        x_adjust += x_step

    # Format axis
    ax.set_xlim((x_pos[0]-0.7, x_pos[-1]+1.8))
    ax.set_xticks(np.append(x_pos, nbins)-0.5)
    ax.set_xticklabels(tstart_bins)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.set_xlabel('CSM interaction start time [rest frame days post-discovery]')
    ax.set_ylabel('Rate of CSM interaction [%]')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # remove errorbars
    handles = [h[0] for h in handles]
    plt.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', '-o', action='store_true', 
            help='Overwrite histograms')
    parser.add_argument('--model', '-m', type=str, default='Chev94', 
            help='CSM spectrum model')
    parser.add_argument('--scale', '-S', type=float, default=SCALE)
    parser.add_argument('--sigma', type=int, nargs='+', default=[SIGMA], 
            help='Detection confidence level (multiple for tiered detections)')
    args = parser.parse_args()

    main(overwrite=args.overwrite, model=args.model, scale=args.scale, 
            sigma=args.sigma)