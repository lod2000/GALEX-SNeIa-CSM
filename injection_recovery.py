from tqdm import tqdm
import itertools
from multiprocessing import Pool
from functools import partial
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import random

from utils import *
from CSMmodel import CSMmodel
from light_curve import LightCurve
from supernova import Supernova

# Default values
DECAY_RATE = 0.3
RECOV_MIN = 50 # minimum number of days after discovery to count as recovery
SIGMA = 3
WIDTH = 250 # days, from PTF11kx

# def main(iterations, overwrite=False, tstart_max=1000, scale_min=0.5, 
#             scale_max=2., t_max=1500, bin_width=50, bin_height=0.1):
def main():

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')

    sn = Supernova('SDSS-II SN 16121', sn_info)
    lc = LightCurve(sn, 'NUV')
    inj = Injection(sn, lc, (600, 610), (0.5, 2))
    print(inj.recover([5, 3], count=[1, 3]))
    print(inj.tstart)
    print(inj.scale)
    inj.plot()

    # sys.path.insert(0, 'CSMmodel')
    # print(sys.path)

    # output_file = Path('out/recovery_%s.csv' % iterations)
    
    # sn = Supernova('SN2007on')
    # times = sample_params(10, sn, 'NUV', 500, 700, 1, 1.1)
    # print(times)

    # # supernovae = ['SN2007on', 'SN2010ai', 'SDSS-II SN 779', 'Hawk', 'HST04Sas']
    # supernovae = sn_info.index.to_list()

    # if overwrite or not output_file.is_file():
    #     recovered_times = run_ir(iterations, supernovae, 0, tstart_max, 
    #             scale_min, scale_max, bin_width, bin_height, t_max, output_file)
    #     count_hist = count_nondetections(recovered_times, bin_width, t_max, 
    #             bin_height, scale_min, scale_max)
    #     output_csv(count_hist, output_file)
    # else:
    #     count_hist = pd.read_csv(output_file, index_col=0)

    # plot_nondetections(count_hist, show=True)


def plot_nondetections(rate_hist, show=False):
    """Plot 2D histogram of recovery rate by time since discovery and scale factor."""

    # Flip y-axis
    rate_hist.sort_index(ascending=True, inplace=True)

    # Calculate data range
    x_bins = rate_hist.columns.to_numpy(dtype=float)
    y_bins = rate_hist.index.to_numpy(dtype=float)
    bin_width = x_bins[1] - x_bins[0]
    bin_height = y_bins[1] - y_bins[0]
    extent = (x_bins[0], x_bins[-1]+bin_width, y_bins[0], y_bins[-1]+bin_height)

    # Plot
    fig, ax = plt.subplots()
    im = ax.imshow(rate_hist, aspect='auto', origin='lower', extent=extent)
    ax.xaxis.set_minor_locator(MultipleLocator(bin_width))
    ax.yaxis.set_minor_locator(MultipleLocator(bin_height))
    ax.set_xlabel('Rest frame time since discovery [days]')
    ax.set_ylabel('Scale factor')
    plt.colorbar(im, label='No. of excluded SNe Ia')
    fig.tight_layout()
    plt.savefig(Path('out/recovery.png'), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def count_nondetections(recovered_times, bin_width, x_max, bin_height, y_min, y_max):
    """Generate 2D histogram of nondetection counts by time and scale factor
    Inputs:
        recovered_times: list of dicts
        bin_width: bin width in heatmap plot, in days
        x_max: max value on x-axis
        bin_height: bin height in heatmap plot, in scale factor fraction
        y_min: minimum value on y-axis
        y_max: max value on y-axis
    Output:
        count_sum: 2D histogram of nondetection counts
    """
    
    print('\nBinning recovery rates...')
    # Make lists of recovered points and all points
    recovered = []
    total = []
    for rec_dict in tqdm(recovered_times):
        recovered += [[rec_dict['sn'], time, rec_dict['scale']] for time in rec_dict['recovered']]
        total += [[rec_dict['sn'], time, rec_dict['scale']] for time in rec_dict['all']]

    recovered = np.array(recovered)
    total = np.array(total)

    # Bin edges
    x_edges = np.arange(RECOV_MIN, x_max, bin_width)
    y_edges = np.arange(y_min, y_max, bin_height)

    # Dummy array if recovered is empty
    if recovered.shape == (0,):
        return pd.DataFrame(np.full((len(x_edges), len(y_edges)), 0), 
                index=y_edges[:-1], columns=x_edges[:-1])

    # Count nondetections per supernova
    counts = []
    for sn_name in list(dict.fromkeys(recovered[:,0])):
        # Select by sn name
        sn_rec = recovered[recovered[:,0] == sn_name][:,1:].astype(float)
        sn_tot = total[total[:,0] == sn_name][:,1:].astype(float)
        # Recovery rate histogram
        rate_hist = recovery_histogram(sn_rec[:,0], sn_rec[:,1], sn_tot[:,0], 
                sn_tot[:,1], x_edges, y_edges)
        counts.append(rate_hist)

    # Sum all histograms
    count_sum = reduce(lambda x, y: x.add(y, fill_value=0), counts)
    return count_sum


def recovery_histogram(x_recovered, y_recovered, x_total, y_total, x_edges, y_edges):
    """Generate histogram of recovery rate given x and y recovered/total values.
    Inputs:
        x_recovered: x-values for recovered data
        y_recovered: y-values for recovered data
        x_total: x-values for all data
        y_total: y-values for all data
        x_edges: bin edges for x data
        y_edges: bin edges for y data
    Output:
        rate_hist: 2D histogram of recovery rate
    """

    # 2D histograms for recovered data and total data
    recovered = np.histogram2d(x_recovered, y_recovered, [x_edges, y_edges])[0]
    total = np.histogram2d(x_total, y_total, [x_edges, y_edges])[0]

    # Calculate recovery rate
    rate_hist = recovered / total

    # Transpose and convert to DataFrame with time increasing along the rows
    # and scale height increasing down the columns. Column and index labels
    # are the lower bound of each bin
    rate_hist = pd.DataFrame(rate_hist.T, index=y_edges[:-1], columns=x_edges[:-1])
    return rate_hist


def run_ir(iterations, supernovae, tstart_min, tstart_max, scale_min, scale_max,
        bin_width, bin_height, t_max, output_file):
    """Run injection recovery with random parameters for a list of supernovae.
    Inputs:
        iterations: number of times to sample parameter space
        supernovae: list of SN names
        tstart_min: minimum CSM model start time
        tstart_max: maximum CSM model start time
        scale_min: minimum CSM model scale factor
        scale_max: maximum CSM model scale factor
    Outputs:
        recovered_times: list of dicts
    """

    # List of supernovae and bands to perform injection-recovery
    recovered_times = []
    to_remove = []
    supernovae = sorted(list(supernovae) * 2)

    # Load progress file, if any
    # progress_file = Path('out/progress_%s.npy' % iterations)
    # if progress_file.is_file():
        # print('\nLoading previous progress file...')
        # recovered_times = list(np.load(progress_file, allow_pickle=True))
        # to_remove = [(rec['sn'], rec['band']) for rec in recovered_times]

    bands = ['FUV', 'NUV'] * len(supernovae)

    # Iterate over supernovae, bands
    for i, (sn_name, band) in enumerate(zip(supernovae, bands)):

        # Skip previously run SNe
        if (sn_name, band) in to_remove:
            continue

        # Save to binary numpy file every 10 iterations (takes a long time)
        if i % 10 == 0 and i != 0:
            print('Saving progress...')
            # np.save(progress_file, np.array(recovered_times))
            print('Progress saved.')

        # Plot histogram every 50 iterations
        if i % 50 == 0 and i != 0:
            rate_hist = count_nondetections(recovered_times, bin_width, t_max, 
                    bin_height, scale_min, scale_max)
            output_csv(rate_hist, output_file)
            plot_nondetections(rate_hist, show=False)

        # Ignore if light curve file doesn't exist
        lc_file = LC_DIR / sn2fname(sn_name, band)
        if not lc_file.is_file():
            print('\nNo light curve file found for %s - %s [%s/%s]' % (sn_name, band, i+1, len(supernovae)))
            recovered_times.append({
                'sn': sn_name,
                'band': band,
                'tstart': -1,
                'scale': -1,
                'recovered': [],
                'all': []
            })
            continue

        try:
            print('\n%s - %s [%s/%s]' % (sn_name, band, i+1, len(supernovae)))
            sn = Supernova(sn_name)
            # Run injection-recovery on many randomly sampled parameters
            sample_times = sample_params(iterations, sn, band, tstart_min, 
                    tstart_max, scale_min, scale_max)
            # Append resulting recovered times
            recovered_times += sample_times
        except KeyError:
            continue

    return recovered_times


def sample_params(iterations, sn, band, tstart_min, tstart_max, scale_min, scale_max):
    """Run injection recovery on a single SN for a given number of iterations.
    Inputs:
        iterations: int
        sn: Supernova object
        band: 'FUV' or 'NUV'
        tstart_min: minimum CSM model start time
        tstart_max: maximum CSM model start time
        scale_min: minimum CSM model scale factor
        scale_max: maximum CSM model scale factor
    Outputs:
        sample_times: list of dicts
    """

    # Randomly sample start times (ints) and scale factors (floats)
    tstarts = np.random.randint(tstart_min, tstart_max, size=iterations)
    scales = (scale_max - scale_min) * np.random.rand(iterations) + scale_min
    params = np.array(list(zip(tstarts, scales)))

    # Import light curve for SN
    lc = LightCurve(sn, band)
    all_times = lc.data[lc.data['t_delta_rest'] >= RECOV_MIN]['t_delta_rest'].to_list()
    sample_times = []

    # Run injection-recovery in parallel for each sampled CSM parameter
    with Pool() as pool:
        func = partial(inject_recover, sn=sn, lc=lc)
        imap = pool.imap(func, params, chunksize=10)
        for times in tqdm(imap, total=params.shape[0]):
            sample_times.append(times)

    # List of recovered times and associated parameters
    sample_times = [
            {   'sn': sn.name,
                'band': band,
                'tstart': params[i,0], 
                'scale': params[i,1], 
                'recovered': sample_times[i],
                'all': all_times} 
            for i in range(iterations)]

    return sample_times


def inject_recover(params, sn, lc):
    """Perform injection and recovery for given SN and model parameters.
    Inputs:
        params: [tstart, scale]
        sn: Supernova object
        lc: LightCurve object
    Output:
        list of times of recovered data
    """

    tstart, scale = params
    injected = inject_model(sn, lc, tstart, scale)
    recovered = recover_model(injected)
    # Return days post-discoverey with recovered detections
    return recovered['t_delta_rest'].to_list()


def inject_model(sn, lc, tstart, scale):
    """
    Inject CSM model into GALEX data and return the resulting light curve.
    Inputs:
        sn: Supernova object
        lc: LightCurve object
        tstart: days after discovery that ejecta impacts CSM
        scale: luminosity scale factor
    Output:
        lc: LightCurve object with injected light curve
    """

    data = lc.data.copy()
    model = CSMmodel(tstart, WIDTH, DECAY_RATE, scale=scale)
    # Calculate luminosity at observation epochs
    injection = model(data['t_delta_rest'], sn.z)[lc.band]
    # Inject CSM curve
    data['luminosity_injected'] = data['luminosity_hostsub'] + injection
    return data


def recover_model(data):
    """
    Recover detections from CSM-injected data which otherwise wouldn't have
    been detected.
    """

    # Calculate significance of each point
    data['sigma_injected'] = data['luminosity_injected'] / data['luminosity_hostsub_err']
    # Recover new detections
    recovered = data[(data['sigma_injected'] >= SIGMA) & (data['sigma'] < SIGMA)]
    # Limit to points some time after discovery (default 50 days)
    recovered = recovered[recovered['t_delta_rest'] >= RECOV_MIN]
    return recovered


class Injection:
    def __init__(self, sn, lc, tstarts, scales, width=250, decay=0.3):
        """Generate random model parameters, initialize model and inject into
        data.
        Inputs:
            sn: Supernova object associated with data
            lc: LightCurve object with data to be injected
            tstarts: (min, max) tuple of tstart parameter bounds
            scales: (min, max) tuple of scale parameter bounds
            width: model width, int
            decay: model decay rate, float
        """
        
        # Generate random parameters
        self.tstart = random.randint(tstarts[0], tstarts[1])
        self.scale = random.uniform(scales[0], scales[1])

        # Other parameters
        self.width = width
        self.decay = decay

        # Get data
        self.time = lc.data['t_delta_rest'].copy()
        self.data = lc.data['luminosity_hostsub'].copy()
        self.err = lc.data['luminosity_hostsub_err'].copy()

        # Inject model
        self.model = CSMmodel(self.tstart, self.width, self.decay, 
                scale=self.scale)
        print(self.model(self.time, sn.z)[lc.band])
        self.injection = self.data + self.model(self.time, sn.z)[lc.band]


    def __call__(self):
        return self.injection


    @classmethod
    def from_name(self, sn_name, band, tstarts, scales, sn_info=[], **kwargs):
        """Generate Injection instance from a supernova name and GALEX band,
        also creating a Supernova and LightCurve object in the process."""

        sn = Supernova(sn_name, sn_info=sn_info)
        lc = LightCurve(sn, band)
        return Injection(sn, lc, tstarts, scales, **kwargs)


    def recover(self, sigma, count=[1], dt_min=50, detections=None):
        """Run detection algorithm on injected data and return points which 
        otherwise would not have been recovered.
        Inputs:
            sigma: float or list, confidence level required for detection (if 
                    multiple, use multi-tier detection)
            count: list, number of points at or above associated sigma to count
                    as a detection (same length as sigma)
            dt_min: minimum time since discovery to include detections
            detections: indices which were detected without injection; pass for
                    faster calcs
        """

        # Run detections on original data
        if detections == None:
            detections = detect_csm(self.time, self.data, self.err, sigma, 
                    count=count, dt_min=dt_min)

        # Run detections on injected data
        recovered = detect_csm(self.time, self.injection, self.err, sigma,
                count=count, dt_min=dt_min)

        # Remove points that would have been detected either way
        recovered = [r for r in recovered if r not in detections]

        return recovered


    def plot(self):
        plt.errorbar(self.time, self.data, yerr=self.err, label='Original', 
                linestyle='none', marker='o')
        plt.errorbar(self.time, self.injection, yerr=self.err, label='Injected',
                linestyle='none', marker='o')
        plt.xlim((0, None))
        plt.xlabel('Time since discovery [rest frame days]')
        plt.ylabel('Luminosity [erg s$^{-1}$ Å$^{-1}$]')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('iter', type=int, help='Iterations')
    # parser.add_argument('--overwrite', '-o', action='store_true',
    #         help='Overwrite recovery rate output file')
    # args = parser.parse_args()
# 
    # main(args.iter, overwrite=args.overwrite)
    main()