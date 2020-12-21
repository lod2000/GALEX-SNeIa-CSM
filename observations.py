from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.utils.exceptions import AstropyWarning

from tqdm import tqdm
import warnings
import platform

# from multiprocessing import Pool
# from itertools import repeat
# from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from pathlib import Path
import pandas as pd
import numpy as np

from supernova import Supernova
from utils import *

OUTPUT_FILE = Path('out/observations.csv')
DATA_DIR = Path('/mnt/d/GALEXdata_v10/fits/')
# SN_INFO_FILE = Path('ref/sn_info.csv')
# STATS_FILE = Path('out/quick_stats.txt')


def main(data_dir=DATA_DIR, overwrite=False, osc_file=OSC_FILE):

    # Suppress Astropy warnings about "dubious years", etc.
    warnings.simplefilter('ignore', category=AstropyWarning)

    # Read Open Supernova Catalog
    osc = pd.read_csv(osc_file, index_col='Name')
    print('Number of SNe from OSC: %s' % len(osc.index))

    # Generate new list of FITS files
    observations = []
    if args.overwrite or not OUTPUT_FILE.is_file():
        # Get all FITS file paths
        fits_files = get_fits_files(data_dir, limit_to=osc.index)

        # Import FITS info and discovery dates
        print('\nImporting GALEX observations...')
        for f in tqdm(fits_files):
            obs = GalexObservation(f)
            disc_date = osc.loc[obs.sn_name, 'Disc. Date']
            obs_df = obs.get_DataFrame(disc_date=disc_date, date_fmt='iso')
            observations.append(obs_df)

        # Concatenate and output CSV
        observations = pd.concat(observations, ignore_index=True)
        observations.to_csv(OUTPUT_FILE, index=False)
    else:
        print('\nFound %s' % OUTPUT_FILE)
        observations = pd.read_csv(OUTPUT_FILE)

    galex_sne = observations['sn_name'].drop_duplicates()
    print('Number of SNe observed by GALEX: %s' % len(galex_sne))

    # Plot histograms of observation epochs
    plot(observations)

    # Select only those with before+after observations
    pre_post_obs = get_pre_post_obs(observations)
    pre_post_obs.to_csv('out/sample_obs.csv', index=False)

    # final_sample = get_pre_post_obs(fits_info) 
    # output_csv(final_sample, 'ref/sample_fits_info.csv')

    # # Output compressed CSV without SN name duplicates
    # sn_info = compress_duplicates(final_sample.copy())
    # output_csv(sn_info, SN_INFO_FILE)

    # # Plot histogram of observations
    # plot_observations(fits_info, final_sample)

    # # Write a few statistics about FITS files
    # write_quick_stats(fits_info, final_sample, sn_info, osc, STATS_FILE)


def get_fits_files(data_dir, limit_to=[]):
    """
    Return list of FITS files in given data directory; limits to SNe listed in
    OSC reference table, if given
    Inputs:
        data_dir (Path or str): parent directory for FITS files
        limit_to (list of str): list of SNe, e.g. OSC, to limit the selection
    Output:
        fits_list (list): list of full FITS file paths
    """
    
    fits_list = [f for f in data_dir.glob('**/*.fits.gz')]
    # Limit to only SNe included in OSC
    if len(limit_to) > 0:
        fits_list = [f for f in fits_list if fname2sn(f)[0] in limit_to]
    return fits_list


def get_pre_post_obs(observations):
    """Limit sample to targets with observations before+after discovery."""

    pre = observations['epochs_pre_disc']
    post = observations['epochs_post_disc']
    pre_post_obs = observations[(pre > 0) & (post > 0)]
    return pre_post_obs


def plot(observations, show=False):
    """Plot histogram of the number of SNe with a given number of observations.
    Inputs:
        observations (DataFrame): all targets with GALEX observations
    """

    print('\nPlotting histogram of observation frequency...')
    bands = ['FUV', 'NUV']

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8,6.5),
            gridspec_kw={'hspace': 0.05})

    for ax, band in zip(axes, bands):
        df = observations[observations['band'] == band]
        all_epochs = df['total_epochs']
        pre_post_epochs = get_pre_post_obs(df)['total_epochs']

        bins = np.logspace(0, np.log10(np.max(all_epochs)), 11)
        color = COLORS[band]

        ax.hist(all_epochs, bins=bins, histtype='step', align='mid', lw=2, 
                color=color, label='all SNe (%s)' % all_epochs.shape[0])
        ax.hist(pre_post_epochs, bins=bins, histtype='bar', align='mid',
                label='before+after (%s)' % pre_post_epochs.shape[0], 
                rwidth=0.95, color=color)

        ax.set_title(band, x=0.08, y=0.8)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(tkr.ScalarFormatter())

        ax.legend()
        ax.label_outer()

    # Outside axis labels only
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, 
            right=False, which='both')
    plt.xlabel('Total number of epochs', labelpad=12)
    plt.ylabel('Number of SNe', labelpad=18)

    plt.savefig(Path('out/observations.pdf'), bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    print('Done!')


# def get_post_obs(fits_info):
#     """
#     Returns DataFrame of SNe with multiple observations post-discovery, but none
#     pre-discovery.
#     """

#     post = fits_info['Epochs Post-SN']
#     pre = fits_info['Epochs Pre-SN']
#     multi_post = fits_info[(post > 1) & (pre == 0)].reset_index(drop=True)
#     multi_post = multi_post.sort_values(by=['Name', 'Band'])
#     multi_post.set_index('Name', drop=True, inplace=True)
#     return multi_post


# def get_pre_post_obs(fits_info):
#     """
#     Returns DataFrame of SNe with at least one observation before and after
#     discovery.
#     """

#     post = fits_info['Epochs Post-SN']
#     pre = fits_info['Epochs Pre-SN']
#     both = fits_info[(post > 0) & (pre > 0)].reset_index(drop=True)
#     both = both.sort_values(by=['Name', 'Band']).set_index('Name', drop=True)
#     return both


# def compress_duplicates(fits_info):
#     """
#     Compressses fits_info down to one entry per SN (removing band-specific
#     information). Observation epochs are summed, and first/last/next epochs are
#     maximized.
#     Input:
#         fits_info (DataFrame): FITS file-specific information
#     Output:
#         sn_info (DataFrame): SN-specific information
#     """

#     duplicated = fits_info.groupby(['Name'])
#     sn_info = pd.DataFrame([], index=pd.Series(fits_info.index, name='name'))
#     old_cols = ['Disc. Date', 'R.A.', 'Dec.', 'Host Name']
#     new_cols = ['disc_date', 'galex_ra', 'galex_dec', 'osc_host']
#     sn_info[new_cols] = fits_info[old_cols].copy()
#     sn_info['epochs_total'] = duplicated['Total Epochs'].transform('sum')
#     sn_info['epochs_pre'] = duplicated['Epochs Pre-SN'].transform('sum')
#     sn_info['epochs_post'] = duplicated['Epochs Post-SN'].transform('sum')
#     sn_info['delta_t_first'] = duplicated['First Epoch'].transform('max')
#     sn_info['delta_t_last'] = duplicated['Last Epoch'].transform('max')
#     sn_info['delta_t_next'] = duplicated['Next Epoch'].transform('min')
#     sn_info = sn_info.loc[~sn_info.index.duplicated()]
#     return sn_info


# def write_quick_stats(fits_info, final_sample, sn_info, osc, file):
#     """
#     Writes quick statistics about sample to text file
#     Input:
#         fits_info (DataFrame): output from compile_fits
#         final_sample (DataFrame): output from get_pre_post_obs
#         sn_info (DataFrame): output from compress_duplicates
#         osc (DataFrame): Open Supernova Catalog reference info
#         file (Path or str): output file
#     """

#     print('Writing quick stats...')
#     sne = fits_info['Name'].drop_duplicates()
#     post_disc = get_post_obs(fits_info)
#     post_disc_sne = post_disc.index.drop_duplicates()
#     final_sne = final_sample.loc[~final_sample.index.duplicated()]
#     fuv = final_sample[final_sample['Band'] == 'FUV']
#     nuv = final_sample[final_sample['Band'] == 'NUV']
#     with open(file, 'w') as f:
#         f.write('Quick stats:\n')
#         f.write('\tnumber of reference SNe: %s\n' % len(osc))
#         f.write('\tnumber of SNe with GALEX data: %s\n' % len(sne))
#         f.write('\tnumber of SNe with observations only after discovery: %s\n' % len(post_disc_sne))
#         f.write('\tnumber of SNe with observations before and after discovery: %s\n' % len(final_sne))
#         f.write('\tfinal sample size: %s\n' % len(sn_info.index))
#         f.write('\tnumber of final SNe with FUV observations: %s\n' % len(fuv))
#         f.write('\tnumber of final SNe with NUV observations: %s\n' % len(nuv))


class GalexObservation:
    def __init__(self, path):
        """Import FITS data and header info."""

        # Get path, SN name and band
        self.path = Path(path)
        self.sn_name, self.band = fname2sn(self.path.name)
        # Import FITS file
        with fits.open(self.path) as hdu:
            self.header = hdu[0].header
            self.data = hdu[0].data
        # exposure times (array for single image is 2D)
        if self.header['NAXIS'] == 2:
            self.epochs = 1
            # single exposure time
            expts = [self.header['EXPTIME']]
            # t_mean is average of exposure start and end times
            tmeans = [(self.header['EXPEND'] + self.header['EXPSTART']) / 2]
        else:
            self.epochs = self.header['NAXIS3']
            expts = [self.header['EXPT'+str(i)] for i in range(self.epochs)]
            try:
                tmeans = [self.header['TMEAN'+str(i)] for i in range(self.epochs)]
            except KeyError:
                tmeans = [self.header['TSTAMP'+str(i)] for i in range(self.epochs)]                
        self.exp_times = np.array(expts)
        self.t_mean = Time(np.array(tmeans), format='gps')
        # world coordinate system
        self.wcs = WCS(self.header)
        # RA and Dec, given in degrees in the FITS header
        self.ra = Angle(str(self.header['CRVAL1'])+'d')
        self.dec = Angle(str(self.header['CRVAL2'])+'d')


    @classmethod
    def from_sn_name(self, sn_name, band, data_dir=DATA_DIR):
        """Generate instance from SN name and GALEX band (strings)."""

        fname = sn2fname(sn_name, band, suffix='.fits.gz')
        return GalexObservation(Path(data_dir) / Path(fname))


    def get_DataFrame(self, disc_date=None, date_fmt=None):
        """Return DataFrame of relevant FITS file info."""

        # General FITS info
        info = {'file': self.path.name,
                'sn_name': self.sn_name,
                'band': self.band,
                'ra': self.ra,
                'dec': self.dec,
                'total_epochs': self.epochs,
                't_mean_first': np.min(self.t_mean).iso.split(' ')[0],
                't_mean_last': np.max(self.t_mean).iso.split(' ')[0]
        }

        # Info related to discovery date
        if disc_date != None:
            self.compare_discovery(disc_date, date_fmt=date_fmt)
            info['disc_date'] = self.disc_date.iso.split(' ')[0]
            info['epochs_pre_disc'] = self.count_pre_disc
            info['epochs_post_disc'] = self.count_post_disc
            info['t_delta_first'] = np.round(np.min(self.t_delta))
            info['t_delta_last'] = np.round(np.max(self.t_delta))
            info['t_delta_next'] = np.round(self.min_post_disc)

        df = pd.DataFrame(info, index=[0])
        return df


    def compare_discovery(self, disc_date, date_fmt=None):
        """Count the number of epochs before and after SN discovery date."""

        # Convert discovery date to astropy.time.Time format if needed
        if type(disc_date) == Time:
            self.disc_date = disc_date
        else:
            self.disc_date = Time(disc_date, format=date_fmt)

        # Observation t_mean - disc_date
        self.t_delta = self.t_mean.mjd - self.disc_date.mjd
        # Count number of GALEX epochs before / after discovery
        self.count_pre_disc = len(self.t_delta[self.t_delta < 0])
        self.count_post_disc = len(self.t_delta[self.t_delta >= 0])
        # Soonest observation post-discovery, if any
        if self.count_post_disc > 0:
            self.min_post_disc = np.min(self.t_delta[self.t_delta >= 0])
        else:
            self.min_post_disc = np.nan


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Classify FITS images by ' + \
            'relative timing to SN discovery date.')
    parser.add_argument('-d', '--dir', type=Path, default=DATA_DIR,
            help='path to FITS data directory')
    parser.add_argument('-o', '--overwrite', action='store_true',
            help='re-generate FITS info file and overwrite existing')
    parser.add_argument('-r', '--reference', type=Path, default=OSC_FILE,
            help='SN reference info CSV')
    args = parser.parse_args()

    main(data_dir=args.dir, overwrite=args.overwrite, osc_file=args.reference)
