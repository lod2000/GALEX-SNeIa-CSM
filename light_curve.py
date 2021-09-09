import pandas as pd
import numpy as np
from pathlib import Path
import platform
import argparse
import matplotlib.pyplot as plt

from astropy.time import Time
from statsmodels.stats.weightstats import DescrStatsW
import astropy.units as u
from astropy.constants import c

from supernova import Supernova
from utils import *

# Default file and directory paths
DATA_DIR = Path('/mnt/d/GALEXdata_v10')     # Path to data directory
#LC_DIR = DATA_DIR / Path('LCs/')            # light curve data dir
# LC_DIR = Path('data')
LC_DIR = Path('historical_LCs')
REF_FILE = Path('ref/sn_info.csv') # SN info reference database
# REF_FILE = Path('out/nearby_historical.csv')

# GALEX spacecraft plate scale
PLATE_SCALE = 6 * u.arcsec / u.pix

# Default detection limits
SIGMA = [5, 3] # detection certainty
SIGMA_COUNT = [1, 3] # Number of points at corresponding sigma to detect

# Plot color palette
COLORS = {'FUV' : '#a37', 'NUV' : '#47a', # GALEX
          'UVW1': '#cb4', 'UVM2': '#283', 'UVW2': '#6ce', # Swift
          'F275W': '#e67', # Hubble
          'B': '#6ce', 'V': '#283', "r'": '#e67', "i'": '#888' # CfA
          }

# Other plot settings
BG_SPAN_ALPHA = 0.2 # background span transparency
BG_LINE_ALPHA = 0.7 # background line transparency
BG_SIGMA = 1 # background uncertainty
DT_MIN = -30 # Separation between background and SN data (days)
DET_SIGMA = 3 # detection threshold & nondetection upper limit
DETRAD_CUT = 0.6 # detector radius cut in degrees


def main(sn_name, make_plot=False, sigma=SIGMA, count=SIGMA_COUNT, tmax=4000, 
        pad=0, swift=False, cfa=False, legend_col=3, all_points=False):

    # sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
    sn_info = pd.read_csv(REF_FILE, index_col='name')
    sn = Supernova(sn_name, sn_info=sn_info, ref_file=REF_FILE)

    detect_cols = ['t_delta_rest', 'flux_hostsub', 'flux_hostsub_err', 
        'luminosity_hostsub', 'luminosity_hostsub_err']
    limit_cols = ['t_delta_rest', 'flux_limit', 'luminosity_limit']

    for band in ['FUV', 'NUV']:
        print('\n%s DATA' % band)
        try:
            lc = LightCurve.from_name(sn_name, band)
            print('Background: %s +/- %s erg/s/cm^2/AA' % (lc.bg, lc.bg_err_tot))
            print('Background obs: %s' % lc.bg_data.shape[0])

            # Display detections
            print('Detections:')
            detections = lc.detect_csm(sigma, count=count)
            if len(detections.index) == 0:
                print('None.')
            else:
                print(detections[detect_cols])

            # Display upper limits
            print('Upper limits:')
            upper_lims = pd.DataFrame([])
            upper_lims['t_delta_rest'] = lc('t_delta_rest')
            upper_lims['flux_limit'] = lc('flux_hostsub_err') * DET_SIGMA
            upper_lims['luminosity_limit'] = lc('luminosity_hostsub_err') * DET_SIGMA
            print(upper_lims[upper_lims['t_delta_rest'] > 0][limit_cols])
        except FileNotFoundError:
            print('No data available.')

    if make_plot:
        print('\nPlotting %s...' % sn.name)
        plot(sn, tmax=tmax, pad=pad, swift=swift, cfa=cfa, legend_col=legend_col, 
                all_points=all_points)


def plot(sn, tmax=4000, pad=0, swift=False, cfa=False, legend_col=3, show=True,
        all_points=False):
    """Plot light curves for both GALEX filters plus external data.
    Inputs:
        sn: Supernova object
        tmax: maximum x-axis value, rest frame days post-discovery
        pad: float, fractional shift for both axes to avoid legend overlap
        swift: bool, add Swift data if available
        cfa: bool, add CfA4 data if available
        legend_col: number of columns in legend
        all_points: plot measured fluxes for all points rather than limits
    """

    fig, ax = plt.subplots(figsize=(4, 2))
    # fig, ax = plt.subplots(figsize=(8, 4))

    # Plot Swift data data
    if swift:
        lc = import_swift(sn.name, sn.disc_date.mjd, sn.z)
        ax = plot_external(ax, sn, lc, ['UVW1', 'UVM2', 'UVW2'])

    # Plot CfA data
    if cfa:
        lc = import_cfa(sn)
        ax = plot_external(ax, sn, lc, ['B', 'V', "r'", "i'"], marker='s')

    # Import and plot GALEX data
    ymin = []
    bg_max = 0
    for band in ['FUV', 'NUV']:
        try:
            lc = LightCurve(sn, band)
            
            if lc.data['t_delta_rest'].iloc[0] < 0:
                # Pre-SN obs.
                before = lc.data[lc('t_delta_rest') <= DT_MIN]
                ymin.append(effective_wavelength(band).value * lc.bg / 1.5)
                bg_max = max(bg_max, lc.bg)
                plot_bg = True
            else:
                plot_bg = False

            ax = plot_lc(ax, lc, tmax, all_points, plot_bg)
        except FileNotFoundError:
            # No data for this channel
            continue

    # Add legend
    plt.legend(loc='best', ncol=legend_col, handletextpad=0.2, 
            handlelength=1.0, fontsize=6, borderaxespad=1.5, borderpad=0.5, columnspacing=0.8)

    # Adjust and label axes
    label_y = 1.05
    ax.set_xlabel('Time since discovery [rest-frame days]')
    ax.set_yscale('log')
    ax.tick_params(axis='y', which='major', left=False)
    ax.set_ylabel('$\lambda F_\lambda$ [erg s$^{-1}$ cm$^{-2}$]')
    if len(ymin) > 0:
        ax.set_ylim((np.min(ymin), None))

    # Adjust limits
    xlim = np.array(ax.get_xlim())
    xlim[1] += (pad / 5) * (xlim[1] - xlim[0])
    ax.set_xlim(xlim)
    ylim = np.array(ax.get_ylim())
    ylim[1] *= 10**pad
    ax.set_ylim(ylim)

    # In-plot labels
    # ax.text(xlim[0]+3, bg_max * 1.1,'host %sσ' % BG_SIGMA)

    # Twin axis with absolute luminosity
    luminosity_ax = ax.twinx()
    ylim_flux = np.array(ax.get_ylim())
    # Assume FUV for extinction; not that big a difference between the two
    ylim_luminosity = flux2luminosity(ylim_flux, 0, sn.dist, sn.dist_err,
            sn.z, sn.z_err, sn.a_v, 'FUV')[0].value
    luminosity_ax.set_yscale('log')
    luminosity_ax.set_ylim(ylim_luminosity)
    luminosity_ax.set_ylabel('$\lambda L_\lambda$ [erg s$^{-1}$]', rotation=270,
            labelpad=18)

    plt.tight_layout(pad=0.3)

    plt.savefig(Path('light_curves/%s.pdf' % sn.name), dpi=300)
    plt.savefig(Path('light_curves/%s.png' % sn.name), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


class LightCurve:
    def __init__(self, sn, band, data_dir=LC_DIR, **kwargs):
        """Import light curve data and calculate background & luminosity."""

        self.sn_name = sn.name
        self.band = band
        self.fname = sn2fname(sn.name, band, suffix='.csv')
        data = import_light_curve(data_dir / self.fname, **kwargs)

        # Convert time to MJD, MJD-disc, and rest frame
        data['t_mean_mjd'] = Time(data['t_mean'], format='gps').mjd
        data['t_delta'] = (data['t_mean_mjd'] - sn.disc_date.mjd)
        data['t_delta_rest'] = 1 / (1 + sn.z) * data['t_delta']
        
        # Background & systematic error
        self.bg_data = data[data['t_delta_rest'] < DT_MIN]
        self.bg, self.bg_err, self.sys_err = get_background(data, self.band)
        self.bg_err_tot = np.sqrt(self.bg_err**2 + self.sys_err**2)
        self.background = self.bg
        self.background_err = self.bg_err

        # Add systematic error
        data['flux_bgsub_err_total'] = np.sqrt(data['flux_bgsub_err']**2 + 
                self.sys_err**2)
        # Subtract host background
        data['flux_hostsub'] = data['flux_bgsub'] - self.bg
        data['flux_hostsub_err'] = np.sqrt(data['flux_bgsub_err_total']**2 +
                self.bg_err**2)

        # Calculate luminosity
        data = add_luminosity(data, sn, band)
        self.data = data


    def __call__(self, col='', tmin=-4000, tmax=4000):
        """Return data column.
        Inputs:
            col: column label
            tmin: minimum t_delta_rest value
            tmax: maximum t_delta_rest value
        """

        data_slice = self.data.copy()
        # Limit to tmin, tmax bounds
        data_slice = data_slice[data_slice['t_delta_rest'] >= tmin]
        data_slice = data_slice[data_slice['t_delta_rest'] < tmax]

        if len(col) > 0:
            data_slice = data_slice[col]

        return data_slice


    @classmethod
    def from_file(self, fname, **kwargs):
        """Initialize LightCurve from file name."""
        sn_name, self.band = fname2sn(fname)
        sn = Supernova(sn_name, ref_file=REF_FILE)
        return LightCurve(sn, self.band, **kwargs)
    

    @classmethod
    def from_name(self, sn_name, band, sn_info=[], **kwargs):
        """Initialize LightCurve from SN name rather than Supernova instance."""
        sn = Supernova(sn_name, sn_info=sn_info, ref_file=REF_FILE)
        return LightCurve(sn, band, **kwargs)


    def to_hz(self):
        """Convert fluxes from per unit wavelength to per unit frequency."""

        data = self.data.copy()
        # List of columns to convert
        convert_cols = [col for col in data.columns if 'flux' in col 
                or 'luminosity' in col]
        for col in convert_cols:
            hz_col = col+'_hz'
            data[hz_col] = wavelength2freq(data[col], effective_wavelength(self.band))
        self.data = data
        return data


    def inject(self, model_data, data_col):
        """Inject model data into data from given column."""

        injection = self.data[data_col] + model_data
        self.data['%s_injected' % data_col] = injection
        return injection


    def detect_csm(self, sigma, count=[1], dt_min=DT_MIN, time_col='t_delta_rest', 
            data_col='flux_hostsub', err_col='flux_hostsub_err'):
        """Detect CSM. For multiple confidence tiers, len(sigma) = len(count).
        Inputs:
            time_col, data_col, err_col: column labels
            sigma: detection confidence level, int or list
            count: number of data points above sigma to count as detection, list
            dt_min: minimum number of days post-discovery
        Output:
            detections: list of indices of detected observations
        """

        # separate SN data from host background data
        all_time = self.data[time_col]
        time = all_time[all_time > dt_min]
        data = self.data[data_col][time.index]
        err = self.data[err_col][time.index]
        # confidence level of data
        conf = data / err

        if type(sigma) == int:
            sigma = [sigma]

        # Tiered detection: requires N points above X sigma or M points above Y sigma
        detections = []
        for s, c in zip(sigma, count):
            detected = time[conf >= s].index.to_list()
            if len(detected) >= c:
                detections += detected
        
        # Remove duplicates, sort, and return indices of detections
        self.detections = sorted(list(dict.fromkeys(detections)))
        return self.data.loc[self.detections]


def add_luminosity(data, sn, band):
    """Calculate luminosity based on flux and add columns to DataFrame."""

    data['luminosity'], data['luminosity_err'] = flux2luminosity(
            data['flux_bgsub'], data['flux_bgsub_err_total'], sn.dist, sn.dist_err, 
            sn.z, sn.z_err, sn.a_v, band)
    data['luminosity_hostsub'], data['luminosity_hostsub_err'] = flux2luminosity(
            data['flux_hostsub'], data['flux_hostsub_err'], sn.dist, sn.dist_err, 
            sn.z, sn.z_err, sn.a_v, band)
    return data


def get_background(data, band, dt_min=-30):
    """Calculate the host background, background scatter, and systematic error.
    Inputs:
        data: light curve DataFrame
        band: 'FUV' or 'NUV'
        dt_min: dividing line between "background" and "supernova" in days post-discovery
    Outputs:
        bg: background flux density per unit wavelength
        bg_err: background statistical error, same units
        sys_err: systematic error, same units
    """

    bg_data = data[data['t_delta_rest'] < dt_min]
    flux = np.array(bg_data['flux_bgsub'])
    flux_err = np.array(bg_data['flux_bgsub_err'])

    # For many background points, calculate the systematic error using a
    # reduced chi-squared fit
    if len(bg_data.index) > 4:
        bg, bg_err, sys_err = fit_rcs(flux, flux_err)

    # For few background points, use the polynomial fit of |MCAT - gAper| errors
    # based on the original gAperture (non-background-subtracted) magnitudes
    else:
        if len(bg_data.index) > 1:
            # Determine background from weighted average of data before discovery
            weighted_stats = DescrStatsW(flux, weights=1/flux_err**2, ddof=0)
            bg = weighted_stats.mean
            bg_err = weighted_stats.std
            ann_flux = np.average(bg_data['flux'] - bg_data['flux_bgsub'], 
                    weights=1/flux_err**2)
        else:
            # Otherwise, just use the first point
            bg = data['flux_bgsub'].iloc[0]
            bg_err = data['flux_bgsub_err'].iloc[0]
            ann_flux = data['flux'].iloc[0] - data['flux_bgsub'].iloc[0]

        # Use background if it's positive, or annulus flux if it's not
        if bg > 0:
            sys_err = fit_sys_err(bg, band)
        else:
            sys_err = fit_sys_err(ann_flux, band)

    return bg, bg_err, sys_err


def fit_rcs(data, err, init=2, step=0.1):
    """Fit systematic error to get reduced chi-squared value close to 1."""

    # Initialize reduced chi-square, sys error values
    rcs = init
    sys_err_step = np.nanmean(err) * step
    sys_err = -sys_err_step

    # Reduce RCS to 1 by adding systematic error in quadrature
    while rcs > 1:
        # Increase systematic error for next iteration
        sys_err += sys_err_step
        # Combine statistical and systematic error
        new_err = np.sqrt(err ** 2 + sys_err ** 2)
        # Determine background from weighted average of data before discovery
        weighted_stats = DescrStatsW(data, weights=1/new_err**2, ddof=0)
        mean = weighted_stats.mean
        # Reduced chi squared test of data vs background
        rcs = redchisquare(data, np.full(data.size, mean), new_err, n=0)

    stat_err = weighted_stats.std
    return mean, stat_err, sys_err


def redchisquare(data, model, sd, n=0):
    """Reduced chi-squared statistic."""
    
    chisq = np.sum(((data-model)/sd)**2)
    return chisq/(len(data)-1-n)


def fit_sys_err(bg_flux, band):
    """Calculate systematic error from gAperture polynomial fit."""

    bg_mag = flux2mag(bg_flux, band)
    # Calculate systematic error from gAperture photometric error
    sys_err_mag = gAper_sys_err(bg_mag.value, band)
    # Convert mag error to SNR
    snr = 1 / (10 ** (sys_err_mag / 2.5) - 1)
    sys_err = bg_flux / snr
    return sys_err


def gAper_sys_err(mag, band):
    """Calculate the systematic error from gAperture at a given magnitude
    based on Michael's polynomial fits.
    """

    # coeffs = {
    #     'FUV': [4.07675572e-04, -1.98866713e-02, 3.24293442e-01, -1.75098239e+00],
    #     'NUV': [3.38514034e-05, -2.88685479e-03, 9.88349458e-02, -1.69681516e+00,
    #             1.45956431e+01, -5.02610071e+01]
    # }
    # fit = np.poly1d(coeffs[band])
    # return fit(mag.value)

    # fit coeffs
    A = {'NUV': 4.94e-7, 'FUV': 4.78e-4}
    B = {'NUV': 6.17, 'FUV': 2.60}
    return A[band] * (mag-14)**B[band]


def import_light_curve(lc_file, detrad_cut=DETRAD_CUT, manual_cuts=[]):
    """Import light curve file for specified SN and band, cutting points 
    with bad flags or sources outside detector radius.
    Inputs:
        lc_file: path to light curve csv
        detrad_cut: maximum source distance from detector center, in degrees
        manual_cuts: list of indices manually identified as bad data"""

    # Read light curve data
    data = pd.read_csv(lc_file, dtype={'flags':int})

    # Weed out bad flags
    fatal_flags = (1 | 2 | 4 | 16 | 64 | 128 | 512)
    data = data[data['flags'] & fatal_flags == 0]

    # Cut sources outside detector radius
    detrad_cut_px = (detrad_cut * u.deg).to('arcsec') / PLATE_SCALE
    data = data[data['detrad'] < detrad_cut_px.value]

    # Cut unphysical flux values
    data = data[np.abs(data['flux_bgsub']) < 1]

    # Cut data with background counts less than 0
    data = data[data['bg_counts'] >= 0]

    # Cut data with background much higher than average (washed-out fields)
    data.insert(29, 'bg_cps', data['bg_counts'] / data['exptime'])
    bg_median = np.median(data['bg_cps'])
    data = data[data['bg_cps'] < 3 * bg_median]

    # Add manual cuts (e.g. previously identified as a ghost image)
    # manual_cuts = pd.read_csv(Path('ref/manual_cuts.csv'))
    # to_remove = manual_cuts[(manual_cuts['name'] == self.sn_name) & (manual_cuts['band'] == self.band)]['index']
    data = data[~data.index.isin(manual_cuts)]

    # Raise error if all data has been removed
    if len(data.index) == 0:
        raise pd.errors.EmptyDataError

    return data


################################################################################
## Plotting
################################################################################


def plot_lc(ax, lc, tmax, all_points=False, plot_bg=True):

    color = COLORS[lc.band]
    fill = {'FUV': 'w', 'NUV': color}[lc.band]
    edge = {'FUV': color, 'NUV': 'k'}[lc.band]

    lambda_eff = effective_wavelength(lc.band).value

    # Data column labels
    time_col = 't_delta_rest'
    data_col = 'flux_bgsub'
    err_col = 'flux_bgsub_err_total'

    if plot_bg:
        # Pre-SN obs.
        before = lc.data[lc(time_col) <= DT_MIN]
    
        # Plot background average of epochs before discovery
        ax.axhline(lambda_eff * lc.bg, 0, 1, color=color, alpha=BG_LINE_ALPHA, 
                linestyle='--', linewidth=1
        )
        # 1-sigma range
        bg_err = np.sqrt(lc.bg_err**2 + lc.sys_err**2)
        ax.axhspan(ymin=lambda_eff * (lc.bg - BG_SIGMA * bg_err), color=color,
                ymax=lambda_eff * (lc.bg + BG_SIGMA * bg_err), alpha=BG_SPAN_ALPHA
        )

    # Plot observed fluxes after discovery: detections
    # after = lc.data[(lc(time_col) > DT_MIN) & (lc(time_col) < tmax)]
    after = lc.data[lc(time_col, tmax=tmax) > DT_MIN]
    # if 'all_points', plot points with error bars for all observations
    # otherwise, just plot detections with limits for everything else
    if all_points:
        points = after
        label = lc.band
    else:
        points = lc.detect_csm(DET_SIGMA, count=[1], dt_min=DT_MIN)
        label = '%s det.' % lc.band
    if len(points.index) > 0:
        ax.errorbar(points[time_col], lambda_eff * points[data_col], 
                yerr=lambda_eff * points[err_col], linestyle='none', ecolor=edge,
                marker='o', ms=4, elinewidth=1, c=fill, mec=edge, mew=1,
                label=label
        )

    # Plot nondetection limits
    if not all_points:
        nondetections = after.drop(lc.detections)
        if len(nondetections.index > 0):
            ax.scatter(nondetections[time_col], 
                    lambda_eff * (nondetections['flux_hostsub_err']*DET_SIGMA + lc.bg), 
                    marker='v', s=16, color=fill, edgecolors=edge, lw=1,
                    label='%s %sσ limit' % (lc.band, DET_SIGMA)
            )

    return ax


def plot_external(ax, sn, lc, bands, marker='D'):
    """Plot data from Swift or CfA."""

    for band in bands:
        data = lc[lc['Filter'] == band]
        lambda_eff = effective_wavelength(band).value
        ax.errorbar(data['t_delta_rest'], lambda_eff * data['flux'], linestyle='none',
                yerr=lambda_eff * data['flux_err'], marker=marker, ms=2.5, label=band,
                elinewidth=0.5, markeredgecolor=COLORS[band], 
                markerfacecolor='white', ecolor=COLORS[band], mew=0.5,
                rasterized=True)

    return ax


def import_swift(sn_name, disc_date_mjd, z):
    """Import Swift light curve data."""

    # Read CSV
    lc = pd.read_csv(Path('ref/%s_uvotB15.1.dat' % sn_name), sep='\s+',
            names=['Filter', 'MJD', 'Mag', 'MagErr', '3SigMagLim', '0.98SatLim', 
            'Rate', 'RateErr', 'Ap', 'Frametime', 'Exp', 'Telapse'], comment='#')
    # Remove limits
    lc = lc[pd.notna(lc['Mag'])]

    # Add days relative to discovery date
    lc['t_delta'] = lc['MJD'] - disc_date_mjd
    # Correct epoch for stretch factor
    lc['t_delta_rest'] = 1 / (1 + z) * lc['t_delta']

    # Convert CPS to flux
    # From Breeveld et al 2011, Table 9
    flux_factor   = {'V': 2.614e-16, 'B': 1.472e-16, 'U': 1.63e-16, 
                     'UVW1': 4.3e-16, 'UVM2': 7.5e-16, 'UVW2': 6.0e-16}
    flux_error    = {'V': 8.7e-19, 'B': 5.7e-19, 'U': 2.5e-18,
                     'UVW1': 2.1e-17, 'UVM2': 1.1e-16, 'UVW2': 6.4e-17}
    c = np.vectorize(flux_factor.get)(lc['Filter'])
    c_err = np.vectorize(flux_error.get)(lc['Filter'])
    lc['flux'] = lc['Rate'] * c
    lc['flux_err'] = lc['flux'] * np.sqrt((lc['RateErr']/lc['Rate'])**2+(c_err/c)**2)

    return lc


def import_cfa(sn):
    """Import light curve data from CfA data release."""

    lc = pd.read_csv(Path('ref/CfA4_lc.dat'), sep='\s+',
            names=['SN', 'Filter', 'MJD', 'Num', 'sigPip', 'sigPhot', 'mag', 
            'e_mag'], skiprows=30)
    lc = lc[lc['SN'] == sn.name.replace('SN', '')]

    # Add days relative to discovery date
    lc['t_delta'] = lc['MJD'] - sn.disc_date.mjd
    # Correct epoch for stretch factor
    lc['t_delta_rest'] = 1 / (1 + sn.z) * lc['t_delta']

    # Convert to fluxes
    # lc['AbsMag'] = lc['mag'] - 5 * np.log10(sn.dist.value) + 5
    zero_point = {'B': 6.32e-9, 'V': 3.63e-9, "r'": 2.83e-9, "i'": 1.85e-9}
    lc['flux'] = np.vectorize(zero_point.get)(lc['Filter']) * 10 **(-2/5 * lc['mag'])
    lc['flux_err'] = lc['flux'] * 2/5 * np.log(10) * lc['e_mag']

    return lc


################################################################################
## Conversions
################################################################################

"""
All conversion functions are designed to accept either float or NumPy array-like
inputs, as long as all input arrays have the same shape. The 'band' should be a
string or array of strings, matching one of the keys of the below conversion
factors. Inputs should be Astropy quantities with units.
"""

cgs_flux = u.erg / (u.s * u.cm**2 * u.AA)

# GALEX conversions from https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html

def zero_point(band):
    """Return zero-point magnitude for given GALEX band."""
    vals = {'FUV': 18.82, 'NUV': 20.08}
    return np.vectorize(vals.get)(band) * u.mag(u.s/u.ct) + 0 * u.ABmag


def flux_factor(band):
    """Return cps to flux factor for given GALEX band."""
    vals = {'FUV': 1.4e-15, 'NUV': 2.06e-16}
    return np.vectorize(vals.get)(band) * cgs_flux / (u.ct/u.s)


def extinction(band):
    """Return Milky Way extinction magnitude for given GALEX band (Bianchi)."""
    vals = {'FUV': 8.06, 'NUV': 7.95, 'U': 4.72, 'B': 4.02, 'V': 3.08}
    return np.vectorize(vals.get)(band) * u.mag


def effective_wavelength(band):
    """Return effective wavelength of given GALEX/Swift/HST/CfA4 band."""
    vals = {'FUV': 1549, 'NUV': 2304.7, 'F275W': 2714.65, 
            'UVW2': 2085.7, 'UVM2': 2245.7, 'UVW1': 2684.1, 
            'B': 4450, 'V': 5510, "r'": 6220, "i'": 7630}
    return np.vectorize(vals.get)(band) * u.AA


def cps2mag(cps, band):
    """Convert GALEX counts per second to AB magnitudes."""
    return u.Magnitude(cps) + zero_point(band)


def convert_extinction(ext_in, band_out, band_in='V'):
    """Convert foreground extinction in magnitudes from one band to another 
    (default: V-band)."""
    E = ext_in / extinction(band_in) # Typically E(B-V)
    ext_out = E * extinction(band_out) # Typically A_FUV or A_NUV
    return ext_out


def flux2mag(flux, band):
    """Convert GALEX fluxes to AB magnitudes."""
    cps = flux / flux_factor(band)
    return cps2mag(cps, band)


def flux2luminosity(flux, flux_err, dist, dist_err, z, z_err, a_v, band):
    """Convert measured flux to luminosity based on distance with error.
    Inputs:
        flux (Array-like): measured fluxes
        flux_err (Array-like): measured flux error
        dist (float): distance in Mpc
        dist_err (float): distance error in Mpc
    Outputs:
        absolute luminosity (Array), luminosity error (Array)
    """

    # Calculate luminosity at distance
    luminosity = 4 * np.pi * dist.to('cm')**2 * flux
    # Correct for redshift
    luminosity *= (1 + z) ** 3
    # Correct for extinction
    a_band = convert_extinction(a_v, band, band_in='V')
    luminosity *= 10 ** (0.4 * a_band.value)

    # Redshift correction error
    z_corr_err = np.nan_to_num(3 * z_err / (1+z))
    # Sum of flux, distance, and redshift errors
    err = np.abs(luminosity) * np.sqrt((2*dist_err/dist)**2 + 
            (flux_err/flux)**2 + z_corr_err**2)
    return luminosity, err


def freq2wavelength(flux, wavelength):
    """Convert flux density from per unit frequency to per unit wavelength."""
    
    return flux * c.to('AA/s') / wavelength**2


def wavelength2freq(flux, wavelength):
    """Convert flux density from per unit wavelength to per unit frequency."""

    return flux * wavelength**2 / c.to('AA/s')


if __name__=='__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run CSM detection algorithm' +
            ' and/or make plots for GALEX SN Ia light curves.')
    parser.add_argument('sn', type=str, help='supernova name')

    # Detection arguments
    parser.add_argument('--sigma', type=int, nargs='+', default=SIGMA, 
            help='Detection confidence level (multiple for tiered detections)')
    parser.add_argument('--sigcount', type=int, nargs='+', default=SIGMA_COUNT,
            help='Number of points at corresponding sigma to count as detection')

    # Plotting arguments
    parser.add_argument('-p', '--plot', action='store_true', 
            help='generate publication-quality light curve plots')
    parser.add_argument('--pad', type=float, default=0.,
            help='extra padding for legend at the top-right')
    parser.add_argument('--tmax', type=float, default=4000,
            help='maximum number of days after discovery to plot')
    parser.add_argument('--swift', action='store_true', 
            help='plot Swift data if available')
    parser.add_argument('--cfa', action='store_true', 
            help='plot CfA4 data if available')
    parser.add_argument('--lcol', type=int, default=3, 
            help='number of columns in legend')
    parser.add_argument('--allpoints', action='store_true', 
            help='plot all points instead of limits')

    args = parser.parse_args()

    main(args.sn, make_plot=args.plot, sigma=args.sigma, 
            count=args.sigcount, tmax=args.tmax, pad=args.pad, swift=args.swift, 
            cfa=args.cfa, legend_col=args.lcol, all_points=args.allpoints)
