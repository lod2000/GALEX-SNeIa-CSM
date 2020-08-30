import pandas as pd
import numpy as np
from pathlib import Path

from astropy.time import Time
from statsmodels.stats.weightstats import DescrStatsW
import astropy.units as u
from astropy.constants import c

from supernova import Supernova
from utils import *

# Default file and directory paths
DATA_DIR = Path('/mnt/d/GALEXdata_v10')     # Path to data directory
LC_DIR = DATA_DIR / Path('LCs/')            # light curve data dir
# LC_DIR = Path('test/')

# GALEX spacecraft plate scale
PLATE_SCALE = 6 * u.arcsec / u.pix

def main():
    lc = LightCurve.from_name('SN2007on', 'NUV')


class LightCurve:
    def __init__(self, sn, band, **kwargs):
        """Import light curve data and calculate background & luminosity."""

        self.sn_name = sn.name
        self.band = band
        self.fname = sn2fname(sn.name, band, suffix='.csv')
        data = import_light_curve(LC_DIR / self.fname, **kwargs)

        # Convert time to MJD, MJD-disc, and rest frame
        data['t_mean_mjd'] = Time(data['t_mean'], format='gps').mjd
        data['t_delta'] = (data['t_mean_mjd'] - sn.disc_date.mjd)
        data['t_delta_rest'] = 1 / (1 + sn.z) * data['t_delta']
        
        # Background & systematic error
        self.bg, self.bg_err, self.sys_err = get_background(data, self.band)
        self.background = self.bg
        self.background_err = self.bg_err

        # Add systematic error
        data['flux_bgsub_err_total'] = np.sqrt(data['flux_bgsub_err']**2 + 
                self.sys_err**2)
        # Subtract host background
        data['flux_hostsub'] = data['flux_bgsub'] - self.bg
        data['flux_hostsub_err'] = np.sqrt(data['flux_bgsub_err_total']**2 +
                self.bg_err**2)
        # Detection confidence level
        data['sigma'] = data['flux_hostsub'] / data['flux_hostsub_err']

        # Calculate luminosity
        data = add_luminosity(data, sn, band)
        self.data = data


    def __call__(self, col):
        return self.data[col]


    @classmethod
    def from_file(self, fname, **kwargs):
        """Initialize LightCurve from file name."""
        sn_name, self.band = fname2sn(fname)
        sn = Supernova(sn_name)
        return LightCurve(sn, self.band, **kwargs)
    

    @classmethod
    def from_name(self, sn_name, band, sn_info=[], **kwargs):
        """Initialize LightCurve from SN name rather than Supernova instance."""
        sn = Supernova(sn_name, sn_info=sn_info)
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
        return data


    def detect(self, sigma, count=[1], dt_min=-30):
        """Detect CSM. For multiple confidence tiers, len(sigma) = len(count).
        Inputs:
            sigma: detection confidence level, int or list
            count: number of data points above sigma to count as detection, list
            dt_min: cutoff between background and SN data, days post-disc.
        """

        # separate SN data from host background data
        sn_data = self.data[self.data['t_delta_rest'] > dt_min]

        if type(sigma) == int:
            sigma = [sigma]

        # Tiered detection: requires N points above X sigma or M points above Y sigma
        detections = []
        for s, c in zip(sigma, count):
            detected = sn_data[sn_data['sigma'] >= s]
            if len(detected.index) >= c:
                detections.append(detected)
        
        detections = pd.concat(detections).sort_index().drop_duplicates()
        return detections


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

    bg_data = data[data['t_delta'] < dt_min]
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
            bg = lc['flux_bgsub'].iloc[0]
            bg_err = lc['flux_bgsub_err'].iloc[0]
            ann_flux = lc['flux'].iloc[0] - lc['flux_bgsub'].iloc[0]

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
    sys_err_mag = gAper_sys_err(bg_mag, band)
    # Convert mag error to SNR
    snr = 1 / (10 ** (sys_err_mag / 2.5) - 1)
    sys_err = bg_flux / snr
    return sys_err


def gAper_sys_err(mag, band):
    """Calculate the systematic error from gAperture at a given magnitude
    based on Michael's polynomial fits
    """

    coeffs = {
        'FUV': [4.07675572e-04, -1.98866713e-02, 3.24293442e-01, -1.75098239e+00],
        'NUV': [3.38514034e-05, -2.88685479e-03, 9.88349458e-02, -1.69681516e+00,
                1.45956431e+01, -5.02610071e+01]
    }
    fit = np.poly1d(coeffs[band])
    return fit(mag)


def import_light_curve(lc_file, detrad_cut=0.55, manual_cuts=[]):
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
    """Return effective wavelength of given GALEX band."""
    vals = {'FUV': 1549, 'NUV': 2304.7}
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
    main()
