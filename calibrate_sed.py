#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
import astropy.constants as const
import astropy.units as u
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from CSMmodel import CSMmodel

# common unit macros
f_nu_units = u.erg * u.s**-1 * u.cm**-2 * u.hertz**-1
f_lambda_units = u.erg * u.s**-1 * u.cm**-2 * u.angstrom**-1
L_nu_units = u.erg * u.s**-1 * u.hertz**-1
L_lambda_units = u.erg * u.s**-1 * u.angstrom**-1

# defined constants
H0 = 70 * u.km * u.s**-1 * u.Mpc**-1

# Stats for SN 2015cp in F275W (Graham+ 2019)
SN2015cp = {'L_nu': 7.6e25 * L_nu_units,
            'L_lambda': 3.1e37 * L_lambda_units,
            'z': 0.0413,
            'dist': 167 * u.Mpc
            }

# filter effective wavelength
lambda_eff = {'FUV': 1549. * u.angstrom, 
              'NUV': 2304.7 * u.angstrom, 
              'F275W': 2714.65 * u.angstrom
              }
# filter effective width
w_eff = {'FUV': 265.57 * u.angstrom, 
         'NUV': 768.31 * u.angstrom, 
         'F275W': 416.66 * u.angstrom
         }

# flux with 0 color in AB system (flat in f_nu)
zero_point_nu = 3.63e-20 * f_nu_units

# Plot color palette
colors = {'FUV' : '#a37', 'NUV' : '#47a', 'F275W': '#e67'}
styles = {'FUV': '-', 'NUV': '--', 'F275W': ':'}

# initialize CSM emission models
line_model = CSMmodel(0, 250, 0.3, scale=1, model='Chev94')
flat_model = CSMmodel(0, 250, 0.3, scale=1, model='flat')

def main(dz=0.001, plot=True, plot_dz=0.01, table=False, sed='Chev94'):
    """
    Generate lookup table and plots for SED calibration.

    Parameters
    ----------
    dz : float, optional
        Step between redshift values for lookup table. The default is 0.001.
    plot : bool, optional
        Generate plots. The default is True.
    plot_dz : float, optional
        Step between redshift values for plot. The default is 0.01.
    table : bool
        Generate calibration lookup tables for each band. The default is False.
    sed : str
        Underlying CSM emission model, either 'Chev94' or 'flat'. The default
        is 'Chev94'.

    """
    # Convert HST luminosity of SN 2015cp to flux
    SN2015cp['f_lambda'] = luminosity2flux(SN2015cp['L_lambda'], SN2015cp['z'], 
                                           SN2015cp['dist'])
    # ratio of AB zero point mean flux density at z_15cp to flux of 15cp
    fratio_15cp = zero_point_mfd(SN2015cp['z'], 'F275W') / SN2015cp['f_lambda']
    
    # Generate calibration lookup table
    if table:
        print('Generating calibration lookup table for...')
        zarr = np.arange(0., 0.5+dz, dz)
        calib_lookup = pd.DataFrame([], index=pd.Series(zarr, name='z'))
        for band in ['FUV', 'NUV', 'F275W']:
            print(band)
            calib_lookup[band] = gen_calibration(zarr, band)
        calib_lookup.to_csv(Path('ref/sed_calibration.csv'))
        print('Complete!')
    
    if plot:
        print('Plotting SED flux comparison...')
        # Plot mean flux density of AB zero point and line-emission model
        zarr = np.arange(0., 0.5+plot_dz, plot_dz)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        with quantity_support():
            for band in ['FUV', 'NUV', 'F275W']:
                # AB zeropoint
                ax.plot(zarr, zero_point_mfd(zarr, band) / fratio_15cp, 
                        color=colors[band], linestyle=':')
                # Chev94 line-emission model
                larr = np.array([line_model(0, z)[band] for z in zarr]) 
                larr *= L_lambda_units
                farr = luminosity2flux(larr, zarr, dist=SN2015cp['dist'])
                ax.plot(zarr, farr, color=colors[band], label=band)
            # flat CSM model (same for all bands)
            larr = np.array([flat_model(0, z)[band] for z in zarr]) 
            larr *= L_lambda_units
            farr = luminosity2flux(larr, zarr, dist=SN2015cp['dist'])
            ax.plot(zarr, farr, color='k', linestyle='--')
        ax.set_xlabel('Redshift')
        ax.set_ylabel(f'Mean flux density [{farr.unit:latex_inline}]')
        ax.set_yscale('log')
        # Add marker for SN 2015cp redshift
        ax.axvline(SN2015cp['z'], 0, 1, c='gray', zorder=0, linewidth=1)
        ax.text(SN2015cp['z'], 9e-17, '15cp', fontsize=9, va='bottom', ha='center')
        plt.tight_layout(pad=0.1)
        plt.subplots_adjust(top=0.86)
        # fig.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
        # Custom second legend
        solid_line = mlines.Line2D([], [], color='k', linestyle='-')
        dashed_line = mlines.Line2D([], [], color='k', linestyle='--')
        dotted_line = mlines.Line2D([], [], color='k', linestyle=':')
        FUV_line = mlines.Line2D([], [], color=colors['FUV'], linestyle='-')
        NUV_line = mlines.Line2D([], [], color=colors['NUV'], linestyle='-')
        F275W_line = mlines.Line2D([], [], color=colors['F275W'], linestyle='-')
        fig.legend([FUV_line, solid_line, NUV_line, dashed_line, F275W_line, dotted_line], 
                   ['FUV', 'Line-emission model', 'NUV', 'Flat spectrum model', 'F275W', 'Scaled AB zero point'],
                   loc='upper right', ncol=3, bbox_to_anchor=(1., 1.),
                   handletextpad=0.5, handlelength=1., borderpad=0.5, 
                   borderaxespad=0.05)
        plt.savefig(Path('out/sed_flux_comparison.png'), dpi=300)
        plt.savefig(Path('out/sed_flux_comparison.pdf'), dpi=300)
        plt.show()
    
        # Plot calibration of each band
        print('Plotting calibration by redshift...')
        fig, ax = plt.subplots()
        for band in ['FUV', 'NUV', 'F275W']:
            ax.plot(zarr, gen_calibration(zarr, band, sed=sed), 
                    label=band, color=colors[band], linestyle=styles[band])
        ax.set_xlabel('Redshift')
        ax.set_ylabel('$\\bar F_{\lambda,\\rm{line}}/\\bar F_{\lambda,\\rm{AB}}$')
        ax.set_yscale('log')
        # Add horizontal marker at unity
        ax.axhline(1, 0, 1, c='gray', zorder=0, linewidth=1)
        # Add marker for SN 2015cp redshift
        ax.axvline(SN2015cp['z'], 0, 1, c='gray', zorder=0, linewidth=1)
        ax.text(SN2015cp['z'], 5., '15cp', fontsize=9, va='bottom', ha='center')
        plt.tight_layout(pad=0.1)
        plt.subplots_adjust(top=0.86)
        fig.legend(loc='lower right', ncol=3, bbox_to_anchor=(1., 0.88),
                   handletextpad=0.5, handlelength=1., borderpad=0.5, 
                   fontsize=8, borderaxespad=0.05)
        plt.savefig(Path('out/sed_calibration_%s.png' % sed), dpi=300)
        plt.savefig(Path('out/sed_calibration_%s.pdf' % sed), dpi=300)
        plt.show()
        print('Done!')


def gen_calibration(z, band, sed='Chev94'):
    """
    Ratio of the line-emission flux density to the scaled AB zero point flux.

    Parameters
    ----------
    z : float or numpy.ndarray
        Redshift of source, which scales the flux by (1+z)^-3.
    band : str
        Filter, one of 'FUV', 'NUV', or 'F275W'.
    sed : str
        Underlying CSM emission model, either 'Chev94' or 'flat'. Default is
        'Chev94'.

    Returns
    -------
    calibration : float
        Scaled flux ratio.

    """
    # Convert HST luminosity of SN 2015cp to flux
    SN2015cp['f_lambda'] = luminosity2flux(SN2015cp['L_lambda'], SN2015cp['z'], 
                                           SN2015cp['dist'])
    # ratio of AB zero point mean flux density at z_15cp to flux of 15cp
    fratio_15cp = zero_point_mfd(SN2015cp['z'], 'F275W') / SN2015cp['f_lambda']
    calibration = csm_model_flux(z, dist=SN2015cp['dist'], band=band, sed=sed)
    calibration /= zero_point_mfd(z, band) / fratio_15cp
    return calibration.value


def zero_point_mfd(z, band):
    """
    Calculate mean flux density of the AB zero point flux.

    Parameters
    ----------
    z : float or numpy.ndarray
        Redshift of source, which scales the flux by (1+z)^-3.
    band : str
        Filter, one of 'FUV', 'NUV', or 'F275W'.

    Returns
    -------
    mfd : astropy.units.quantity.Quantity
        Mean flux density per unit wavelength of the AB zero point.

    """    
    wavelength, transmission, effective_area = import_response_table(band)
    # convert AB zero point flux from frequency space to wavelength space
    zero_point_lambda = lambda l: zero_point_nu * const.c.to('AA Hz') * l**-2
    # calculate mean flux density of the AB zero point in the given band
    mfd0 = mean_flux_density(zero_point_lambda, transmission, wavelength) 
    # scale mean flux density by redshift factor
    mfd = mfd0 * (1+z)**-3
    return mfd


def mean_flux_density(source_flux, throughput, wavelength):
    """
    Integrate a source flux distribution over a bandpass filter.
    
    Calculate the mean flux density to be observed given an assumed
    intrinsic source flux distribution and passband throughput over a range
    of wavelengths.
    
    Parameters
    ----------
    source_flux : function 
        Source flux distribution as a function of wavelength.
    throughput : numpy.ndarray
        Throughput of passband filter.
    wavelength : astropy.units.quantity.Quantity or numpy.ndarray 
        Wavelengths corresponding to throughput. Array must be same length as 
        throughput. If provided unitless, assumed to be angstroms.
    
    Returns
    -------
    mfd : astropy.units.quantity.Quantity
        Observed mean flux density per unit wavelength.
    """
    # Calculate step between successive wavelengths
    if type(wavelength) == np.ndarray:
        wavelength *= u.angstrom
    wstep = wavelength[1:] - wavelength[:-1]
    wstep = np.append(wstep, wstep[-1])
    mfd = np.sum(source_flux(wavelength) * throughput * wavelength * wstep)
    mfd /= np.sum(throughput * wavelength * wstep)
    return mfd


def csm_model_flux(z, dist=0., band='F275W', sed='Chev94'):
    """
    Calculate flux from CSM model in a given redshift and filter.

    Parameters
    ----------
    z : float or numpy.ndarray
        Redshift of source, which applies a horizontal shift to the spectrum
        as well as an overall decrease in flux.
    dist : astropy.units.quantity.Quantity or float, optional
        Distance to source, which is assumed to be in Mpc if provided as a
        float. If 0, the distance is automatically calculated from the redshift
        by the Hubble relation. The default is 0.
    band : str, optional
        Bandpass filter. Options are 'F275W' (default), 'FUV', or 'NUV'.
    sed : str
        Underlying CSM emission model, either 'Chev94' or 'flat'. Default is
        'Chev94'.

    Returns
    -------
    csm_flux : astropy.units.quantity.Quantity
        Spectral flux density per unit wavelength.
    """
    # Avoid initializing model more than necessary
    if sed == 'Chev94':
        csm_model = line_model
    else:
        csm_model = flat_model
    if type(z) == np.ndarray:
        csm_lum = np.array([csm_model(0, val)[band] for val in np.nditer(z)])
        csm_lum *= L_lambda_units
    else:
        csm_lum = csm_model(0, z)[band] * L_lambda_units
    csm_flux = luminosity2flux(csm_lum, z, dist=dist)
    return csm_flux


def luminosity2flux(luminosity, z, dist=0., h0=H0):
    """
    Convert intrinsic luminosity to observed flux assuming a flat cosmology.
    
    Parameters
    ----------
    luminosity : float or np.Array
        Spectral luminosity density per unit wavelength.
    z : float
        Redshift of source.
    dist : astropy.units.quantity.Quantity or float, optional
        Distance to source, which is assumed to be in Mpc if provided as a
        float. If 0, the distance is automatically calculated from the redshift
        by the Hubble relation. The default is 0.
    H0 : astropy.units.quantity.Quantity
        The Hubble constant to be used to calculate distance.
        
    Returns
    -------
    flux : astropy.units.quantity.Quantity
        Spectral flux density per unit wavelength.
    """
    if dist == 0.:
        dist = const.c.to('km/s') * z / h0
    if type(dist) == float:
        dist *= u.Mpc
    flux = luminosity / (4 * np.pi * dist.to('cm')**2 * (1+z)**3)
    return flux


def import_response_table(band):
    """
    Import SVO filter response table.

    Parameters
    ----------
    band : str
        Filter, one of 'FUV', 'NUV', or 'F275W'.

    Raises
    ------
    ValueError
        Value for 'band' outside acceptable list.

    Returns
    -------
    wavelength : astropy.units.quantity.Quantity
        Array of wavelengths in response table.
    transmission : numpy.ndarray
        Filter transmission at each wavelength.
    effective_area : astropy.units.quantity.Quantity
        Effective area at each wavelength.

    """
    if band not in ['FUV', 'NUV', 'F275W']:
        raise ValueError('Band must be FUV, NUV, or F275W')
    
    response_table = np.genfromtxt(Path('ref/%s.resp' % band), delimiter=' ')
    wavelength = response_table[:,0] * u.angstrom
    # HST and GALEX responses are given in different units
    if band == 'F275W':
        transmission = response_table[:,1]
        hst_aper_rad = 120 * u.cm
        effective_area = transmission * np.pi * hst_aper_rad**2
    else:
        effective_area = response_table[:,1] * u.cm**2
        galex_aper_rad = 25 * u.cm
        transmission = effective_area / (np.pi * galex_aper_rad**2)
    
    return wavelength, transmission, effective_area


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='Chev94', 
            help='CSM spectral model: "flat" or "Chev94", default "Chev94"')
    args = parser.parse_args()
    
    main(sed=args.model)
