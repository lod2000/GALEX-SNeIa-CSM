#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
import astropy.constants as const
import astropy.units as u
from astropy.visualization import quantity_support
from tqdm import tqdm
from matplotlib import pyplot as plt
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
zero_point_lambda = lambda l: zero_point_nu * const.c.to('AA Hz') * l**-2

# initialize CSM line emisison model
line_model = CSMmodel(0, 250, 0.3, scale=1, model='Chev94')

def main():
    # import response curve, convert to effective area
    response_table = np.genfromtxt(Path('ref/F275W.resp'), delimiter=' ')
    wavelength = response_table[:,0] * u.angstrom
    transmission = response_table[:,1]
    hst_aper_rad = 120 * u.cm
    effective_area = transmission * np.pi * hst_aper_rad**2
    
    # calculate mean flux density of AB zero point as function of redshift
    mfd0 = mean_flux_density(zero_point_lambda, transmission, wavelength) # at z=0
    mfd_ab = lambda z: mfd0 * (1+z)**-3
    print('mean flux density of AB zero point at z_15cp: ', f'{mfd_ab(SN2015cp["z"]):0.03e}')
    
    # mean flux density of line emission model at scale 1 (L_15cp)
    # line_model = CSMmodel(0, 250, 0.3, scale=1, model='Chev94')
    # line_lum = line_model(0, SN2015cp['z'])['F275W'] * L_lambda_units
    # line_flux = luminosity2flux(line_lum, SN2015cp['z'], SN2015cp['dist'])
    # print('flux density of line model at 15cp scale: ', f'{line_flux:0.03e}')
    
    # ratio between AB zero point and line model as function of redshift
    fratio = lambda z: mfd_ab(z) / line_emission_flux(z)
    print(fratio(SN2015cp['z']))
    
    # Convert HST luminosity to counts for SN 2015cp
    SN2015cp['f_lambda'] = luminosity2flux(SN2015cp['L_lambda'], SN2015cp['z'], SN2015cp['dist'])
    print('F275W flux of SN 2015cp: ', f"{SN2015cp['f_lambda']:0.03e}")
    fratio_15cp = mfd_ab(SN2015cp['z']) / SN2015cp['f_lambda']
    print(fratio_15cp)
    mfd_ab_scaled = lambda z: mfd_ab(z) / fratio_15cp
    
    calibration = lambda z: mfd_ab_scaled(z) / line_emission_flux(z, dist=SN2015cp['dist'])
    
    # Plot fluxes
    zarr = np.arange(0.01, 0.5, 0.01)
    with quantity_support():
        plt.plot(zarr, mfd_ab(zarr) / fratio_15cp, label='scaled AB zero point')
        larr = np.array([line_model(0, z)['F275W'] for z in zarr]) * L_lambda_units
        farr = luminosity2flux(larr, zarr, dist=SN2015cp['dist'])
        plt.plot(zarr, farr, label='line-emission')
        plt.axvline(SN2015cp['z'], 0, 1, c='r', label='SN 2015cp')
        plt.yscale('log')
        plt.xlabel('z')
        plt.ylabel('mean flux density')
        plt.suptitle('F275W')
        plt.legend()
        plt.show()
        
    # Plot calibrations
    plt.plot(zarr, [calibration(z) for z in zarr])
    plt.axvline(SN2015cp['z'], 0, 1, c='r', label='SN 2015cp')
    plt.yscale('log')
    plt.xlabel('z')
    plt.ylabel('calibration')
    plt.suptitle('F275W calibration')
    plt.show()
    
    # fratio_15cp = SN2015cp['f_lambda'] / (mfd_flat / (1+SN2015cp['z'])**3)
    # fratio = lambda z: SN2015cp['f_lambda'] / (mfd_flat / (1+z)**3)
    # fratio_15cp = fratio(SN2015cp['z'])
    # photon_energy = (const.h * const.c / lambda_eff['F275W']).to('erg')
    # electron_flux = SN2015cp['f_lambda'] / photon_energy * u.count # ct/s/cm2/AA
    # wstep = wavelength[1:] - wavelength[:-1]
    # wstep = np.append(wstep, wstep[-1])
    # SN2015cp['counts'] = electron_flux * np.sum(wstep * effective_area)
    # print(SN2015cp['counts'])
    # print(SN2015cp['f_lambda'] / SN2015cp['counts'])
    
    # Calculate line model flux in F275W band at scale 1
    # print(SN2015cp['f_lambda'] / line_flux)
    # print(fratio_15cp * mfd_flat / line_flux)
    
    test_z = 0.1
    # test_lum = line_model(0, test_z)['F275W'] * L_lambda_units
    # test_flux = luminosity2flux(test_lum, test_z)
    # print(fratio(test_z) * (mfd_flat / (1+test_z)**3) / test_flux)

def mean_flux_density(source_flux, throughput, wavelength):
    """
    Calculate the mean flux density to be observed given an assumed
    intrinsic source flux distribution and passband throughput over a range
    of wavelengths.
    
    Parameters
    ----------
    source_flux : function 
        Source flux distribution as a function of wavelength.
    throughput : ndarray
        Throughput of passband filter.
    wavelength : Quantity or ndarray 
        Wavelengths corresponding to throughput. Array must be same length as 
        throughput. If provided unitless, assumed to be angstroms.
    
    Returns
    -------
    mfd : Quantity
        Observed mean flux density per unit wavelength
    """
    
    # Calculate step between successive wavelengths
    if type(wavelength) == np.ndarray:
        wavelength *= u.angstrom
    wstep = wavelength[1:] - wavelength[:-1]
    wstep = np.append(wstep, wstep[-1])
    mfd = np.sum(source_flux(wavelength) * throughput * wavelength * wstep)
    mfd /= np.sum(throughput * wavelength * wstep)
    return mfd

def line_emission_flux(z, dist=0., band='F275W'):
    """
    Calculate flux from CSM line-emission model in a given redshift and filter.

    Parameters
    ----------
    z : float
        Redshift of source, which applies a horizontal shift to the spectrum
        as well as an overall decrease in flux.
    dist : Quantity or float, optional
        Distance to source, which is assumed to be in Mpc if provided as a
        float. If 0, the distance is automatically calculated from the redshift
        by the Hubble relation. The default is 0.
    band : str, optional
        Bandpass filter. Options are 'F275W' (default), 'FUV', or 'NUV'.

    Returns
    -------
    line_flux : Quantity
        Spectral flux density per unit wavelength.
    """
    
    line_lum = line_model(0, z)[band] * L_lambda_units
    line_flux = luminosity2flux(line_lum, z, dist=dist)
    return line_flux   

def luminosity2flux(luminosity, z, dist=0., h0=H0):
    """
    Convert intrinsic luminosity to observed flux at a given redshift and 
    distance, assuming a flat cosmology.
    
    Parameters
    ----------
    luminosity : float or np.Array
        Spectral luminosity density per unit wavelength.
    z : float
        Redshift of source.
    dist : Quantity or float, optional
        Distance to source, which is assumed to be in Mpc if provided as a
        float. If 0, the distance is automatically calculated from the redshift
        by the Hubble relation. The default is 0.
    H0 : Quantity
        The Hubble constant to be used to calculate distance.
        
    Returns
    -------
    flux : Quantity
        Spectral flux density per unit wavelength.
    """

    if dist == 0.:
        dist = const.c.to('km/s') * z / h0
    if type(dist) == float:
        dist *= u.Mpc
    flux = luminosity / (4 * np.pi * dist.to('cm')**2 * (1+z)**3)
    return flux 

if __name__ == '__main__':
    main()
