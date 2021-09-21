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

# Stats for SN 2015cp in F275W (Graham+ 2019)
SN2015cp = {'L_nu': 7.6e25 * L_nu_units,
            'L_lambda': 3.1e37 * L_lambda_units,
            'z': 0.0413,
            'dist': 167 * u.Mpc
            }

# filter parameters
lambda_eff = {'FUV': 1549. * u.angstrom, 
              'NUV': 2304.7 * u.angstrom, 
              'F275W': 2714.65 * u.angstrom
              }
w_eff = {'FUV': 265.57 * u.angstrom, 
         'NUV': 768.31 * u.angstrom, 
         'F275W': 416.66 * u.angstrom
         }

# flux with 0 color in AB system (flat in f_nu)
zero_point_nu = 3.63e-20 * f_nu_units
zero_point_lambda = lambda l: zero_point_nu * const.c.to('AA Hz') * l**-2

def luminosity2flux(luminosity, z, dist=0., H0=70):
    """
    Convert luminosity to observed flux, assuming flat cosmology.
    
    Parameters
    ----------
        luminosity : float or np.Array
        z : float, redshift
        dist : float, distance to source
            if 0, calculates Hubble distance from given redshift
        H0 : float, Hubble constant in km/s/Mpc
    """

    if dist == 0:
        dist = const.c.to('km/s') * z / (H0 * u.km / u.s / u.Mpc)
    flux = luminosity / (4 * np.pi * dist.to('cm')**2 * (1+z)**3)
    return flux

def mean_flux_density(flux_dist, throughput, wavelength):
    """
    Calculate the mean flux density to be observed given an assumed
    intrinsic flux distribution and passband throughput over a range
    of wavelengths.
    
    Parameters
    ----------
        flux_dist : function which accepts a wavelength as input
        throughput : np.Array
        wavelength : np.Array of same length as throughput
    """
    
    # step between successive wavelengths
    wstep = wavelength[1:] - wavelength[:-1]
    wstep = np.append(wstep, wstep[-1])
    mfd = np.sum(flux_dist(wavelength) * throughput * wavelength * wstep)
    mfd /= np.sum(throughput * wavelength * wstep)
    return mfd

# initialize CSM line emisison model
line_model = CSMmodel(0, 250, 0.3, scale=1, model='Chev94')
def line_emission_flux(z, dist=0., band='F275W'):
    line_lum = line_model(0, z)[band] * L_lambda_units
    line_flux = luminosity2flux(line_lum, z, dist=dist)
    return line_flux    

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
line_lum = line_model(0, SN2015cp['z'])['F275W'] * L_lambda_units
line_flux = luminosity2flux(line_lum, SN2015cp['z'], SN2015cp['dist'])
print('flux density of line model at 15cp scale: ', f'{line_flux:0.03e}')

# ratio between AB zero point and line model as function of redshift
fratio = lambda z: mfd_ab(z) / line_emission_flux(z)
print(fratio(SN2015cp['z']))

# Convert HST luminosity to counts for SN 2015cp
SN2015cp['f_lambda'] = luminosity2flux(SN2015cp['L_lambda'], SN2015cp['z'], SN2015cp['dist'])
print('F275W flux of SN 2015cp: ', f"{SN2015cp['f_lambda']:0.03e}")
fratio_15cp = mfd_ab(SN2015cp['z']) / SN2015cp['f_lambda']
print(fratio_15cp)

calibration = lambda z: mfd_ab(z) / (fratio_15cp * line_emission_flux(z, dist=SN2015cp['dist']))

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
