#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 17:35:08 2021

@author: dubay.11
"""

import numpy as np
from pathlib import Path
import astropy.constants as const
import astropy.units as u
from astropy.visualization import quantity_support
from tqdm import tqdm
from matplotlib import pyplot as plt
from CSMmodel import CSMmodel

# redshift step
dz = 0.01

# common unit macros
f_nu_units = u.erg * u.s**-1 * u.cm**-2 * u.hertz**-1
f_lambda_units = u.erg * u.s**-1 * u.cm**-2 * u.angstrom**-1

# flux with 0 color in AB system (flat in f_nu)
zero_point_nu = 3.63e-20 * f_nu_units
zero_point_lambda = lambda l: zero_point_nu * const.c.to('AA Hz') * l**-2

lambda_eff = {'FUV': 1549., 'NUV': 2304.7, 'F275W': 2714.65}
w_eff = {'FUV': 265.57, 'NUV': 768.31, 'F275W': 416.66}

def luminosity2flux(luminosity, z, H0=70):
    """Convert luminosity to observed flux, assuming flat cosmology."""

    dist = const.c.to('km/s') * z / (H0 * u.km / u.s / u.Mpc)
    flux = luminosity / (4 * np.pi * dist.to('cm')**2 * (1+z)**3)
    return flux

galex_lookup_table = {}
galex_lookup_table['z'] = np.arange(dz, 0.5, dz)
N_table = np.shape(galex_lookup_table['z'])[0]

for band in ['FUV', 'NUV', 'F275W']:
    galex_lookup_table[band] = np.zeros(N_table)
    
    # import response curve
    response_table = np.genfromtxt(Path('ref/%s.resp' % band), delimiter=' ')
    wavelength = response_table[:,0] * u.angstrom
    effective_area = response_table[:,1] * u.cm**2
    if band == 'F275W':
         # HST band given in transmission %, not effective area
        effective_area *= np.pi * 120**2
    
    # dlambda in response file
    wavelength_step = [wavelength[i+1].value - wavelength[i].value for i in range(len(wavelength)-1)]
    wavelength_step.append(wavelength_step[-1])
    wavelength_step *= u.angstrom
    
    # calculate mean flux density of AB zero point
    flux_zero_point = zero_point_lambda(wavelength)
    numerator = np.sum(flux_zero_point * effective_area * wavelength_step / wavelength)
    denominator = np.sum(effective_area * wavelength_step / wavelength)
    mean_flux_density = numerator / denominator
    
    # calculate total flux of flat SED convolved over filter response curve
    total_flux = np.sum(flux_zero_point * effective_area * wavelength_step)
    
    # calculate total flux of line SED at z=0
    line_model = CSMmodel(0, 250, 0.3, scale=1, model='Chev94')
    line_lum = line_model(0, 0)[band] * u.erg / u.s / u.angstrom
    line_lum *= w_eff[band] * u.angstrom
    
    # put flat, line SEDs on same scale at z=0
    scale = line_lum / total_flux
    
    # ratio between flux density of flat, line SEDs at varying z
    line_model = CSMmodel(0, 250, 0.3, scale=1/scale, model='Chev94')
    for i, z in enumerate(tqdm(galex_lookup_table['z'])):
        line_lum = line_model(0, z)[band] * u.erg / u.s / u.angstrom / u.cm**2
        galex_lookup_table[band][i] = line_lum / mean_flux_density

# Plot
fig, ax = plt.subplots()
with quantity_support():
    for band in ['FUV', 'NUV', 'F275W']:
        ax.plot(galex_lookup_table['z'], galex_lookup_table[band],
                label=band)
ax.set_xlabel('z')
ax.set_ylabel('conversion factor')
ax.set_yscale('log')
plt.legend()
plt.show()