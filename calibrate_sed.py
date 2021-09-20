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
    """Convert luminosity to observed flux, assuming flat cosmology.
    
    Parameters
    ----------
        luminosity: float or np.Array
        z: float, redshift
        dist: float, distance to source
            if 0, calculates Hubble distance from given redshift
        H0: float, Hubble constant in km/s/Mpc
    """

    if dist > 0:
        dist = const.c.to('km/s') * z / (H0 * u.km / u.s / u.Mpc)
    flux = luminosity / (4 * np.pi * dist.to('cm')**2 * (1+z)**3)
    return flux

# import response curve
response_table = np.genfromtxt(Path('ref/F275W.resp'), delimiter=' ')
wavelength = response_table[:,0] * u.angstrom
hst_aper_rad = 120 * u.cm
effective_area = response_table[:,1] * np.pi * hst_aper_rad**2

# calculate zero point flux at relevant wavelengths
flux_zero_point = zero_point_lambda(wavelength)

# dlambda in response file
wavelength_step = [wavelength[i+1].value - wavelength[i].value for i in range(len(wavelength)-1)]
wavelength_step.append(wavelength_step[-1])
wavelength_step *= u.angstrom

# calculate mean flux density
numerator = np.sum(flux_zero_point * effective_area * wavelength_step / wavelength)
denominator = np.sum(effective_area * wavelength_step / wavelength)
mean_flux_density = numerator / denominator
print(mean_flux_density)

# Convert HST luminosity to counts for SN 2015cp
SN2015cp['f_lambda'] = luminosity2flux(SN2015cp['L_lambda'], SN2015cp['z'], SN2015cp['dist'])
photon_energy = (const.h * const.c / lambda_eff['F275W']).to('erg')
SN2015cp['counts'] = SN2015cp['f_lambda'] / photon_energy * u.count
SN2015cp['counts'] *= np.pi * hst_aper_rad**2 * w_eff['F275W']
SN2015cp['counts'] = SN2015cp['counts'].to('ct/s')
# print(SN2015cp['counts'])

# Calculate line model flux in F275W band
line_model = CSMmodel(0, 250, 0.3, scale=1, model='Chev94')
line_lum = line_model(0, SN2015cp['z'])['F275W'] * L_lambda_units
line_flux = luminosity2flux(line_lum, SN2015cp['z'], SN2015cp['dist'])
line_flux_density = np.sum(line_flux * effective_area * wavelength_step / wavelength)
line_flux_density /= denominator
print(line_flux_density)