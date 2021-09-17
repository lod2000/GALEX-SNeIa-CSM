#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:36:42 2021

@author: dubay.11
"""

import numpy as np
from pathlib import Path
import astropy.constants as const
import astropy.units as u

# input params
BAND = 'FUV'

# common unit macros
f_nu_units = u.erg * u.s**-1 * u.cm**-2 * u.hertz**-1
f_lambda_units = u.erg * u.s**-1 * u.cm**-2 * u.angstrom**-1

# flux with 0 color in AB system (flat in f_nu)
zero_point_nu = 3.63e-20 * f_nu_units
zero_point_lambda = lambda l: zero_point_nu * const.c.to('AA Hz') * l**-2

# import response curve
response_table = np.genfromtxt('%s.resp' % BAND, delimiter=' ')
wavelength = response_table[:,0] * u.angstrom
effective_area = response_table[:,1] * u.cm**2

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

def galex_flux2mag(flux, band):
    """Convert flux in f_lambda to AB magnitudes in either GALEX band."""
    
    factor = {'FUV': 1.40e-15 * f_lambda_units,
              'NUV': 2.06e-16 * f_lambda_units}
    zero_point_ab = {'FUV': 18.82, 'NUV': 20.08}
    return -2.5 * np.log10(flux/factor[band]) + zero_point_ab[band]    

print('Calculated mean flux density: ', f'{mean_flux_density:0.03e}')
print('Conversion to AB magnitudes: ', f'{galex_flux2mag(mean_flux_density, BAND):.02f}')