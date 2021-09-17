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
from CSMmodel import Chev94Model
from CSMmodel import CSMmodel

# common unit macros
f_nu_units = u.erg * u.s**-1 * u.cm**-2 * u.hertz**-1
f_lambda_units = u.erg * u.s**-1 * u.cm**-2 * u.angstrom**-1

# flux with 0 color in AB system (flat in f_nu)
zero_point_nu = 3.63e-20 * f_nu_units
zero_point_lambda = lambda l: zero_point_nu * const.c.to('AA Hz') * l**-2


def luminosity2flux(luminosity, z, H0=70):
    """Convert luminosity to observed flux, assuming flat cosmology."""

    dist = const.c.to('km/s') * z / (H0 * u.km / u.s / u.Mpc)
    flux = luminosity / (4 * np.pi * dist.to('cm')**2 * (1+z)**3)
    return flux

galex_lookup_table = {}
galex_lookup_table['z'] = np.arange(0.01, 0.51, 0.01)

for band in ['FUV', 'NUV']:
    galex_lookup_table[band] = []
    
    # import response curve
    response_table = np.genfromtxt(Path('ref/%s.resp' % band), delimiter=' ')
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
    print(mean_flux_density)
    
    for z in galex_lookup_table['z']:
        line_model = CSMmodel(0, 250, 0.3, scale=1, model='Chev94')
        line_lum = line_model(0, z)[band] * u.erg / u.s
        line_flux = luminosity2flux(line_lum, z)
        galex_lookup_table[band].append(line_flux)
    
    print(galex_lookup_table[band])
    # TODO not sure where to go from here