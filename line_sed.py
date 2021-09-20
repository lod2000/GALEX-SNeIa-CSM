#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 16:32:07 2021

@author: dubay.11
"""

import numpy as np
from pathlib import Path
import astropy.constants as const
import astropy.units as u
from CSMmodel import Chev94Model
from CSMmodel import CSMmodel

z = 0.01
band = 'FUV'

# From CSMmodel settings
# W0 = 1000. #model wwavelength start
# W1 = 3000. #model wavelength end
# DW = 0.1 #model wavelength step
# wavelength = np.arange(W0, W1, DW) * u.angstrom

def luminosity2flux(luminosity, z, H0=70):
    """Convert luminosity to observed flux, assuming flat cosmology."""

    dist = const.c.to('km/s') * z / (H0 * u.km / u.s / u.Mpc)
    flux = luminosity / (4 * np.pi * dist.to('cm')**2 * (1+z)**3)
    return flux

# chev94 = Chev94Model()
# line_lum = chev94.gen_model(0) * u.erg / u.s
# line_flux = luminosity2flux(line_lum, z)
# line_flux_density = line_flux / wavelength
# print(line_flux_density)

line_model = CSMmodel(0, 250, 0.3, scale=1, model='Chev94')
line_lum = line_model(0, z)[band] * u.erg / u.s
line_flux = luminosity2flux(line_lum, z)
print(line_flux)
