from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import ipdb
from lmfit.models import GaussianModel
from scipy.interpolate import interp1d
from utils import *

W0 = 1000. #model wwavelength start
W1 = 3000. #model wavelength end
DW = 0.1 #model wavelength step

GALEX_EFF_AREA = np.pi*25.0**2. # cm2
HST_AREA = (1.2e2)**2. * np.pi # cm2

# Luminosity of SN 2015cp CSM interaction (Graham+ 2019)
# Used as baseline, scale = 1
F275W_LAMBDA_EFF = 2714.65
L_2015cp = 7.6e25 # erg/s/Hz
L_2015cp_cgs = L_2015cp * (3e18) / (F275W_LAMBDA_EFF**2) # erg/s/A
Z_2015cp = 0.0413
CHEV94_2015cp_SCALE = 52.035

T0 = 0. #model time start
T1 = 3000. #model time end
DT = 0.1 #model time step


def main(tstart, twidth, decay_rate, scale, model='Chev94', show=False):

    # Plot Chev94 spectrum at original scale
    chev_model = Chev94Model(scale=scale/CHEV94_2015cp_SCALE)
    chev_model.plot(0, save=True, show=show)

    # Plot Chev94 model vs redshift
    csm_model = CSMmodel(tstart, twidth, decay_rate, scale=scale, model='Chev94')
    csm_model.plot_redshift(save=True, show=show)

    # Plot CSM light curve
    fig, ax = plt.subplots(tight_layout=True)

    t = np.arange(tstart-1, tstart+500, 1)
    csm_lum = csm_model(t, Z_2015cp)
    band = 'F275W'
    ax.plot(t, F275W_LAMBDA_EFF * csm_lum[band], label=band)

    ax.set_yscale('log')
    ax.set_ylim((2e39, 1.5e41))
    ax.set_xlabel('Time since discovery [days]')
    ax.set_ylabel('$\lambda L_\lambda$ [erg s$^{-1}$]')

    plt.savefig(Path('out/CSM_model.png'), dpi=300)
    plt.close()

    # Plot flat-spectrum model
    # fig, ax = plt.subplots(tight_layout=True)
    # flat_model = FlatModel(scale=1.)
    # wl = np.arange(W0, W1, DW)
    # fl = flat_model.gen_model(0)
    # ax.plot(wl, fl)
    # plt.show()


class CSMmodel:
    def __init__(self, tstart, twidth, decay_rate, scale=1, vwidth=2000,
            model='Chev94'):
        """
        Initializes a CSM model with input variables:
            tstart: Days after peak that ejecta impacts CSM 
            twidth: length of the light curve plateau in days
            decay_rate: decay rate per 100 days after plateau ends - see Graham paper
            scale: multiplicative scale factor applied to the CSM model to indicative brighter/fainter CSM interaction
            vwidth: velocity width of the CSM emission lines in km/s, using 2000 based on typical emission lines of SNe Ia-CSM
            model: 'Chev94' for emission line-based or 'flat' for flat

        Attributes:
            wlarr: np.array of wavelength values for the Chevalier CSM model determined by W0,W1,DW
            spec_model: Class instance of Chev94Model or FlatModel with input scale and vwidth parameters
            time: time array used for model interpolation determined by T0,T1,DT
            tscale: scale factor applied to the CSM light curve after platuea regime has ended determined by decay_rate

        Methods (see indvidual methods for more info):
            ___call___: generates a model CSM profile at input redshift and time and compute the resulting filter luminosity
        """

        # Parameters
        self.tstart = tstart
        self.twidth = twidth
        self.decay_rate = decay_rate
        self.scale = scale

        self.wlarr = np.arange(W0, W1, DW)

        if model == 'Chev94':
            self.spec_model = Chev94Model(vwidth=vwidth, scale=scale)
        elif model == 'flat':
            self.spec_model = FlatModel(scale=scale)
        else:
            print('No such model %s' % model)

        self.time = np.arange(T0, T1, DT)
        self.tscale = np.zeros_like(self.time)
        idx1 = (self.time >= tstart) * (self.time < (tstart + twidth))
        self.tscale[idx1] = 1.
        idx2 = (self.time >= (tstart + twidth))
        self.tscale[idx2] = (decay_rate)**((self.time[idx2]-(tstart+twidth))/100.)



    def __call__(self, t, z, plot_spec=False):
        """
        Computes the filter luminosity for the CSMmodel at a given epoch and redshift
        Arguments:
            t (float or np.array): time of observations to return filter lumnoisites
            z (float): redshift
            plot_spec (bool, optional): plot the redshifted+scaled spectrum before returning filter luminosities? Default: False

        Returns:
            filter_lum (dict): Luminosities per filter for this CSM model at input times t

        """

        #only running it at 1yr epoch
        init_spec = self.spec_model.gen_model(366.) # erg /s / A

        #shift to redshift      
        wl_obs = self.wlarr * (1.+z)

        #compute fluxes
        tscale = np.interp(t, self.time, self.tscale)

        if plot_spec:
            plt.plot(self.wlarr, init_spec, 'k-', label='original')
            if isinstance(t, float):
                plt.plot(wl_obs, tscale*init_spec, label='shifted+scaled (t=%.1f)' % t)
            else:
                for _t, ts in zip(t, tscale):
                    plt.plot(wl_obs, ts*init_spec, label='shifted+scaled (t=%.1f)' % _t)
            plt.legend()
            plt.show()

        filters = ['FUV', 'NUV', 'F275W']
        fluxes = dict([(f, filter_flux(f, wl_obs, init_spec)*tscale) for f in filters])

        return fluxes


    def plot_redshift(self, show=False, save=True, zmin=0., zmax=0.5, tstep=0.):
        """Plot filter luminosity vs redshift."""

        fig, ax = plt.subplots()

        # Set up range of redshifts
        z_vals = np.arange(0., 0.51, 0.01)
        luminosity = {
            'FUV':[],
            'NUV':[],
            'F275W':[]
        }
        line_style = {'F275W': ':', 'NUV': '--', 'FUV': '-'}
        lambda_eff = {'FUV': 1549., 'NUV': 2304.7, 'F275W': 2714.65}

        for z in z_vals:
            f = self(self.tstart + tstep, z)
            for band, fl in f.items(): luminosity[band].append(fl)

        text_idx = 42
        for i, band in enumerate(['F275W', 'NUV', 'FUV']):
            l = lambda_eff[band]
            l_array = l * np.array(luminosity[band])
            ax.plot(z_vals, l_array, color=COLORS[band], label=band, 
                    lw=1, ls=line_style[band])
            ax.text(z_vals[text_idx], luminosity[band][text_idx] * l*1.1, band, 
                    color=COLORS[band], size=10, ha='center', va='bottom')

        ax.set_yscale('log')

        ax.set_xlabel('Redshift')
        ax.set_ylabel('$\lambda L_\lambda$ [erg s$^{-1}$]')

        plt.tight_layout(pad=0.3)

        plt.savefig(Path('out/Chev94_redshift.pdf'), dpi=300)
        plt.savefig(Path('out/Chev94_redshift.png'), dpi=300)
        if show:
            plt.show()
        else:
            plt.close()


def filter_flux(band, wl, flux):
    """Scale fluxes at given wavelengths by filter response.
    Inputs:
        band: 'FUV', 'NUV', or 'F275W'
        wl: array, wavelength
        flux: array, flux (same length as wl)
    Output:
        filter_flux: array, scaled flux at given wavelength
    """

    # Import filter response curve
    det_wl, det_fl = np.genfromtxt(Path('ref/%s.resp' % band), unpack=True, 
            dtype=float)
    # Interpolate curve of filter throughput
    filt = interp1d(det_wl, det_fl, kind='slinear', bounds_error=False, 
            fill_value=0.)
    # Compute flux observed by filter
    obs_fl = filt(wl) * flux # erg/s/A^2
    # Integrate total observed flux and normalize by filter response
    filter_flux = np.trapz(obs_fl) / np.trapz(filt(wl)) # erg/s/A
    return filter_flux


class Chev94Model:
    def __init__(self, vwidth=2000., scale=1.):
        """
        CSM model based on Chevalier+Fransson 1994 paper derived for Type II SNe
        This ignores continuum contributions and only computes the expected line emission
        Arguments:
            vwidth (float): velocity width of the line profiles in km/s, default=2000
            scale (float): multiplicative scale factor to shift the CSM model bright/fainter, default=1.
        """

        self.fname = Path('ref/chev_model.dat')
        self.times = np.array([1., 2., 5., 10., 17.5, 30.])*365.25
        self.model_data = {}
        self.line_wl = {}
        for (name, wl, fl1, fl2, fl5, fl10, fl17, fl30) in [line.strip().split() for line in open(self.fname, 'r').readlines() if not line.startswith('#')]:
            if name == 'Hbeta':
                coeffs = np.array([float(fl)*1e36*scale for fl in [fl1, fl2, fl5, fl10, fl17, fl30]])
                continue
            linelum = np.array([float(fl) for fl in [fl1,fl2,fl5,fl10,fl17,fl30]])*coeffs
            linelum *= CHEV94_2015cp_SCALE # scale so L(2015cp) = 1
            model = LineModel(float(wl), self.times, linelum)
            self.model_data[name] = model
            self.line_wl[name] = wl


    def gen_model(self, t):
        wl = np.arange(W0, W1, DW)
        fl = np.zeros_like(wl)
        for name, model in self.model_data.items():
            fl += model(t)
        return fl


    def plot(self, t, save=True, show=False):
        """Plot spectral model."""

        fig, ax = plt.subplots()

        wl = np.arange(W0, W1, DW)
        fl = self.gen_model(t)
        # ax.plot(wl, fl/1e37)
        ax.plot(wl, fl)

        y_max = 6

        # Label peaks
        for name in self.line_wl.keys():
            # Set label position
            peak_wl = float(self.line_wl[name])
            x = peak_wl + 40
            peak_fl = fl[np.argwhere(np.round(wl,1) == peak_wl)[0][0]]
            # y = max(min(peak_fl/1e37, y_max - 0.4), 0.4)
            y = peak_fl
            # Text alignment
            va = 'center'
            ha = 'left'
            # Ignore smaller lines which get in the way
            if name in ['OV]', 'SiII]']:
                continue
            # Adjust names
            text = name
            text = text.replace('I', ' I', 1)
            text = text.replace('V', ' V', 1)
            text = text.replace('Lyman_alpha', 'Ly-α')
            text = text.replace('V I', 'VI')
            text = text.replace('I V', 'IV')
            # Custom adjustments
            if text in ['C II]', 'C I]', 'O VI', 'C IV', 'Mg II', 'Ly-α']:
                x = peak_wl
                ha = 'center'
                va = 'bottom'
                # y = peak_fl/1e37 + 0.2
                y = peak_fl * 1.1
            elif text in ['He II', '[Ne IV]']:
                x = peak_wl - 50
                ha = 'left'
                va = 'bottom'
                # y = peak_fl/1e37 + 0.3
                y = peak_fl * 1.1
            elif text in ['O III]']:
                x = peak_wl
                va = 'bottom'
            ax.text(x, y, text, size=8, va=va, ha=ha)

        ax.set_xlim((850, 3100))
        # ax.set_ylim((-0.5, 6.5))
        ax.set_yscale('log')
        ax.set_ylim((1e35, 1e39))

        ax.set_xlabel('Wavelength [Å]')
        # ax.set_ylabel('Luminosity [$10^{37}$ erg s$^{-1}$]')
        ax.set_ylabel('Luminosity [erg s$^{-1}$]')

        plt.tight_layout(pad=0.3)

        if save: 
            plt.savefig(Path('out/Chev94_spectrum.pdf'), dpi=300)
            plt.savefig(Path('out/Chev94_spectrum.png'), dpi=300)
        if show:
            plt.show()
        else:
            plt.close()


class LineModel:
    def __init__(self, wl, times, linelum, vwidth=500.):
        """
        Model for a given emission line profile
        Arguments:
            wl: wl array used for generating the model
            times: input times for corresponding line luminosities
            linelum: integrated luminosity of the line at input times
            vwidth: velocity width of the lines in km/s
        """

        self.wl = wl
        self.vwidth = vwidth
        self.times = times
        self.linelum = linelum
        spectrum = gen_spectrum(wl, vwidth)
        lum = np.array([lum*spectrum for lum in self.linelum])

        self.interper = interp1d(self.times, lum.T, kind='quadratic', bounds_error=False, fill_value='extrapolate')

    def __call__(self, t):
        if t < 0: raise ValueError
        else:
            return self.interper(t)


class FlatModel:
    def __init__(self, scale=1.):
        """CSM model based on a flat spectrum."""
        self.model_data = L_2015cp_cgs * scale # erg/s/A

    def gen_model(self, t):
        wl = np.arange(W0, W1, DW)
        fl = np.zeros_like(wl) + self.model_data
        return fl


def gen_spectrum(wl, vwidth):
    arr = np.arange(W0, W1, DW)
    gauss = GaussianModel()
    fl = gauss.eval(x=arr, center=wl, sigma=wl*vwidth/3e5, amplitude=1.)
    return fl


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tstart', '-s', help='When the ejecta impacts the CSM, days after max', default=300., type=float)
    parser.add_argument('--twidth', '-w', help='plateau width [days]', default=200., type=float)
    parser.add_argument('--decay-rate', '-D', help='fractional decay rate per 100 days', default=0.3, type=float)
    parser.add_argument('--scale', '-S', help='scale factor', default=1., type=float)
    parser.add_argument('--model', '-m', type=str, default='Chev94', help='Spectral model (flat or Chev94)')
    parser.add_argument('--show', action='store_true', help='Show plots')

    args = parser.parse_args()
    main(args.tstart, args.twidth, args.decay_rate, args.scale, args.model, args.show)
