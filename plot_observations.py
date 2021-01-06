import matplotlib.pyplot as plt

from light_curve import LightCurve
from supernova import Supernova
from utils import *

# Settings
DET_SIGMA = [5, 3] # detection significance threshold
DET_COUNT = [1, 3] # number of points above DET_SIGMA to count as detection
PLOT_SIGMA = 1 # multiple of uncertainty to plot as luminosity limit
YLIM = (1e22, 1e28) # erg/s/Hz
XLIM = (-50, 2250) # days

# Plot design
COLORS = {  'SN2007on': 'cyan',
            'SN2008hv': 'orange',
            'SN2009gf': 'green',
            'SN2010ai': 'magenta'
}
MARKERS = { 'SN2007on': 'o',
            'SN2008hv': 's',
            'SN2009gf': 'p',
            'SN2010ai': 'd'
}
DET_MS = 6 # detection marker size
DET_LIMIT_ALPHA = 0.6

def main():

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
    det_sne = ['SN2007on', 'SN2008hv', 'SN2009gf', 'SN2010ai']

    x_col = 't_delta_rest'
    y_col = 'luminosity_hostsub_hz'
    yerr_col = 'luminosity_hostsub_err_hz'
    
    fig, ax = plt.subplots(tight_layout=True)

    # Plot near-peak SNe Ia
    for sn_name in det_sne:
        sn = Supernova(sn_name, sn_info)
        band = 'NUV'
        lc = LightCurve(sn, band)
        lc.to_hz() # Convert to erg/s/Hz units
        # Plot near-peak detections
        detections = lc.detect_csm(DET_SIGMA, count=DET_COUNT)
        ax.errorbar(detections[x_col], detections[y_col], yerr=detections[yerr_col],
                label='%s (%s)' % (sn.name, band), linestyle='none', ms=DET_MS, 
                marker=MARKERS[sn.name], color=COLORS[sn.name], mec='k', ecolor='k', elinewidth=1, 
                zorder=9)
        
        # Plot nondetection limits
        nondetections = lc(tmin=XLIM[0], tmax=XLIM[1]).drop(detections.index)
        ax.scatter(nondetections[x_col], PLOT_SIGMA * nondetections[yerr_col], 
                marker='v', s=DET_MS**2, color=COLORS[sn.name], edgecolors='k', 
                alpha=DET_LIMIT_ALPHA, zorder=8)

    ax.set_xlabel('Time since discovery [days]')
    ax.set_ylabel('UV Luminosity [erg s$^{-1}$ Hz$^{-1}$]')
    ax.set_xlim(XLIM)
    ax.set_yscale('log')
    ax.set_ylim(YLIM)

    plt.show()


if __name__ == '__main__':

    main()