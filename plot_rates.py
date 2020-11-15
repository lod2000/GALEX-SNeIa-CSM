from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import binom_conf_interval
from utils import *
from plot import sum_hist

TSTART_BINS = [0, 100, 500, 100]
CONF = 0.9
MODEL = 'Chev94'
SCALE = 50

def main():
    hist_file = Path('out/%s-rates-%s-%ssigma' % (study, model, sigma))
    hist = 
    plot()


def plot(model=MODEL):

    fig, ax = plt.subplots()

    # Include all nondetections below the luminosity of 2015cp
    below_graham = nondetections[nondetections['luminosity_hostsub_err_hz'] * LIMIT_SIGMA < cutoff]
    # Also include limits from near-peak SNe
    below_graham.append(lc_non[lc_non['luminosity_hostsub_err_hz'] * LIMIT_SIGMA < cutoff])
    # Only those after discovery
    below_graham = below_graham[below_graham['t_delta_rest'] > 0]
    print('Number of SNe with limits fainter than 2015cp: %s' % len(below_graham.drop_duplicates('name').index))
    print('Number of observations with limits fainter than 2015cp: %s' % len(below_graham.index))
    bins = [0, 100, 500, 2500]
    k = []
    n = []
    labels = []
    for i in range(len(bins)-1):
        limits = below_graham[(below_graham['t_delta_rest'] >= bins[i]) & (below_graham['t_delta_rest'] < bins[i+1])]
        discrete_sne = limits.drop_duplicates('name')
        k.append(0)
        n.append(len(discrete_sne.index))
        labels.append('%s - %s' % (bins[i], bins[i+1]))
    print(bins)
    print(n)
    bci = 100 * binom_conf_interval(k, n, confidence_level=CONF, interval='jeffreys')
    print(bci)
    midpoint = np.mean(bci, axis=0)
    x_pos = np.arange(len(bins)-1)
    ax.errorbar(x_pos, midpoint, yerr=np.abs(bci - midpoint), capsize=10, 
            marker='o', linestyle='none', ms=10, mec='r', c='r', mfc='w',
            label='This study')

    # Confidence interval from Yao 2019
    ztf_bci = 100 * binom_conf_interval(1, 127, confidence_level=CONF, interval='jeffreys')
    print(ztf_bci)
    ztf_mean = np.mean(ztf_bci)
    ax.errorbar([0.1], [ztf_mean], yerr=([ztf_mean - ztf_bci[0]], [ztf_bci[1] - ztf_mean]),
            marker='o', c='b', linestyle='none', ms=10, capsize=10, mec='b', mfc='w',
            label='ZTF')

    # ASAS-SN interval
    asassn_bci = 100 * binom_conf_interval(3, 460, confidence_level=CONF, interval='jeffreys')
    print(asassn_bci)
    asassn_mean = np.mean(asassn_bci)
    ax.errorbar([0.2], [asassn_mean], yerr=([asassn_mean - asassn_bci[0]], [asassn_bci[1] - asassn_mean]),
            marker='o', c='orange', linestyle='none', ms=10, capsize=10, mec='orange', mfc='w',
            label='ASAS-SN')

    # Confidence interval & assumed late-onset rate from Graham 2019
    graham_rate = 6
    graham_bci = 100 * binom_conf_interval(1, 64, confidence_level=CONF, interval='jeffreys')
    print(graham_bci)
    ax.errorbar([2.1], [graham_rate], yerr=([graham_rate - graham_bci[0]], [graham_bci[1] - graham_rate]),
            marker='v', color='g', linestyle='none', ms=15, capsize=10, label='G19')
    # ax.annotate('G19', (2.1, graham_rate), textcoords='offset points', 
    #         xytext=(10, 0), ha='left', va='center', size=18, color='g')

    ax.set_xlim((x_pos[0]-0.5, x_pos[-1]+0.5))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.set_xlabel('Rest frame time since discovery [days]')
    ax.set_ylabel('Rate of CSM interaction [%]')

    # Preliminary!
    if args.presentation:
        fig.text(0.95, 0.05, 'PRELIMINARY', fontsize=72, color='gray', 
                rotation='30', ha='right', va='bottom', alpha=0.5)

    plt.tight_layout()
    plt.legend()
    plt.savefig(Path('out/%s-rates.png' % model), dpi=300)
    plt.show()


if __name__ == '__main__':
    main()