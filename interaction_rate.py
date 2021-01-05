

def main(tstart_min, tstart_max, scale_min, scale_max, model='Chev94', 
        sigma=SIGMA, iterations=ITERATIONS):
    
    # Bin edges
    x_edges = np.array([tstart_min, tstart_max])
    y_edges = np.array([scale_min, scale_max])

    # Import sum histograms for GALEX and G19 non-detections
    galex_hist = import_recovery('galex', model, sigma, x_edges, y_edges)
    graham_hist = import_recovery('graham', model, sigma, x_edges, y_edges)
    # and G19 detections
    graham_det_hist = import_recovery('graham', model, sigma, x_edges, y_edges,
            detections=True)

    # DataFrame for number of trials per tstart bin and data source
    tstart_bins = pd.Series(x_edges[:-1])
    trials = pd.DataFrame([], index=tstart_bins)
    trials['G19'] = (graham_hist + graham_det_hist).T
    trials['GALEX'] = galex_hist.T
    trials['This study'] = trials['GALEX'] + trials['G19']

    # DataFrame for number of detections per tstart bin and data source
    detections = pd.DataFrame([], index=tstart_bins)
    detections['G19'] = graham_det_hist.T
    detections['GALEX'] = np.zeros((nbins, 1))
    detections['This study'] = detections['GALEX'] + detections['G19']

    # Import ASAS-SN and ZTF SNe
    asassn_det, asassn_all = count_asassn_sne()
    ztf_det, ztf_all = count_ztf_sne()

    # Calculate binomial confidence intervals
    bci_lower, bci_upper = bci_nan(detections, trials, conf=CONF)
    # Convert to percentages
    bci_lower *= 100
    bci_upper *= 100

    # Calculate binomial confidence intervals for external data
    asassn_bci = 100 * binom_conf_interval(asassn_det, asassn_all, 
            confidence_level=CONF, interval='jeffreys')
    ztf_bci = 100 * binom_conf_interval(ztf_det, ztf_all, confidence_level=CONF, 
            interval='jeffreys')
    external_bci = pd.DataFrame([asassn_bci, ztf_bci], index=['ASAS-SN', 'ZTF'],
            columns=['bci_lower', 'bci_upper'])

    # table(detections, trials, bci_upper, tstart_bins=TSTART_BINS, 
    #         output_file=Path('out/rates_%s.tex' % model))

    scale_mean = int(np.mean(y_edges))
    plot(bci_lower, bci_upper, external_bci, show=True, y_max=y_max,
            output_file=Path('out/rates_%s_scale%s.pdf' % (model, scale_mean))) 


def import_recovery(study, model, sigma, x_edges, y_edges, detections=False,
        iterations=ITERATIONS):
    """Import recovery save files and sum histograms with given bounds."""

    # File names
    save_dir = run_dir(study, model, sigma, detections)
    save_files = list(Path(save_dir).glob('*-%s.csv' % iterations))
    # Generate summed histogram
    print('Importing and summing %s saves from %s' % (study, save_dir))
    hist = sum_hist(save_files, x_edges, y_edges, save=False)
    count = np.nan_to_num(hist.iloc[0].to_numpy())

    return count


if __name__ == '__main__':
    main()