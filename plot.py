from functools import reduce
from functools import partial
import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from utils import *


def main(iterations, t_min=RECOV_MIN, t_max=1500, scale_min=SCALE_MIN,
        scale_max=SCALE_MAX, bin_width=50, bin_height=0.1):
    
    # List of files in save dir
    save_files = get_save_files(iterations)

    # Bin edges
    x_edges = np.arange(t_min, t_max, bin_width)
    y_edges = np.arange(scale_min, scale_max, bin_height)

    hist = []
    with Pool() as pool:
        func = partial(get_hist, x_edges=x_edges, y_edges=y_edges)
        imap = pool.imap(func, save_files, chunksize=10)
        for h in tqdm(imap, total=len(save_files)):
            hist.append(h)

    # Import recovery save files
    # hist = []
    # for f in tqdm(save_files):
        # rd = RecoveryData(f)
        # Append histogram of time vs scale for recovered data
        # hist.append(rd.hist(x_edges, y_edges))

    # Sum 2D histograms
    hist = reduce(lambda x, y: x.add(y, fill_value=0), hist)
    print(hist)


def get_save_files(iterations, band='*', save_dir=SAVE_DIR):
    """Return list of files in save directory for given iterations"""

    save_files = list(Path(save_dir).glob('*-%s-%s.csv' % (band, iterations)))
    return save_files


def get_hist(fname, x_edges, y_edges):
    """Import recovery save data and return histogram."""

    rd = RecoveryData(fname)
    return rd.hist(x_edges, y_edges)


class RecoveryData:
    def __init__(self, fname):
        """Import recovery save data."""

        # Import save file; convert columns from strings to lists
        split_list = lambda x: x[1:-1].split(', ')
        data = pd.read_csv(fname, 
                converters={'recovered_times': split_list, 
                            'all_times': split_list},
        )
        
        # Join lists of recovered times and all times
        join_list = lambda x: [float(t) for l in x for t in l if t != '']
        self.recovered_times = join_list(data.recovered_times)
        self.all_times = join_list(data.all_times)

        # Join lists of recovered scales and all scales
        count_scale = lambda x: [data.loc[i,'scale'] for i, l in enumerate(x) 
                for t in l if t != '']
        self.recovered_scales = count_scale(data.recovered_times)
        self.all_scales = count_scale(data.all_times)


    def __call__(self, tstart, scale):
        row = self.data[(self.data['tstart'] == tstart) & (self.data['scale'] == scale)]
        return row['all_times'].iloc[0]


    def hist(self, x_edges, y_edges):
        """Generate 2D histogram of recovery rate according to given bin edges."""

        # 2D histograms for recovered data and total data
        recovered = np.histogram2d(self.recovered_times, self.recovered_scales, 
                [x_edges, y_edges])[0]
        total = np.histogram2d(self.all_times, self.all_scales, 
                [x_edges, y_edges])[0]

        # Calculate recovery rate
        rate_hist = recovered / total

        # Transpose and convert to DataFrame with time increasing along the rows
        # and scale height increasing down the columns. Column and index labels
        # are the lower bound of each bin
        rate_hist = pd.DataFrame(rate_hist.T, index=y_edges[:-1], columns=x_edges[:-1])
        return rate_hist


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', '-i', type=int, default=10000, help='Iterations')
    args = parser.parse_args()

    main(args.iterations)