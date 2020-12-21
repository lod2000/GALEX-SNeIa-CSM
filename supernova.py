from pathlib import Path
import platform
import pandas as pd
import numpy as np
from astropy.time import Time
import astropy.units as u
from utils import *

TYPE_CATALOG_FILE = Path('ref/full_type_catalog.csv')


def main():
    """Import supernovae from OSC, then cut by redshift & classification."""

    osc = pd.read_csv(OSC_FILE, index_col='Name')

    # Remove SNe with disputed classifications
    # types = osc['Type']
    # type_cuts = '(\?|\/|Ib|Ic|II)'
    # osc_reduced = osc[~types.str.contains(type_cuts, regex=True)]
    # print(osc_reduced)

    # Import GALEX observations with before+after epochs
    sample_obs = pd.read_csv(Path('out/sample_obs.csv'))
    sample_sne = sample_obs['sn_name'].drop_duplicates().to_list()
    print(sample_sne[:10])
    sample = osc.loc[sample_sne]
    print(sample)


class Supernova:
    def __init__(self, name, sn_info=[], ref_file='ref/sn_info.csv'):
        """Initialize Supernova by importing reference file."""

        # Import supernova info CSV
        if len(sn_info) == 0:
            sn_info = pd.read_csv(Path(ref_file), index_col='name')

        self.name = name
        self.data = sn_info.loc[name].to_dict()

        self.z = self.data['z']
        self.z_err = self.data['z_err']
        self.dist = self.data['pref_dist'] * u.Mpc
        self.dist_err = self.data['pref_dist_err'] * u.Mpc
        self.disc_date = Time(self.data['disc_date'], format='iso')
        self.a_v = self.data['a_v'] * u.mag


    def __call__(self, key=None):
        """Return value associated with key."""

        if key == None:
            return self.data
        return self.data[key]


    @classmethod
    def from_fname(self, fname, **kwargs):
        """Extract SN name and band from a file name, and return a Supernova 
        object with the GALEX band."""

        sn, band = fname2sn(fname)
        return Supernova(sn, **kwargs)


    def to_fname(self, band, **kwargs):
        """Return filename based on SN name and band."""

        return sn2fname(self.name, band, **kwargs)


if __name__ == '__main__':
    main()
