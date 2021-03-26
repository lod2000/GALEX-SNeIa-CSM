import numpy as np
import pandas as pd
from pathlib import Path
import requests
from astropy.coordinates import SkyCoord
from astropy import units as u

BIB_FILE = Path('tex/table_references.bib')
CAT_FILE = Path('tex/catalog_codes.txt')
LATEX_TABLE_TEMPLATE = Path('ref/table_template.tex')
LATEX_TABLE_FILE = Path('out/table.tex')
SHORT_TABLE_FILE = Path('out/short_table.tex')
SHORT_TABLE_LENGTH = 19 # Number of rows to display in short table

sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
columns = ['epochs_total', 'delta_t_first',
           'delta_t_last', 'delta_t_next', 'z', 'pref_dist', 'pref_dist_err']
table = sn_info[columns]

# Format coordinates
coords = SkyCoord(sn_info.galex_ra, sn_info.galex_dec)
table['galex_ra'] = coords.ra.to_string(sep=':', unit=u.hourangle)[0]
print(coords.dec.to_string()[0])

def get_catalogs(sn_info):
    """
    Get catalog refcodes (denoted by trailing ':' in NED)
    """

    ref_cols = ['posn_ref', 'z_ref', 'pref_dist_ref']
    sn_info[ref_cols] = sn_info[ref_cols].replace(np.nan, '')
    catalogs = []
    for col in ref_cols:
        catalogs += list(sn_info[sn_info[col].str.contains(':')][col])
    catalogs = list(dict.fromkeys(catalogs))
    catalog_str = '\n'.join(catalogs)
    with open(CAT_FILE, 'w') as file:
        file.write(catalog_str)
    return catalogs


def table_ref(bibcodes):
    """
    Formats reference bibcodes for LaTeX table
    Input:
        bibcodes (str): list of reference codes joined with ';'
    """
    bibcodes = bibcodes.replace(';nan', '').split(';')
    bibcodes = list(dict.fromkeys(bibcodes)) # remove duplicates
    return '\citet{' + ','.join([b.replace('A&A', 'AandA') for b in bibcodes]) + '}'