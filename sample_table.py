import numpy as np
import pandas as pd
from pathlib import Path
import requests
from astropy.coordinates import SkyCoord
from astropy import units as u

BIB_FILE = Path('out/table_references.bib')
CAT_FILE = Path('out/catalog_codes.txt')
LATEX_TABLE_TEMPLATE = Path('ref/table_template.tex')
LATEX_TABLE_FILE = Path('out/sample.tex')
SHORT_TABLE_FILE = Path('out/short_sample.tex')
SHORT_TABLE_LENGTH = 19 # Number of rows to display in short table
pd.set_option('max_colwidth', 1000)

def main():
    # Import data
    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
    columns = ['disc_date', 'epochs_total', 'delta_t_first', 'delta_t_last', 
               'delta_t_next', 'z', 'pref_dist', 'pref_dist_err', 'a_v']
    ref_cols = ['posn_ref', 'z_ref', 'pref_dist_ref']
    table = sn_info[columns]

    # Format coordinates
    coords = SkyCoord(sn_info.galex_ra, sn_info.galex_dec)
    table.insert(1, 'ra',  coords.ra.to_string(sep=':', unit=u.hourangle, pad=True, format='latex'))
    table.insert(2, 'dec', coords.dec.to_string(sep=':', unit=u.deg, alwayssign=True, pad=True, format='latex'))
    # table['dec'] = table['dec'].astype(str).replace('-', '$-$')
    # table['dec'] = table['dec'].replace('+', '$+$')
    # print(table['dec'])

    # Format epochs
    table['delta_t_first'] = -1 * table['delta_t_first']

    # Format redshift and distance
    table['z'] = table['z'].round(5).astype(str)
    table['pref_dist'] = table['pref_dist'].round(0).astype(int)
    table['pref_dist_err'] = table['pref_dist_err'].round(0).astype(int)

    # Format references
    table['refs'] = sn_info[ref_cols].astype('str').agg(';'.join, axis=1)

    # print(table)

    # Check for BibTeX file
    overwrite = True
    if BIB_FILE.is_file():
        over_in = input('Previous bibliography detected. Overwrite? [y/N] ')
        overwrite = (over_in == 'y')

    # Get BibTeX entries and write bibfile
    if overwrite:
        print('Pulling BibTeX entries from ADS...')
        refs = list(sn_info['posn_ref']) + list(sn_info['pref_dist_ref']) + list(sn_info['z_ref'])
        refs = list(dict.fromkeys(refs)) # remove duplicates
        bibcodes = {'bibcode':refs}
        with open(Path('ref/ads_token'), 'r') as file:
            token = file.readline()
        ads_bibtex_url = 'https://api.adsabs.harvard.edu/v1/export/bibtex'
        r = requests.post(ads_bibtex_url, headers={'Authorization': 'Bearer ' + token}, data=bibcodes)
        bibtex = r.json()['export'].replace('A&A', 'AandA') # replace pesky ampersands
        with open(BIB_FILE, 'w') as file:
            file.write(bibtex)

    # Write catalog bibcodes for manual lookup
    sn_info[ref_cols] = sn_info[ref_cols].replace(np.nan, '')
    catalogs = []
    for col in ref_cols:
        catalogs += list(sn_info[sn_info[col].str.contains(':')][col])
    catalogs = list(dict.fromkeys(catalogs))
    catalog_str = '\n'.join(catalogs)
    with open(CAT_FILE, 'w') as file:
        file.write(catalog_str)

    # Output full LaTeX table
    to_latex(table)

    # Output short LaTeX table
    short_table = table.iloc[0:SHORT_TABLE_LENGTH]
    to_latex(short_table, fname=SHORT_TABLE_FILE)


def to_latex(table, fname=LATEX_TABLE_FILE, template=LATEX_TABLE_TEMPLATE):
    """Convert DataFrame to LaTeX table and output to file."""

    latex_table = table.to_latex(na_rep='N/A', index=True, escape=False, 
            formatters={'refs':table_ref})

    # Replace table header and footer with template
    # Edit this file if you need to change the number of columns or description
    with open(template, 'r') as file:
        dt_file = file.read()
        header = dt_file.split('===')[0]
        footer = dt_file.split('===')[1]
    latex_table = header + '\n'.join(latex_table.split('\n')[5:-3]) + footer

    # Write table
    with open(fname, 'w') as file:
        file.write(latex_table)

    return latex_table


def table_ref(bibcodes):
    """
    Formats reference bibcodes for LaTeX table
    Input:
        bibcodes (str): list of reference codes joined with ';'
    """
    bibcodes = bibcodes.replace(';nan', '').split(';')
    bibcodes = list(dict.fromkeys(bibcodes)) # remove duplicates
    return '\citet{' + ','.join([b.replace('A&A', 'AandA') for b in bibcodes]) + '}'


if __name__ == '__main__':
    main()