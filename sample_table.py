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

    # Add table note about large host offsets
    table['offset_note'] = ''
    table.loc[sn_info['offset'] > 30, 'offset_note'] = '\tablenotemark{a}'
    table['dec'] = table[['dec', 'offset_note']].agg(''.join, axis=1)
    table.drop('offset_note', axis=1, inplace=True)

    # Format epochs
    table['delta_t_first'] = -1 * table['delta_t_first']

    # Format redshift and distance
    table['z'] = table['z'].round(5).astype(str)
    table['pref_dist'] = table['pref_dist'].round(0).astype(int)
    table['pref_dist_err'] = table['pref_dist_err'].round(0).astype(int)

    # Format references
    table['refs'] = sn_info[ref_cols].astype('str').agg(';'.join, axis=1)

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
    print('Output full table')

    # Output short LaTeX table
    short_table = table.iloc[0:SHORT_TABLE_LENGTH]
    to_latex(short_table, fname=SHORT_TABLE_FILE)
    print('Output short table to %s' % SHORT_TABLE_FILE)

    # Output CSV for machine-readable table
    # Machine-readable columns
    mr_cols = ['Disc.Y', 'Disc.M', 'Disc.D', 'r_Disc.D', 'RAh', 'RAm', 'RAs',
            'DE-', 'DEd', 'DEm', 'DEs', 'f_DEs', 'Obs', 'tFirst', 'tLast', 'tNext', 
            'z', 'r_z', 'Dist', 'e_Dist', 'r_Dist', 'VExt']
    mr_tab = pd.DataFrame(columns=mr_cols, index=pd.Series(table.index, name='Name'))
    # Split discovery date into Y, M, D
    mr_tab[['Disc.Y', 'Disc.M', 'Disc.D']] = table.disc_date.str.split('-', expand=True)
    # Discovery reference
    mr_tab['r_Disc.D'] = sn_info['posn_ref']
    # Split RA
    mr_tab[['RAh', 'RAm', 'RAs']] = table.ra.str.replace('$', '').str.split(':', expand=True)
    # Split Dec
    mr_tab[['DEd', 'DEm', 'DEs']] = table.dec.str.replace('$', '').str.split(':', expand=True)
    mr_tab['DE-'] = mr_tab.DEd.str[:1]
    mr_tab['DEd'] = mr_tab.DEd.str[1:]
    # Offset flag
    mr_tab[['DEs', 'f_DEs']] = mr_tab.DEs.str.split('\tablenotemark{', expand=True)
    mr_tab['f_DEs'] = mr_tab.f_DEs.str.replace('}', '')
    # GALEX observations relative to disc date
    mr_tab[['Obs', 'tFirst', 'tLast', 'tNext']] = table[['epochs_total', 
            'delta_t_first', 'delta_t_last', 'delta_t_next']]
    # Redshift, distance and distance error
    mr_tab[['z', 'Dist', 'e_Dist']] = table[['z', 'pref_dist', 'pref_dist_err']]
    # References
    mr_tab[['r_z', 'r_Dist']] = sn_info[['z_ref', 'pref_dist_ref']]
    # Extinction
    mr_tab['VExt'] = table['a_v']
    
    mr_tab.to_csv(Path('out/sample_mr.csv'))
    print('Output machine-readable CSV')
    print(mr_tab)


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