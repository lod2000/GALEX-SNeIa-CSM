from pathlib import Path
import platform


def fname2sn(fname):
    """Extract SN name and band from a file name."""

    fname = Path(fname)
    split = fname.stem.split('-')
    sn_name = '-'.join(split[:-1])
    band = split[-1]
    # Windows replaces : with _ in some file names
    if 'CSS' in sn_name or 'MLS' in sn_name:
        sn_name.replace('_', ':', 1)
    sn_name.replace('_', ' ')
    return sn_name, band


def sn2fname(sn_name, band, suffix='.csv'):
    """Convert SN name and GALEX band to a file name, e.g. for a light curve CSV."""

    fname = '-'.join((sn_name, band)) + suffix
    fname = fname.replace(' ', '_')
    # Make Windows-friendly
    if (platform.system() == 'Windows') or ('Microsoft' in platform.release()):
        fname = fname.replace(':', '_')
    return Path(fname)
