from query_lightcurves import Wrapper
import pandas as pd
import time, sys
from astropy.coordinates import SkyCoord

INTERVAL = 30 #sec b/w checks

def main(name, band, ra=None, dec=None):

    #parse coords
    if ra is None and dec is None:
        ra, dec = get_coords(name)
    elif 'h' in ra and 'd' in dec:
        coord = SkyCoord(ra, dec)
        ra = coord.ra.deg
        dec = coord.dec.deg
    else:
        ra = float(ra)
        dec = float(dec)

    #load job
    if band == 'NUV':
        job = Wrapper(name, ra, dec, 'NUV')
    elif band == 'FUV':
        job = Wrapper(name, ra, dec, 'FUV')
    else:
        print('Unrecognized band: %s' % band)
        return
    
    #run query
    job.start()
    while True:
        job.check()
        if job.done: break
        time.sleep(INTERVAL)
    return

def get_coords(name):
    fname = None
    if fname is None:
        print('Need to setup name processor, exiting')
        sys.exit()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', help='name of SN', default=None, type=str)
    parser.add_argument('--band', '-b', help='band to query, NUV or FUV', default=None, type=str)
    parser.add_argument('--ra', '-r', help='ra in deg or HHhMMmSS.Ss, optional (can use name lookup)', default=None, type=str)
    parser.add_argument('--dec', '-d', help='dec in deg or DDdMMmSS.Ss, optional', default=None, type=str)

    args = parser.parse_args()
    if args.name is None or args.band is None:
        print('Need a SN name (--name) and UV band (--band)!')
        sys.exit()
    main(args.name, args.band, args.ra, args.dec)
