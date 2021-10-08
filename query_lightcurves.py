import time
from datetime import datetime
import numpy as np
import os, sys, shutil, glob
import subprocess
from pathlib import Path
from gPhoton import gMap, gAperture as gAp
import pandas as pd
from astropy.coordinates import SkyCoord

np.random.seed()
Nproc = 10

APRAD = 6.
INRAD = 10.
OUTRAD = 15.

LCpath = 'historical_LCs'
gAp_cmd='/home/dubay.11/anaconda3/bin/gAperture'
# gAp_cmd = Path('C:\\Users\\dubay.11\\Anaconda3\\Scripts\\gAperture')

INTERVAL = 60


class Wrapper:
	def __init__(self, name, ra, dec, band):
		self.name = np.str.replace(name, ' ', '_')
		self.ra = ra
		self.dec = dec
		self.band = band
		if band == 'NUV':
			self.csv = LCpath+'/'+self.name+'-NUV.csv'
		elif band == 'FUV':
			self.csv = LCpath+'/'+self.name+'-FUV.csv'
		else:
			raise ValueError('Unexpected band: %s' % band)
	
	def start(self):
		if os.path.exists(self.csv):
			self.done = True
			return
		else:
			self.done = False
		print('Starting query for: %s (band=%s) -> %s' % (self.name, self.band, self.csv))
		cmd = make_cmd(ra=self.ra, dec=self.dec, csvff=self.csv, band=self.band)
		self.starttime = datetime.utcnow()
		self.process = subprocess.Popen(cmd)

	def check(self):
		status = self.process.poll()
		if status is None: #still running
			return
		else:
			self.done = True
			dt = (datetime.utcnow() - self.starttime).total_seconds() / 3600.
			print('\tquery finished %s (%.2f hrs)' % (self.csv, dt))


def main(count=None):

	catalog = pd.read_csv(Path('out/nearby_historical_obs.csv'))
	bands = catalog['band']
	RA, DEC = convert_coords(catalog['ra'], catalog['dec'])

	clean_empty_lcs()


	jobs = []
	for name, ra, dec, band in zip(catalog['sn_name'], RA, DEC, bands):
		if band == 'both':
			for b in ['NUV', 'FUV']:
				job = Wrapper(name, ra, dec, b)
				jobs.append(job)
		else:
			job = Wrapper(name, ra, dec, band)
			jobs.append(job)

	np.random.shuffle(jobs)

	if count is not None:
		jobs = [jobs[i] for i in range(count)]


	running = []
	while True:
		running = check_running(running)
		while len(running) < Nproc and len(jobs) > 0:
			job = jobs.pop(0)
			job.start()
			if not job.done: running.append(job)

		if len(jobs) == 0 and len(running) == 0: break
		time.sleep(INTERVAL)

def make_cmd(ra, dec, csvff, band):

	aper = APRAD/3600.
	ann1 = INRAD/3600.
	ann2 = OUTRAD/3600.

	cmd = [gAp_cmd, '--ra', str(ra), '--dec', str(dec), '--aperture', str(aper), '--inner', str(ann1), '--outer', str(ann2), '--csvfile', csvff, '--band', band]
	return cmd

def check_running(procs):
	keep = []
	for proc in procs:
		proc.check()
		if not proc.done: keep.append(proc)
	return keep

def clean_empty_lcs():
	for ff in glob.glob('historical_LCs/*.csv'):
		Nlines = len([line.strip() for line in open(ff, 'r').readlines()])
		if Nlines <= 1: os.remove(ff)

def convert_coords(ralist, declist):
	coords = SkyCoord(ralist, declist, unit='deg')
	ra = coords.ra.deg
	dec = coords.dec.deg
	return ra, dec


if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--count', '-c', help='Number to run before exiting', default=None, type=int)

	args = parser.parse_args()

	main(count=args.count)
