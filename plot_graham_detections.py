import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter, MultipleLocator
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import *
from CSMmodel import CSMmodel
from injection_recovery_graham import import_graham_data


graham_detections = import_graham_data('ref/Graham_detections.csv', sigma=1)
print(graham_detections)

iterations = 10000
tstart_lims = [TSTART_MIN, TSTART_MAX]
scale_lims = [SCALE_MIN, SCALE_MAX]
params = gen_params(iterations, tstart_lims, scale_lims, log=True)

for sn in graham_detections.index:
    # Isolate detection info
    detection = graham_detections.loc[sn]
    t = detection['Rest Phase']
    lum = detection['Luminosity [erg/s/A]']
    lum_err = detection['Luminosity Limit']

    # Maximum interaction start time: time of observation
    tstart_max = t

    # Minimum interaction start time, assuming observation is on the plateau
    tstart_min = t - WIDTH # default CSM plateau width, 250 d