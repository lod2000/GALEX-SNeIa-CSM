# GALEX-SNeIa-CSM

A repository for scripts used to search for circumstellar medium (CSM) interaction
in Type Ia supernovae (SNe Ia) observed by the *Galaxy Evolution Explorer* (*GALEX*)
spacecraft.

## Accessing *GALEX* Light Curves

Functions to import and plot *GALEX* light curves and run the CSM detection
algorithm are in ``light_curve.py``. In the command line, specify the name of
the supernova to target. Use the ``--detect`` flag to run the detection algorithm
on both FUV and NUV bands if possible. For example,

    python light_curve.py SN2007on --detect --sigma 5 3 --count 1 3

will run the detection algorithm on light curves for SN 2007on, specifying that
a detection should be flagged if there is one data point above 5 sigma significance
or at least three greater than 3 sigma significance.
