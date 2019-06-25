#!/usr/bin/env python
import sys
import os
import pickle
import numpy as np

npart = 1000
nturns = 20000

filename_error_table = sys.argv[1]
#qqx, qqy = int(np.round((qx%1) * 100)), int(np.round((qy%1) * 100))

#filename_error_table = "./errors_{qqx}_{qqy}_{eseed:d}".format(
#    qqx=qqx, qqy=qqy, eseed=e_seed)

if os.path.exists(filename_error_table + '_summary.p'):
    sys.exit()

print (filename_error_table)

store = {}

try:
    store['losses'] = np.sum(np.load(filename_error_table + '_alive.npy'))
except FileNotFoundError:
    sys.exit()

try:
    x = np.load(filename_error_table + '_x.npy')
    x = x.reshape((nturns, npart)).T
    store['std_x'] = np.mean(np.std(x, axis=0)[-50:])
except FileNotFoundError:
    sys.exit()

try:
    y = np.load(filename_error_table + '_y.npy')
    y = y.reshape((nturns, npart)).T
    store['std_y'] = np.mean(np.std(y, axis=0)[-50:])
except FileNotFoundError:
    sys.exit()

#import pprint
#pprint.pprint(store)

pickle.dump(store, open(filename_error_table + '_summary.p', 'wb'))
