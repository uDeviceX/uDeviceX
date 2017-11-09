#!/usr/bin/env python2 

import sys
import numpy as np
import h5py as h5

def shift(a):
    return a.pop(0)

argv = sys.argv
argc = len(argv)

if argc != 5:
    print "usage: %s <dump_coords [0/1]> <field [density/u/v/w]> <remaining dir [0/1/2]> <file.h5>" % argv[0]
    exit(1)

shift(argv);
dump_coords = int(shift(argv))
field = shift(argv)
adir = int(shift(argv))
fname = shift(argv);
    
f = h5.File(fname, "r")

ff = f[field]
(nz, ny, nx, nu) = ff.shape
nn = [nx, ny, nz]

av1=(adir+1)%3
av2=(adir+2)%3

ff = ff.value
ff = ff.reshape(nz, ny, nx)
ff = np.sum(ff, (2-av1,2-av2)) / (nn[av1] * nn[av2])

nc = nn[adir]
cc = np.arange(nc) + 0.5 - nc / 2

if dump_coords:
    ff = ff.reshape(nc, 1)
    cc = cc.reshape(nc, 1)
    np.savetxt(sys.stdout, np.concatenate((cc, ff), axis=1), "%g %g");
else:
    np.savetxt(sys.stdout, ff, "%g");

f.close()
