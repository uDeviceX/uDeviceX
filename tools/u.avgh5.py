#!/usr/bin/env python2 

import sys
import numpy as np
import h5py as h5

def err(s): sys.stderr.write(s)
def shift(a):
    return a.pop(0)
def fopen(n):
    try:
        f = h5.File(fname, "r")
    except IOError:
        err("u.avgh5: fails to open <%s>\n" % n)
        sys.exit(2)
    return f

argv = sys.argv
argc = len(argv)

if argc < 5:
    err("usage: %s <dump_coords [0/1]> <[density/u/v/w]> <remaining dir [0/1/2]> <file0.h5> <file1.h5> ...\n" % argv[0])
    exit(0)

shift(argv);
dump_coords = int(shift(argv))
field = shift(argv)
adir = int(shift(argv))

fff = []
i = 0

for fname in argv:
    f = fopen(fname)
    ff = f[field]
    (nz, ny, nx, nu) = ff.shape
    nn = [nx, ny, nz]
    
    av1=(adir+1)%3
    av2=(adir+2)%3
    
    ff = ff.value
    ff = ff.reshape(nz, ny, nx)
    ff = np.sum(ff, (2-av1,2-av2)) / (nn[av1] * nn[av2])

    if i == 0:
        fff = ff
    else:
        fff = fff + ff
        
    nc = nn[adir]
    cc = np.arange(nc) + 0.5 - nc / 2
    i += 1

fff = fff * (1.0 / i)

if dump_coords:
    fff = fff.reshape(nc, 1)
    cc = cc.reshape(nc, 1)
    np.savetxt(sys.stdout, np.concatenate((cc, fff), axis=1), "%g %g");
else:
    np.savetxt(sys.stdout, fff, "%g");

f.close()
