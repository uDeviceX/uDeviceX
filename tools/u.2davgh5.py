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

if argc < 4:
    err("usage: %s <direction [0/1/2]> <[density/u/v/w]> <file0.h5> <file1.h5> ...\n" % argv[0])
    exit(0)

shift(argv)
adir = int(shift(argv))
field = shift(argv)

fff = []
t = 0

for fname in argv:
    f = fopen(fname)
    ff = f[field]
    (nz, ny, nx, nu) = ff.shape
    nn = [nx, ny, nz]

    ff = ff.value
    ff = ff.reshape(nz, ny, nx)
    ff = np.sum(ff, (2-adir)) / (nn[adir])

    if t == 0:
        fff = ff
    else:
        fff = fff + ff

    t += 1

fff = fff * (1.0 / t)

np.savetxt(sys.stdout, fff, "%g");

f.close()
