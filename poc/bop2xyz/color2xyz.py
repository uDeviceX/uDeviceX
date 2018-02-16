#!/usr/bin/env python

import os
import sys
import numpy as np


def dump_xyz(p, fc):
    n = np.shape(p)[0]
    x = p[:,0]
    y = p[:,1]
    z = p[:,2]
    u = p[:,3]
    v = p[:,4]
    w = p[:,5]

    fname = "%s.xyz" % (fc)
    f = open(fname,'w')

    # Write header lines
    f.write("%d\n" % n)
    f.write("Generated with color2xyz.py\n")

    # Write data
    for i in range(0,n):
        f.write( "%d %g %g %g %g %g %g\n" % (1, x[i],y[i],z[i], u[i],v[i],w[i]) )

    f.close()


def main():
    fc=sys.argv[1] #file with colors
    fd=sys.argv[2] #file with data

    colors = np.fromfile(fc, dtype=np.uint32) 
    data   = np.fromfile(fd, dtype=np.float32)
    nc = np.shape(colors)[0]
    nd = np.shape(data)[0]
    assert(nc==nd/6)

    p = np.reshape(data,(nc,6))

    p_inside = p[colors==1,:]

    dump_xyz(p_inside, fc)


if __name__ == '__main__':
    main()
