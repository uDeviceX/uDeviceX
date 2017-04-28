#!/usr/bin/env python-wrap

import struct
import sys, os
import numpy as np

if len(sys.argv) != 2:
    print "usage:", sys.argv[0], "<file.bop>"
    exit(1)

fh = open(sys.argv[1], "r")
n = long(fh.readline().rstrip())
fvname = fh.readline().rstrip().split()[1]
fh.close()

fvname = os.path.dirname(sys.argv[1]) + '/' + fvname

fv = open(fvname, "rb")
rawdata = fv.read()
fv.close()

data = np.array(struct.unpack("f"*n*6, rawdata))

data = data.reshape((n, 6))

np.savetxt(sys.stdout, data[:,:3], "%.6e")
