import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt

assert len(sys.argv) == 2

filename = sys.argv[1]

f = h5py.File(filename, "r")

vx = np.array(f['u'])

# average in x and z directions
vx = np.mean(vx, (0, 2))
vx = vx.reshape((-1,))

yy = np.array(range(len(vx)))

vxl = vx[:len(vx)/2]
yyl = yy[:len(yy)/2]

poll = np.polyfit(yyl, vxl, deg=2)

vxr = vx[len(vx)/2:]
yyr = yy[len(yy)/2:]

polr = np.polyfit(yyr, vxr, deg=2)

if 0:
    plt.figure()
    plt.plot(yyl, vxl, '+')
    plt.plot(yyr, vxr, '+')
    plt.plot(yyl, np.polyval(poll, yyl), '-')
    plt.plot(yyr, np.polyval(polr, yyr), '-')
    plt.show()

def visc(coeff):
    hydrostatic_a = 0.02
    return hydrostatic_a / (2*coeff)

print 0.5 * (visc(poll[0]) + visc(-polr[0]))
