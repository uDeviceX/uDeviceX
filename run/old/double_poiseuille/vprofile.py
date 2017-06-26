import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt

nfiles = len(sys.argv)-1

assert nfiles >= 1

vxl = np.array([])
vxr = np.array([])
yyl = np.array([])
yyr = np.array([])

for filename in sys.argv[1:]:

    f = h5py.File(filename, "r")
    
    vx = np.array(f['u'])

    # average in x and z directions
    vx = np.mean(vx, (0, 2))
    vx = vx.reshape((-1,))
    
    yy = np.array(range(len(vx)))

    if len(vxl) == 0:
        vxl = vx[:len(vx)/2]
        yyl = yy[:len(yy)/2]
    
        vxr = vx[len(vx)/2:]
        yyr = yy[len(yy)/2:]

    else:
        vxl += vx[:len(vx)/2]
        vxr += vx[len(vx)/2:]
    
vxl /= nfiles
vxr /= nfiles

scale = 1./max(abs(vxr))

vxl *= scale
vxr *= scale


W = float(max(yyr)) + 1
yyl = (yyl+0.5)/W - 0.5
yyr = (yyr+0.5)/W - 0.5

poll = np.polyfit(yyl, vxl, deg=2)
polr = np.polyfit(yyr, vxr, deg=2)

plt.figure(1)
plt.plot(yyl, vxl, '+b', label = "DPD")
plt.plot(yyr, vxr, '+b')

yl = np.linspace(-0.5, 0, 100)
yr = np.linspace(0, 0.5, 100)

plt.plot(yl, np.polyval(poll, yl), '-k', label = "Quadratic fit")
plt.plot(yr, np.polyval(polr, yr), '-k')

plt.xlabel(r"$y/W$")
plt.ylabel(r"$v/v_{max}$")

plt.xlim([-0.5, 0.5])
plt.grid()
plt.legend(loc='upper left')

plt.savefig("vprofile.pdf", transparent=True)

plt.show()
