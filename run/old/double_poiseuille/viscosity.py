import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt

driving_force = float(sys.argv[1])

nfiles = len(sys.argv)-2

assert nfiles >= 1

mu = [0.] * nfiles
t = range(nfiles)

c = 0

for filename in sys.argv[2:]:

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
        plt.figure(0)
        plt.plot(yyl, vxl, '+')
        plt.plot(yyr, vxr, '+')
        plt.plot(yyl, np.polyval(poll, yyl), '-')
        plt.plot(yyr, np.polyval(polr, yyr), '-')

    def visc(coeff):
        rho = 10.
        return rho * driving_force / (2*coeff)

    mu[c] = 0.5 * (visc(poll[0]) + visc(-polr[0]))

    c += 1

plt.figure(0)
plt.plot(t, mu)
plt.show()

print mu[-1]
