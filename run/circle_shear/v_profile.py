import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

fname = sys.argv[1]

f = h5py.File(fname, "r")

start = 3*len(f) / 4

XS = 64.
YS = 32.
ZS = 16.

dx = 0.25
dy = 0.25

def name2step(n):
    return int (n.split("#")[1]) 

def domainsizes():
    data0 = f['Step#0']
    XS = np.ceil(max(np.array(data0['x'])))
    YS = np.ceil(max(np.array(data0['y'])))
    ZS = np.ceil(max(np.array(data0['z'])))
    return XS, YS, ZS

def avg(x, y, t, x0, field):

    # select solvent
    #mask = (t == 0)

    # or select all particles
    mask = len(x)*[True]

    dxh = 0.5 * dx
    
    mask = np.logical_and(np.logical_and(mask, (x > x0 - dxh)), (x <= x0 + dxh))

    y = y[mask]
    field = field[mask]
    
    bins = np.linspace(-0.5*YS, 0.5*YS, N, True)
    counts = np.histogram(y, bins=N, range=(0., YS))[0]
    sums = np.histogram(y, bins=N, range=(0., YS), weights=field)[0]

    return sums / np.maximum(1, counts)

(XS, YS, ZS) = domainsizes()
print "Domain size:", XS, YS, ZS

N = int( YS / dy )

plt.figure(0)
uavg0 = np.zeros(N)
uavg1 = np.zeros(N)
yavg = np.linspace(-0.5*YS, 0.5*YS, N)
nsteps = 0

for n in f:
    step = name2step(n)
    data = f[n]
    
    if step >= start:
        print "processing step", step

        x = np.array(data["x"])
        y = np.array(data["y"])
        z = np.array(data["z"])

        u = np.array(data["u"])
        v = np.array(data["v"])
        w = np.array(data["w"])

        t = np.array(data["type"])

        uavg0 += avg(x, y, t, dx*0.5, u)
        uavg1 += avg(x, y, t, XS*0.5, u)
        
        nsteps += 1
        
uavg0 = (1.0 / nsteps) * uavg0
uavg1 = (1.0 / nsteps) * uavg1

plt.plot(yavg, uavg0, '-+')
plt.plot(yavg, uavg1, '-+')
plt.plot(yavg, yavg*0.05, '-')
#plt.plot(yavg, yavg*0.1, '-')

plt.show()
