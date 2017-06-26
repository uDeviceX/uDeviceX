import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print "usage:", sys.argv[0], "<file.h5part>"
    exit(1)

fname = sys.argv[1]

f = h5py.File(fname, "r")

start = 3*len(f) / 4

XS = 64.
YS = 32.
ZS = 16.

dx = 0.5
dy = 2.

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

def rhoavg(x, y, t, x0):

    # select solvent
    #mask = (t == 0)

    # or select all particles
    mask = len(x)*[True]

    dxh = 0.5 * dx
    
    mask = np.logical_and(np.logical_and(mask, (x > x0 - dxh)), (x <= x0 + dxh))

    y = y[mask]
    
    bins = np.linspace(-0.5*YS, 0.5*YS, N, True)
    counts = np.histogram(y, bins=N, range=(0., YS))[0]
    
    return counts / (dx * dy * ZS)

(XS, YS, ZS) = domainsizes()
print "Domain size:", XS, YS, ZS

N = int( YS / dy )

uavg0 = np.zeros(N)
uavg1 = np.zeros(N)
rhoavg0 = np.zeros(N)
rhoavg1 = np.zeros(N)
yavg = np.linspace(-0.5*YS, 0.5*YS, N)
nsteps = 0

for n in f:
    step = name2step(n)
    data = f[n]
    
    if step >= start:
        #print "processing step", step

        x = np.array(data["x"])
        y = np.array(data["y"])
        z = np.array(data["z"])

        u = np.array(data["u"])
        v = np.array(data["v"])
        w = np.array(data["w"])

        t = np.array(data["type"])

        uavg0 += avg(x, y, t, dx*0.5, u)
        uavg1 += avg(x, y, t, XS*0.5, u)

        rhoavg0 += rhoavg(x, y, t, dx*0.5)
        rhoavg1 += rhoavg(x, y, t, XS*0.5)
        
        nsteps += 1
        
uavg0 = (1.0 / nsteps) * uavg0
uavg1 = (1.0 / nsteps) * uavg1

rhoavg0 = (1.0 / nsteps) * rhoavg0
rhoavg1 = (1.0 / nsteps) * rhoavg1


yavg = yavg[uavg0 != 0]
uavg1 = uavg1[uavg0 != 0]
uavg0 = uavg0[uavg0 != 0]

pshear = np.polyfit(yavg, uavg0, 1)

yavg /= YS
uavg0 /= YS*pshear[0]
uavg1 /= YS*pshear[0]


plt.figure(0)

plt.plot(yavg, uavg0, 'ob', label = "boundary")
plt.plot(yavg, uavg1, 'og', label = "object")

pshear = np.polyfit(yavg, uavg0, 1)
plt.plot(yavg, np.polyval(pshear, yavg), '--k', label = "analytic")

plt.xlabel(r"$y/W$")
plt.ylabel(r"$u/\dot{\gamma}W$")
plt.grid()
plt.legend(loc='best')

plt.savefig("vprofile.pdf", transparent=True)
#plt.savefig("vprofile.pgf")


if 0:
    plt.figure(1)
    plt.plot(yavg, rhoavg0, '-+')
    plt.plot(yavg, rhoavg1, '-+')

plt.show()
