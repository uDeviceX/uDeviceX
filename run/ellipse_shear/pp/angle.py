import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print "Usage:", sys.argv[0], "<solid_diag.txt>"
    exit(1)

fname = sys.argv[1]

data = np.loadtxt(fname)

tstart = len(data[:,0]) / 4

t   = data[tstart:,0]
omz = data[tstart:,9]
e0x = data[tstart:,16]
e0y = data[tstart:,17]

phi = np.arctan2(e0x, e0y)

#e1x = data[tstart:,19]
#e1y = data[tstart:,20]

#phi = np.arctan2(e1x, e1y)

phi = phi / np.pi

dphi = np.diff(phi)
jumps = np.where(abs(dphi) > 0.5)[0] + 1

tint = np.split(t  , jumps)
phis = np.split(phi, jumps)

tint = tint[1:-1]
phis = phis[1:-1]

def getT(t_):
    return t_[-1] - t_[0]

Ts = [getT(t_) for t_ in tint]
T = np.mean(Ts)
sigmaT = np.sqrt(np.var(Ts))

print "T =", T, "+-", sigmaT
#print "om = ", 2*np.pi/T, "+-", sigmaT*2*np.pi/(T*T)

print "GT =", 0.0125*T

def rescale(t_):
    t_ = t_ - t_[0]
    t_ = t_ / t_[-1]
    return t_

plt.figure(0)

tth = np.linspace(0, T, 1000)
phith = np.arctan(2*np.tan(2*np.pi*tth/T))
dphi = np.diff(phith)
jumps = np.where(abs(dphi) > 0.5)[0] + 1
phith[:jumps[0]] -= np.pi
phith[jumps[-1]:] += np.pi

plt.plot(tth/T, phith/np.pi, '-k', label="Jeffery")

idint = -1
plt.plot(rescale(tint[idint]), phis[idint], '--b', label="DPD")

plt.grid()
plt.xlabel(r"$t/T$")
plt.ylabel(r"$\phi/\pi$")
plt.legend(loc='best')
plt.savefig("angle_ellipse.pdf", transparent = True)

plt.show()

