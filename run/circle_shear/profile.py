import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt

assert len(sys.argv) >= 2

for fid in range(1, len(sys.argv)):
    filename = sys.argv[fid]
    
    f = h5py.File(filename, "r")

    if fid == 1:
        vx = np.array(f['u'])
        vy = np.array(f['v'])
    else:
        vx += np.array(f['u'])
        vy += np.array(f['v'])

vx = vx * (1.0 / (len(sys.argv)-1))
vy = vy * (1.0 / (len(sys.argv)-1))

ZS, YS, XS, dummy = vx.shape

vx = np.reshape(vx, (ZS, YS, XS,))
vy = np.reshape(vy, (ZS, YS, XS,))

# average in z direction
vx = np.mean(vx, (0))
vy = np.mean(vy, (0))

def plot_at_x(v, i):
    #select border only (x = 0)
    v_ = v[:,i]
    
    yy = np.array(range(len(v_)))
    
    yy = yy[np.abs(v_) > 1e-8]
    v_ = v_[np.abs(v_) > 1e-8]
    
    # remove end points
    #v_ = v_[1:-2]
    #yy = yy[1:-2]
    
    #pol = np.polyfit(yy, v_, deg=1)
    
    #print "shear rate =", pol[0]

    plt.plot(yy, v_, '-+', label='x = '+str(ix))
    #plt.plot(yy, np.polyval(pol, yy), '-')
    
plt.figure()

indices = [0, XS/4-1, XS/2-1]

for ix in indices:
     plot_at_x(vx, ix)

plt.xlabel('y')
plt.ylabel('vx')
plt.legend()
plt.show()
