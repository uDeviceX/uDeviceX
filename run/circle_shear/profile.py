import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt

assert len(sys.argv) == 2

filename = sys.argv[1]

f = h5py.File(filename, "r")

vx = np.array(f['u'])
vy = np.array(f['v'])

# average in z direction
vx = np.mean(vx, (2))
vy = np.mean(vy, (2))

def plot_at_x(v, i):
    #select border only (x = 0)
    v_ = v[i,:,:]
    v_ = v_.reshape((-1,))
    
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

for ix in [0, 7, 15]:
     plot_at_x(vx, ix)

plt.xlabel('y')
plt.ylabel('vx')
plt.legend()
plt.show()
