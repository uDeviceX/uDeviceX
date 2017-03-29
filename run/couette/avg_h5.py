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

yy = yy[np.abs(vx) > 1e-8]
vx = vx[np.abs(vx) > 1e-8]

# remove end points
vx = vx[1:-2]
yy = yy[1:-2]

pol = np.polyfit(yy, vx, deg=1)

print "shear rate =", pol[0]

# for i in range(len(vx)):
#     print yy[i], vx[i]

plt.figure()
plt.plot(yy, vx, '+')
plt.plot(yy, np.polyval(pol, yy), '-')
plt.show()
