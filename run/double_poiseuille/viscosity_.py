# Compute fluid viscosity
# run: python viscosity.py <datafile> <acc>
#
# for daint: module load h5py
import h5py
import sys
import numpy as np

assert len(sys.argv) == 3

filename = sys.argv[1]
f = h5py.File(filename, "r")
vx = np.array(f['u'])

# average in x and z directions
vx = np.mean(vx, (0, 2))
vx = vx.reshape((-1,))
vx1 = vx[:int(len(vx)/2)]
vx2 = vx[int(len(vx)/2):]
av1 = abs(np.mean(vx1));
av2 = abs(np.mean(vx2));
av = (av1+av2)/2.
# compute viscosity
# Ref: Backer et al., J. Chem. Phys., 2005
# Vav = (rho * acc * D^2) / (12*eta)
#   rho: density
#   acc: acceleration in direction of flow
#   D: distance between 1 (!) Poiseuille profile
#   eta: fluid viscosity
Vav = av
rho = 10.
acc = float(sys.argv[2])
D = 16.
eta = (rho * acc * D*D) / (12.*Vav)

print(eta)
