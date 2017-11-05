import numpy as np
import math
from  plyfile import PlyData
import os

# https://github.com/minillinim/ellipsoid
def ellipsoid_plot(center, radii, rotation, ax, plotAxes=False, cageColor='b', cageAlpha=0.2):
    """Plot an ellipsoid"""

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in xrange(len(x)):
        for j in xrange(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    if plotAxes:
        # make some purdy axes
        axes = np.array([[radii[0],0.0,0.0],
                         [0.0,radii[1],0.0],
                         [0.0,0.0,radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)


        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cageColor)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)


# http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
# for arbitrary axes
def ellipsoid_fit(X):
    X = X.astype('double')
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    D = [ x * x + y * y - 2 * z * z,
          x * x + z * z - 2 * y * y,
          2 * x * y,
          2 * x * z,
          2 * y * z,
          2 * x,
          2 * y,
          2 * z,
          1 + 0 * x ]  # ndatapoints x 9 ellipsoid parameters
    D = np.array(D)

    # solve the normal system of equations
    d2 = x * x + y * y + z * z # the RHS of the llsq problem (y's)
    d2 = d2.reshape((d2.shape[0], 1))
    Q = np.dot(D, D.T)
    b = np.dot(D, d2)
    u = np.linalg.solve(Q, b)  # solution to the normal equations

    v = np.zeros((u.shape[0]+1, u.shape[1]))
    v[0] = u[0] +     u[1] - 1
    v[1] = u[0] - 2 * u[1] - 1
    v[2] = u[1] - 2 * u[0] - 1
    v[3:10] = u[2:9]

    A = np.array([v[0], v[3], v[4], v[6],
                  v[3], v[1], v[5], v[7],
                  v[4], v[5], v[2], v[8],
                  v[6], v[7], v[8], v[9]]).reshape((4, 4))

    center = np.linalg.solve(-A[:3,:3], v[6:9])
    T = np.eye(4)
    T[3,:3] = center.T
    center = center.reshape((3,))
    R = T.dot(A).dot(T.conj().T)
    evals, evecs = np.linalg.eig(R[:3,:3] / -R[3,3])

    # sort eigenvalues
    idx = np.argsort(evals); evals = evals[idx]; evecs = evecs[:, idx]
    sgns = np.sign(evals)
    radii = np.sqrt(sgns / evals)

    # orient the eigenvectors so that they are aligned with the axes
    if np.dot(evecs[:,0], np.array([1, 0, 0])) < 0: evecs[:,0] *= -1
    if np.dot(evecs[:,1], np.array([0, 1, 0])) < 0: evecs[:,1] *= -1
    if np.dot(evecs[:,2], np.array([0, 0, 1])) < 0: evecs[:,2] *= -1

    # calculate difference of the fitted points from the actual data normalized by the conic radii
    d = np.array([x - center[0], y - center[1], z - center[2]]) # shift data to origin
    d = np.dot(d.T, evecs) # rotate to cardinal axes of the conic
    d = np.array([d[:,0] / radii[0], d[:,1] / radii[1], d[:,2] / radii[2]]).T # normalize to the conic radii
    chi2 = np.sum(np.abs(1 - np.sum(d**2 * np.tile(sgns, (d.shape[0], 1)), axis=1)))

    return center, radii, evecs, v, chi2


# dump x, y, z coordinates to file
def ellipsoid_dump(fname, rot, radii):
    a = radii[0]; b = radii[1]; c = radii[2]

    ndump = 100
    uu = np.linspace(0, 2*np.pi, ndump)
    vv = np.linspace(0,   np.pi, ndump)
    [uu, vv] = np.meshgrid(uu, vv); n = uu.size
    uu = uu.reshape(n); vv = vv.reshape(n)
    xx = np.zeros(n); yy = np.zeros(n); zz = np.zeros(n)

    for i in range(n):
        u = uu[i]; v = vv[i]
        x = a*np.cos(u)*np.sin(v) # u: [0, pi]
        y = b*np.sin(u)*np.sin(v) # v: [0, pi]
        z = c*np.cos(v)

        r = np.array([x, y, z])
        r = np.dot(r, rot.T)
        r = r.reshape(3)

        xx[i] = r[0]; yy[i] = r[1]; zz[i] = r[2]

    with open(fname, 'w') as f:
        f.write('x y z sc\n')
        sc = np.zeros(n) # fake scalar
        for i in range(n):
            f.write('%g %g %g %g\n' % (xx[i], yy[i], zz[i], sc[i]))


def ellipsoid_dump_ply(fname, rot, radii):
    ply = PlyData.read(os.path.expanduser('~/.udx/sphere.ply'))
    vertex = ply['vertex']
    xyz = np.array([vertex[p] for p in ('x', 'y', 'z')]).T
    sc = radii / np.max(xyz)
    xyz *= np.tile(sc, (xyz.shape[0], 1))
    xyz = np.dot(xyz, rot.T)
    vertex['x'] = xyz[:, 0]; vertex['y'] = xyz[:, 1]; vertex['z'] = xyz[:, 2]
    ply.write(fname)


# 1) read ply from 'ip', fit an ellipsoid
# 2) dump ellipsoid into 'oe'
def fit_ellipsoid_ply(ip):
    ply = PlyData.read(ip)
    vertex = ply['vertex']
    xyz = np.array([vertex[p] for p in ('x', 'y', 'z')]).T
    uvw = np.array([vertex[p] for p in ('u', 'v', 'w')]).T
    center, radii, rot, v, chi2 = ellipsoid_fit(xyz)
    return center, rot, radii, chi2, xyz, uvw
