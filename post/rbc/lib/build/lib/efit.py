import numpy as np
import math
import os

def dbl(x, y, z):
    return \
        [x.astype("double"), y.astype("double"), z.astype("double")]

def ellipsoid0(x, y, z):
    [x, y, z] = dbl(x, y, z)
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

    # calculate difference of the fitted points from the actual data normalized by the conic radii
    d = np.array([x - center[0], y - center[1], z - center[2]]) # shift data to origin
    d = np.dot(d.T, evecs) # rotate to cardinal axes of the conic
    d = np.array([d[:,0] / radii[0], d[:,1] / radii[1], d[:,2] / radii[2]]).T # normalize to the conic radii
    chi2 = np.sum(np.abs(1 - np.sum(d**2 * np.tile(sgns, (d.shape[0], 1)), axis=1)))

    return center, radii, evecs, v, chi2

def ellipsoid(X):
    x = X[:,0]; y = X[:,1]; z = X[:,2]
    return ellipsoid0(x, y, z)

def rot0(x0, y0, phi):
    c, s = np.cos(phi), np.sin(phi)
    x = c*x0 - s*y0
    y = s*x0 + c*y0
    return [x, y]

def rot(x, y, vx, vy, phi):
    [x, y]   = rot0(x, y, phi)
    [vx, vy] = rot0(vx, vy, phi)
    return x, y, vx, vy

def keller_phi0(x, y, vx, vy, q, phi):
    [x, y, vx, vy] = rot(x, y, vx, vy, -phi) # clock-wise
    fr = keller(x, y, vx, vy, q)
    res = keller_res(x, y, vx, vy, q,   fr)
    return fr, res

def keller_phi(x, y, vx, vy, q):
    n = 100
    m_res = 1e42
    for phi in np.linspace(0, np.pi/2, n):
        fr, res = keller_phi0(x, y, vx, vy, q,  phi)
        if res < m_res:
            m_res, m_fr, m_phi = res, fr, phi
    return m_fr, m_phi, m_res


def keller_res(x, y, vx0, vy0, q,  fr):
    """
    Mean squarer residues of the Keller-Skalak fit
    q = a/b, [vx, vy] ~ [fr*a/b*y, -fr*b/a*x]
    """
    vx =  fr*q    *y
    vy = -fr*(1/q)*x
    dvx = vx - vx0
    dvy = vy - vy0
    return np.mean(dvx**2) + np.mean(dvy**2)

def keller(x, y, vx, vy, q):
    """
    Mean squarer fit of Keller-Skalak frequency (`fr').
    q = a/b, [vx, vy] ~ [fr*a/b*y, -fr*b/a*x]
    """
    sm = np.sum
    svxy, svyx = sm(vx*y), sm(vy*x)
    sxx,   syy = sm( x*x), sm( y*y)
    q2 = q**2
    q4 = q**4
    fr = (q*(q2*svxy-svyx))/(q4*syy+sxx)
    return fr

# Ref:
# http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
