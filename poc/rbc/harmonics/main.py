#!/usr/bin/python2

from scipy.special import sph_harm as Y

def yre(m, n, t, p): return Y(m, n, t, p).real
def yim(m, n, t, p): return Y(m, n, t, p).imag

m = 1
n = 4
t = 0.1
p = 3

print yim(m, n, t, p)
