import sys

G = float(sys.argv[1])
R = 4.
mu = 2.67
rho = 10.

L = 2 * R
U = L * G

print rho * L * U / mu
