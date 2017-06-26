import numpy as np
import sys

G = float(sys.argv[1])
R = 5.0
rhodpd = 10
gammadpd = 8
kBT = 0.0444
rc = 1
U = R * G
Dcoeff = 45. / (2 * np.pi * gammadpd * rhodpd * rc**3)
D = Dcoeff * kBT

print "Pe for kBT =", kBT, " :", R * U / D

Pe = 1000.
print R*R/(Dcoeff*Pe)
print "kBT for Pe =", Pe, " :", R*U/(Dcoeff*Pe)

