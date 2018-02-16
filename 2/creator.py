#!/usr/bin/env python
#
# DPD parameters:
# Given: kb, G*
#
# 1. Kinetic energy ~ Bending energy: kc = 70kBT => get system kBT
# 2. Ma (Vthermal): Vth = 0.3*c = sqrt(3*kBT) => get c
# 3. Ma (compressibility): Vmax = 0.3*c => get aij
# 4. Ma (compressibility): Vmax = 0.3*c = sh*Ro => get sh
# 5. Wi: G* = sh*eta0 => get eta0

import os
import sys
import math
import numpy as np

def dpd_params(fac, G, kb, rho, Area):
    kc = fac * kb * math.sqrt(3)/2. #approximation!
    kbt = kc/70. #based on exp. data: kc = 70kBT

    Ma = 0.6
    Vth = math.sqrt(3.*kbt)
    c = Vth/Ma
    a = 0.101 #from fit: GW,1997
    aij = (c*c-kbt) / (2.*a*rho)

    Ro = math.sqrt(Area/(4.*math.pi))
    Vmax = Ma*c
    sh = Vmax/2.

    Const = Ro*Ro*Ro * 2. / (kb * math.sqrt(3))
    eta0 = G/(sh*Const)
    a=3.09; b=7.24 #fit for nd=10!
    gij = (eta0 - b)/a

    print "For G=%7.2f, kb=%6.2f:   aij=%7.3f,  gij=%10.4f,  sh=%7.4f,  kbt=%7.4f" % (G, kb, aij, gij, sh, kbt)
    return aij, gij, kbt, sh


def main():
    # Model Params: kb, gC, x0, ks
    # System Input: G

    XS = 24
    YS = 56
    ZS = 52

    G_  = np.linspace(50., 1000., 3)

    KB_ = np.array([10., 50., 100.])
    GC_ = np.array([10., 100., 500., 1000., 5000.])
    KS_ = np.array([3.0, 5.0, 8.8, 10.0])
    X0_ = np.array([1./2.2])

    #sfree_ = np.array(["0", "sph", "rbc"]) #3
    sfree_ = np.array(["rbc"])

    nd = 10
    NV_ = 1986
    sc = 1

    NV =np.array([498,      642,      1986,     2562])    # number of vertices on rbc
    A0 =np.array([59.036,   80.0594,  944.577,  1593.56]) # exact initial areas for given meshes
    V0 =np.array([27.3985,  43.3586,  1753.5,   3853.96]) # exact initial volumes for given meshes
    A0s=np.array([59.036,   80.0594,  245.684,  319.489]) # scales rbc size (smaller)
    V0s=np.array([27.3985,  43.3586,  232.604,  345.97])

    for i in range(0,len(NV)):
        if NV[i] != NV_:
            continue

        inv  = NV[i]
        if sc==1:
            a0   = A0s[i]
            v0   = V0s[i]
        else:
            a0   = A0[i]
            v0   = V0[i]


    for KB in KB_:
        for G in G_:
            for GC in GC_:
                for KS in KS_:
                    for X0 in X0_:
                        for sfree in sfree_:
                            [aij, gij, kbt, SH] = dpd_params(1.0, G, KB, nd, a0)
                            RNAME="F2_Ma0.6_kd4900_sfree%s_nv%d_xo%g_ks%g_kb%g_gc%g_G%g_SH%g_gij%g_aij%g_kbt%g" % (sfree, inv, X0, KS, KB, GC, G, SH, gij, aij, kbt)
                            print RNAME

                            # Print params in txt file
                            f = open("params.txt", 'w')
                            f.write( "%d %d %d %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % (inv, XS, YS, ZS, aij, gij, aij, gij/3., GC, kbt, KB, KS, X0, a0, v0, SH))
                            f.close()
 
                            cmd="sh ./launch.sh %s %d %s" % (RNAME, sc, sfree)
                            os.system(cmd)


if __name__ == '__main__':
    main()
