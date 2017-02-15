#include "dpd-forces.h"
#include <cstdio>
#include <mpi.h>
#include <cuda-dpd.h>
#include "common.h"

__device__ bool inbox(float x, float y, float z,
        float xl=-3, float xh=3,
        float yl=-3, float yh=3,
        float zl=-3, float zh=3) {
    return xl < x && x < xh && yl < y && y < yh  && zl < z && z < zh;
}

__device__ float3 compute_dpd_force_traced(int type1, int type2,
        float3 pos1, float3 pos2, float3 vel1, float3 vel2, float myrandnr) {
    /* return the DPD interaction force based on particle types
     * type: 0 -- outer solvent, 1 -- inner solvent, 2 -- membrane, 3 -- wall */

    /************** User-defined constants ***********/
    // All particles: DPD interactions
    const float gammadpd[4] = {8., 8., 8., 8.};  // default: 8
    const float aij[4] = {4. / RC_FX, 4. / RC_FX, 4. / RC_FX, 4. / RC_FX};  // default: 4
    const float gammadpd_pair = 0.5 * (gammadpd[type1] + gammadpd[type2]);
    const float sigmaf_pair = sqrt(2*gammadpd_pair*kBT / dt);

    // RBC-RBC: Lennard-Jones interactions
    const float ljsigma = 0.3;  // default: 0.3
    const float ljepsilon = 1.0 / (RC_FX*RC_FX);  // default: 1
    /********** End of User-defined constants ********/


    const float _xr = pos1.x - pos2.x;
    const float _yr = pos1.y - pos2.y;
    const float _zr = pos1.z - pos2.z;

    const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
    const float invrij = rsqrtf(rij2);
    const float rij = rij2 * invrij;
    if (rij2 >= 1)
        return make_float3(0, 0, 0);

    const float argwr = 1.f - rij;
    const float wr = viscosity_function<-VISCOSITY_S_LEVEL>(argwr);

    const float xr = _xr * invrij;
    const float yr = _yr * invrij;
    const float zr = _zr * invrij;

    const float rdotv =
        xr * (vel1.x - vel2.x) +
        yr * (vel1.y - vel2.y) +
        zr * (vel1.z - vel2.z);

    float strength = (-gammadpd_pair * wr * rdotv + sigmaf_pair * myrandnr) * wr;
    if (type1 == 2 && type2 == 2) {  // membrane contact
        const float invr2 = invrij * invrij;
        const float t2 = ljsigma * ljsigma * invr2;
        const float t4 = t2 * t2;
        const float t6 = t4 * t2;
        const float lj = min(1e4f, max(0.f, ljepsilon * 24.f * invrij * t6 * (2.f * t6 - 1.f)));
        strength += lj;
    } else {
        const float aij_pair = 0.5 * (aij[type1] + aij[type2]);
        strength += aij_pair * argwr;
    }

    return make_float3(strength*xr, strength*yr, strength*zr);
}
