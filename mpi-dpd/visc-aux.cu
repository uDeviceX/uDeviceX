/* helper functions for DPD MSD calculations */
#include "visc-aux.h"
#include "common.h"

__device__ bool inbox(float x, float y, float z) {
    float xl = -3, xh = 3;
    float yl = -3, yh = 3;
    float zl = -3, zh = 3;
    return xl < x && x < xh && yl < y && y < yh  && zl < z && z < zh;
}

__device__ float3 compute_dpd_force_traced(int type1, int type2,
        float3 pos1, float3 pos2, float3 vel1, float3 vel2, float myrandnr) {
    /* return the DPD interaction force based on particle types
       type: 0 -- outer solvent, 1 -- inner solvent, 2 -- membrane, 3 -- wall */

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

    // particle type dependent constants
    const float gammadpd[4] = {56, 8, 56, 56};              // default: 4.5
    const float aij[4] = {4 / RC_FX, 4 / RC_FX, 4 / RC_FX, 4 / RC_FX}; // default: 75*kBT/numberdensity -- Groot and Warren (1997)

    const float aij_pair = 0.5 * (aij[type1] + aij[type2]);
    const float gammadpd_pair = 0.5 * (gammadpd[type1] + gammadpd[type2]);
    const float sigmaf_pair = sqrt(2*gammadpd_pair*kBT / dt);

    const float strength = aij_pair * argwr + (-gammadpd_pair * wr * rdotv + sigmaf_pair * myrandnr) * wr;

    return make_float3(strength*xr, strength*yr, strength*zr);
}
