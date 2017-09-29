namespace dev {

/* assume very small portion of non zero momentum changes */
__global__ void collect_rig_mom(int ns, int nt, int nv, const int4 *tt, const Particle *pp, const Momentum *mm, /**/ Solid *ss) {
    int i, sid;
    i = threadIdx.x + blockDim.x * blockIdx.x;

    sid = i / nt;

    if (sid >= ns) return;

    Momentum m = mm[i];

    if (nonzero(&m)) {
        rPa A, B, C;
        real3_t dr;
        
        fetch_triangle(i, nt, nv, tt, pp, /**/ &A, &B, &C);

        dr.x = ss[sid].com[X];
        dr.y = ss[sid].com[Y];
        dr.z = ss[sid].com[Z];
        
        dr.x -= 0.333333 * (A.r.x + B.r.x + C.r.x);
        dr.y -= 0.333333 * (A.r.y + B.r.y + C.r.y);
        dr.z -= 0.333333 * (A.r.z + B.r.z + C.r.z);

        mom_shift_ref(dr, /**/ &m); 

        const float fac = dpd_mass / dt;
        
        atomicAdd(ss[sid].fo + X, fac * m.P[X]);
        atomicAdd(ss[sid].fo + Y, fac * m.P[Y]);
        atomicAdd(ss[sid].fo + Z, fac * m.P[Z]);

        atomicAdd(ss[sid].to + X, fac * m.L[X]);
        atomicAdd(ss[sid].to + Y, fac * m.L[Y]);
        atomicAdd(ss[sid].to + Z, fac * m.L[Z]);
    }
}


} // dev
