namespace dev {

/* assume very small portion of non zero momentum changes */
__global__ void collect_rig_mom(const Momentum *mm, int ns, int nt, /**/ Solid *ss) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int is = i / nt;

    if (i >= ns * nt) return;

    Momentum m = mm[i];
    
    if (nonzero(&m)) {

        mom_shift_ref(ss[is].com, /**/ &m); 

        const float fac = dpd_mass / dt;
        
        atomicAdd(ss[is].fo + X, fac * m.P[X]);
        atomicAdd(ss[is].fo + Y, fac * m.P[Y]);
        atomicAdd(ss[is].fo + Z, fac * m.P[Z]);

        atomicAdd(ss[is].to + X, fac * m.L[X]);
        atomicAdd(ss[is].to + Y, fac * m.L[Y]);
        atomicAdd(ss[is].to + Z, fac * m.L[Z]);
    }
}


} // dev
