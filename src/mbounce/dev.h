namespace mbounce {
namespace sub {
namespace dev {

template <typename Tri>
__global__ void bounce(const Force *ff, const Tri *tt, int nt, int nv, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts, const int *tids,
                       const int n, /**/ Particle *pp, Momentum *mm) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= n) return;
        
    const Particle p1 = pp[i];
            
    Particle p0; rvprev(p1.r, p1.v, ff[i].f, /**/ p0.r, p0.v);

    const int xcid = max(0, min(int (p1.r[X] + XS/2), XS-1));
    const int ycid = max(0, min(int (p1.r[Y] + YS/2), YS-1));
    const int zcid = max(0, min(int (p1.r[Z] + ZS/2), ZS-1));

    float h = 2*dt; // must be higher than any valid result
    float rw[3], vw[3];

    int icol = -1; /* id of the collision triangle */
        
    const int cid = xcid + XS * (ycid + YS * zcid);
    const int start = tcellstarts[cid];
    const int count = tcellcounts[cid];
                
    for (int j = start; j < start + count; ++j) {
        const int tid = tids[j];
        const int it  = tid % nt;
        const int mid = tid / nt;
                    
        if (find_better_intersection(tt, it, i_pp + mid * nv, &p0, /*io*/ &h, /**/ rw, vw))
            icol = tid;
    }

    if (icol != -1) {
        Particle pn;
        bounce_back(&p0, rw, vw, h, /**/ &pn);

        float dP[3], dL[3];
        lin_mom_change(    p1.v, pn.v, /**/ dP);
        ang_mom_change(rw, p1.v, pn.v, /**/ dL);

        check_pos(pn.r);
        check_vel(pn.v);
        
        pp[i] = pn;

        atomicAdd(mm[icol].P + X, dP[X]);
        atomicAdd(mm[icol].P + Y, dP[Y]);
        atomicAdd(mm[icol].P + Z, dP[Z]);

        atomicAdd(mm[icol].L + X, dL[X]);
        atomicAdd(mm[icol].L + Y, dL[Y]);
        atomicAdd(mm[icol].L + Z, dL[Z]);
    }
}

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

__global__ void collect_rbc_mom(const Momentum *mm, int nc, int nt, int nv, const int4 *tt, const Particle *pp, /**/ Force *ff) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int4 t;
    Particle pa, pb, pc;
    float fa[3], fb[3], fc[3];
    int off, ic, it;

    if (i >= nc * nt) return;

    Momentum m = mm[i];

    ic = i/nt;
    it = i%nt;
    
    if (nonzero(&m)) {
        t = tt[it];
        off = ic * nv;
                
        pa = pp[off + t.x];
        pb = pp[off + t.y];
        pc = pp[off + t.z];

        M2f(m, pa.r, pb.r, pc.r, /**/ fa, fb, fc);

        for (int c = 0; c < 3; ++c) {
            atomicAdd(ff[off + t.x].f + c, fa[c]);
            atomicAdd(ff[off + t.y].f + c, fb[c]);
            atomicAdd(ff[off + t.z].f + c, fc[c]);
        }
    }
}

} // dev
} // sub
} // mbounce
