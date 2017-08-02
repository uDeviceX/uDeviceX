namespace mbounce {
namespace dev {

__global__ void bounce_tcells(const Force *ff, const Mesh m, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts, const int *tids,
                              const int n, /**/ Particle *pp, Solid *ss) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= n) return;
        
    const Particle p1 = pp[i];
            
    Particle p0; rvprev(p1.r, p1.v, ff[i].f, /**/ p0.r, p0.v);

    const int xcid = max(0, min(int (p1.r[X] + XS/2), XS-1));
    const int ycid = max(0, min(int (p1.r[Y] + YS/2), YS-1));
    const int zcid = max(0, min(int (p1.r[Z] + ZS/2), ZS-1));

    float h = 2*dt; // must be higher than any valid result
    float rw[3], vw[3];

    int sid = -1;
        
    const int cid = xcid + XS * (ycid + YS * zcid);
    const int start = tcellstarts[cid];
    const int count = tcellcounts[cid];
                
    for (int j = start; j < start + count; ++j) {
        const int tid = tids[j];
        const int it  = tid % m.nt;
        const int mid = tid / m.nt;
                    
        if (find_better_intersection(m.tt, it, i_pp + mid * m.nv, &p0, /*io*/ &h, /**/ rw, vw))
        sid = mid;
    }

    if (sid != -1) {
        Particle pn;
        bounce_back(&p0, rw, vw, h, /**/ &pn);

        float dP[3], dL[3];
        lin_mom_solid(p1.v, pn.v, /**/ dP);
        ang_mom_solid(ss[sid].com, rw, p0.v, pn.v, /**/ dL);
                
        pp[i] = pn;

        atomicAdd(ss[sid].fo + X, dP[X]);
        atomicAdd(ss[sid].fo + Y, dP[Y]);
        atomicAdd(ss[sid].fo + Z, dP[Z]);

        atomicAdd(ss[sid].to + X, dL[X]);
        atomicAdd(ss[sid].to + Y, dL[Y]);
        atomicAdd(ss[sid].to + Z, dL[Z]);
    }
}


} // dev
} // mbounce
