/* assume very small portion of non zero momentum changes */
__global__ void collect_rig_mom(float dt, int ns, int nt, int nv, const int4 *tt, const Particle *pp, const Momentum *mm, /**/ Solid *ss) {
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

        m.L[X] += dr.y * m.P[Z] - dr.z * m.P[Y];
        m.L[Y] += dr.z * m.P[X] - dr.x * m.P[Z];
        m.L[Z] += dr.x * m.P[Y] - dr.y * m.P[X];

        const float fac = 1.f / dt;
        
        atomicAdd(ss[sid].fo + X, fac * m.P[X]);
        atomicAdd(ss[sid].fo + Y, fac * m.P[Y]);
        atomicAdd(ss[sid].fo + Z, fac * m.P[Z]);

        atomicAdd(ss[sid].to + X, fac * m.L[X]);
        atomicAdd(ss[sid].to + Y, fac * m.L[Y]);
        atomicAdd(ss[sid].to + Z, fac * m.L[Z]);
    }
}

static __device__ void addForce(const real3_t f, int i, Force *ff) {
    enum {X, Y, Z};
    atomicAdd(ff[i].f + X, f.x);
    atomicAdd(ff[i].f + Y, f.y);
    atomicAdd(ff[i].f + Z, f.z);
}

__global__ void collect_rbc_mom(float dt, int nc, int nt, int nv, const int4 *tt, const Particle *pp, const Momentum *mm, /**/ Force *ff) {
    int i, cid, tid;
    int4 t;
    i = threadIdx.x + blockDim.x * blockIdx.x;

    tid = i % nt;
    cid = i / nt;

    if (cid >= nc) return;

    Momentum m = mm[i];

    if (nonzero(&m)) {
        rPa A, B, C;
        real3_t fa, fb, fc;

        t = tt[tid];
        t.x += cid * nv;
        t.y += cid * nv;
        t.z += cid * nv;
        
        A = P2rP( pp + t.x );
        B = P2rP( pp + t.y );
        C = P2rP( pp + t.z );

        rbc_M2f(dt, m, A.r, B.r, C.r, /**/ &fa, &fb, &fc);

        addForce(fa, t.x, /**/ ff);
        addForce(fb, t.y, /**/ ff);
        addForce(fc, t.z, /**/ ff);
    }
}
