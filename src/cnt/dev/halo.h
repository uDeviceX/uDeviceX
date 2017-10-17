static __device__ void pp2p(Particle *pp, int i, /**/ forces::Pa *a) {
    float *r, *v;
    pp += i;
    r = pp->r;
    v = pp->v;
    forces::rvk2p(r, v, SOLID_KIND, /**/ a);
}

static __device__ void fetch_b(const float2 *pp, int i, /**/ forces::Pa *b) {
    float2 s0, s1, s2;
    pp += 3*i;
    s0 = __ldg(pp++);
    s1 = __ldg(pp++);
    s2 = __ldg(pp  );
    forces::f2k2p(s0, s1, s2, SOLID_KIND, /**/ b);
}

__device__ void halo0(forces::Pa a, int aid, float seed,
                      /**/ float *fA) {
    enum {X, Y, Z};
    Map m;
    int mapstatus;
    forces::Pa b;
    float fx, fy, fz;
    float x, y, z;
    float xforce = 0, yforce = 0, zforce = 0;
    int zplane;
    int i;
    int slot;
    int objid, bid, sentry;
    float rnd;

    forces::p2r3(&a, /**/ &x, &y, &z);
    for (zplane = 0; zplane < 3; ++zplane) {
        mapstatus = tex2map(zplane, x, y, z, /**/ &m);
        if (mapstatus == EMPTY) continue;
        for (i = 0; !endp(m, i); ++i) {
            slot = m2id(m, i);
            get(slot, &objid, &bid);
            fetch_b(c::PP[objid], bid, /**/ &b);
            rnd = rnd::mean0var1ii(seed, aid, bid);
            pair(a, b, rnd, /**/ &fx, &fy, &fz);
            xforce += fx;
            yforce += fy;
            zforce += fz;
            
            sentry = 3 * bid;
            atomicAdd(c::FF[objid] + sentry,     -fx);
            atomicAdd(c::FF[objid] + sentry + 1, -fy);
            atomicAdd(c::FF[objid] + sentry + 2, -fz);
        }
    }
    fA[X] += xforce; fA[Y] += yforce; fA[Z] += zforce;
}

__global__ void halo(int27 starts, Pap26 hpp, Fop26 hff, int n, float seed) {
    int aid, start;
    int fid;
    forces::Pa a;
    float *fA;    
    
    aid = threadIdx.x + blockDim.x * blockIdx.x;
    if (aid >= n) return;

    fid = k_common::fid(starts.d, aid);
    start = starts.d[fid];
    pp2p(hpp.d[fid], aid - start, &a);
    fA = hff.d[fid][aid - start].f;
    halo0(a, aid, seed, /**/ fA);
}
