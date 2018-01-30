static __device__ Pa warp2p(const Particle *pp, int i) {
    enum {X, Y, Z};
    Pa p;
    pp += i;

     p.x = pp->r[X];  p.y = pp->r[Y];  p.z = pp->r[Z];
    p.vx = pp->v[X]; p.vy = pp->v[Y]; p.vz = pp->v[Z];
    return p;
}

static __device__ void halo0(int3 L, const int *start, Pa A, int aid, Cloud cloud, int nb, float seed, /**/ float *fA, float *ffB) {
    enum {X, Y, Z};

    Pa B; /* remote particles */
    Fo f;

    Map m;
    int zplane;
    int i, bid;
    float fx, fy, fz;

    fx = fy = fz = 0;
    for (zplane = 0; zplane < 3; ++zplane) {
        if (!tex2map(L, start, zplane, nb, A.x, A.y, A.z, /**/ &m)) continue;
        for (i = 0; !endp(m, i); ++i) {
            bid = m2id(m, i);
            cloud_get(cloud, bid, /**/ &B);
            f = ff2f(ffB, bid);
            pair(A, B, random(aid, bid, seed), /**/ &fx, &fy, &fz,   f);
        }
    }
    fA[X] += fx; fA[Y] += fy; fA[Z] += fz;
}

static __device__ void halo1(int3 L, const int *cellsstart, int27 starts, Pap26 pp, Fop26 ff, int aid, Cloud cloud, int nb, float seed, /**/ float *ffB) {
    int fid; /* fragment id */
    int start;
    Pa A;
    float *fA;

    fid = fragdev::frag_get_fid(starts.d, aid);
    start = starts.d[fid];

    A = warp2p(pp.d[fid], aid - start);
    A.kind = SOLID_KIND;

    fA = ff.d[fid][aid-start].f;

    halo0(L, cellsstart, A, aid, cloud, nb, seed, /**/ fA, ffB);
}

__global__ void halo(int3 L, const int *cellsstart, int27 starts, Pap26 pp, Fop26 ff, Cloud cloud, int na, int nb, float seed, /**/ float *ffB) {
    int aid;
    aid = threadIdx.x + blockDim.x * blockIdx.x;
    if (aid >= na) return;
    halo1(L, cellsstart, starts, pp, ff, aid, cloud, nb, seed, /**/ ffB);
}
