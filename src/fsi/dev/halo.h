namespace dev {
static __device__ Pa warp2p(const Particle *pp, int i) {
    enum {X, Y, Z};
    Pa p;
    pp += i;
    
     p.x = pp->r[X];  p.y = pp->r[Y];  p.z = pp->r[Z];
    p.vx = pp->v[X]; p.vy = pp->v[Y]; p.vz = pp->v[Z];
    return p;
}

static __device__ void halo0(const float *ppB, Pa A, float *fA, int nb, float seed, int aid, int dw, int dwe, int nunpack,
                             Particle *pp, Force *ff,
                             /**/ float *ffB) {
    enum {X, Y, Z};
    Pa B; /* remote particles */
    Fo f;
    float *dst = NULL;

    Map m;
    int nzplanes;
    int zplane;
    int i;
    int bid; /* remote particle id */
    float xforce, yforce, zforce;

    xforce = yforce = zforce = 0;
    nzplanes = dw < nunpack ? 3 : 0;
    for (zplane = 0; zplane < nzplanes; ++zplane) {
        if (!tex2map(zplane, nb, A.x, A.y, A.z, /**/ &m)) continue;
        for (i = 0; !endp(m, i); ++i) {
            bid = m2id(m, i);
            pp2p(ppB, bid, /**/ &B);
            f = ff2f(ffB, bid);
            pair(A, B, random(aid, bid, seed), /**/ &xforce, &yforce, &zforce,   f);
        }
    }

    fA[X] = xforce; fA[Y] = yforce; fA[Z] = zforce;
}

static __device__ void halo1(const float *ppB, int nb, float seed, int aid, int ws, int dw, /**/ float *ffB) {
    int fid; /* fragment id */
    int start, count;
    Particle *pp;
    Force *ff;
    int nunpack, dwe;
    Pa A;
    float *fA;

    fid = k_common::fid(g::starts, ws);
    start = g::starts[fid];
    count = g::counts[fid];
    pp = g::pp[fid];
    ff = g::ff[fid];
    dwe = ws - start;
    nunpack = min(32, count - dwe);
    if (nunpack == 0) return;

    A = warp2p(pp, aid - start);
    fA = ff[aid-start].f;
    halo0(ppB, A, fA, nb, seed, aid, dw, dwe, nunpack, pp, ff, /**/ ffB);
}

__global__ void halo(const float *ppB, int n0, int nb, float seed, float *ffB) {
    int dw, warp, ws;
    int i; /* particle id */
    warp = threadIdx.x / warpSize;
    dw = threadIdx.x % warpSize;
    ws = warpSize * warp + blockDim.x * blockIdx.x;
    if (ws >= n0) return;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    halo1(ppB, nb, seed, i, ws, dw, /**/ ffB);
}
}
