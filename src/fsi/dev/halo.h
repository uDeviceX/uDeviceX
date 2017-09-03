namespace dev {
static __device__ Pa warp2p(const Particle *pp, int i) {
    enum {X, Y, Z};
    Pa p;
    pp += i;

     p.x = pp->r[X];  p.y = pp->r[Y];  p.z = pp->r[Z];
    p.vx = pp->v[X]; p.vy = pp->v[Y]; p.vz = pp->v[Z];
    return p;
}

static __device__ void halo0(const float *ppB, Pa A, float *fA, int nb, float seed, int aid, /**/ float *ffB) {
    enum {X, Y, Z};
    Pa B; /* remote particles */
    Fo f;

    Map m;
    int zplane;
    int i;
    int bid; /* remote particle id */
    float xforce, yforce, zforce;

    xforce = yforce = zforce = 0;
    for (zplane = 0; zplane < 3; ++zplane) {
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

static __device__ void halo1(const float *ppB, int nb, float seed, int aid, /**/ float *ffB) {
    int fid; /* fragment id */
    int start;
    Pa A;
    float *fA;

    fid = k_common::fid(g::starts, aid);
    start = g::starts[fid];

    A = warp2p(g::pp[fid], aid - start);
    fA = g::ff[fid][aid-start].f;

    halo0(ppB, A, fA, nb, seed, aid, /**/ ffB);
}

__global__ void halo(const float *ppB, int na, int nb, float seed, /**/ float *ffB) {
    int aid;
    aid = threadIdx.x + blockDim.x * blockIdx.x;
    if (aid >= na) return;
    halo1(ppB, nb, seed, aid, /**/ ffB);
}
}
